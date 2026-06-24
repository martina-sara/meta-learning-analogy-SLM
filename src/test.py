import os
import argparse
import logging
import json
from collections import defaultdict
from typing import List, Dict, Any, Iterator

import numpy as np
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from dataset import get_dataset, ANSWER_TOKEN, STOP_TOKEN
from prompts import get_system_prompt
from utils import (
    load_best_trained_model,
    get_model_id,
    set_seed,
)


def setup_args() -> argparse.Namespace:
    """Setup and return command line arguments.
    
    Returns:
        Namespace object containing all runtime arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--cache_dir", type=str, help="Directory to store cached models.")
    parser.add_argument(
        "--save_model_dir", type=str, default="analogy-llms",
        help="Directory where trained models were saved."
    )
    parser.add_argument("--seed", type=int, default=1048, help="Random seed for reproducibility.")
    parser.add_argument("--model", type=str, default="qwen2.5-7b", help="Name of the model to use.")
    parser.add_argument(
        "--ft_type", type=str, default="full", choices=["full", "lora", "none"],
        help="Fine-tuning type of the model being evaluated. 'none' = prompt the base model."
    )
    parser.add_argument(
        "--test_model_type", type=str, default="base", choices=["base", "meta"],
        help="Condition the model was trained on (used to locate the saved checkpoint)."
    )
    parser.add_argument(
        "--dataset", type=str, default="base", choices=["base", "meta"],
        help="Condition to evaluate on: 'base' (query only) or 'meta' (study + query)."
    )
    parser.add_argument(
        "--train_path", type=str, default="data/train_episodes.jsonl",
        help="Path to the training-episode .jsonl file (needed to build the splits)."
    )
    parser.add_argument(
        "--test_path", type=str, default="data/test_episodes.jsonl",
        help="Path to the test-episode .jsonl file (disjoint relations)."
    )
    parser.add_argument(
        "--subsample_train", type=int, default=None,
        help="Subsample value used at train time (only used to locate the checkpoint name)."
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on (cuda/cpu).")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum number of new tokens to generate.")

    args = parser.parse_args()

    # Derived attributes
    args.model_id = get_model_id(args.model)
    args.model_name = f"{args.model}_{args.ft_type}_{args.test_model_type}_seed_{args.seed}"
    if args.subsample_train:
        args.model_name += f"_{args.subsample_train}"

    # System prompt used when ft_type == "none"
    args.system = get_system_prompt(args.dataset)
    return args


def batches(lst: Any, n: int) -> Iterator[Any]:
    """Yield successive n-sized batches from lst.
    
    Args:
        lst: Input list to be batched
        n: Batch size
    
    Yields:
        Batches of size n from the input list
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


@torch.inference_mode()
def get_model_predictions(
    model: Any,
    tokenizer: Any,
    batch: Dict[str, List],
    max_new_tokens: int,
) -> List[str]:
    """Generate predictions from the model for a batch of inputs.

    Args:
        model: The pretrained model to use for inference
        tokenizer: Tokenizer for processing input text
        batch: Dictionary containing input data for the batch
        max_new_tokens: Maximum number of new tokens to generate

    Returns:
        List of decoded model outputs as strings
    """
    prompts = [inp + f" {ANSWER_TOKEN}" for inp in batch["input"]]
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


@torch.inference_mode()
def get_model_completions_from_prompts(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    max_new_tokens: int,
) -> List[str]:
    """Generate completion for each prompt.

    Args:
        model: The pretrained model to use for inference.
        tokenizer: Tokenizer for processing input prompts.
        prompts: Fully rendered prompts (including system + user content).
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        List of decoded model completions as strings.

    Raises:
        ValueError: If the tokenizer does not return an attention mask.
    """
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )

    attn = inputs.get("attention_mask")
    if attn is None:
        raise ValueError("Expected tokenizer to return 'attention_mask' for completion slicing")

    completions: List[str] = []
    for i in range(outputs.shape[0]):
        prompt_len = int(attn[i].sum().item())
        completion_ids = outputs[i, prompt_len:]
        completions.append(tokenizer.decode(completion_ids, skip_special_tokens=True))
    return completions


def extract_predictions(texts: List[str]) -> List[str]:
    """Extract clean predictions from model outputs by removing formatting.
    
    Args:
        texts: Raw model output texts
        
    Returns:
        List of cleaned prediction strings
    """
    preds: List[str] = []
    for text in texts:
        parts = text.split(ANSWER_TOKEN)
        if len(parts) > 1:
            pred = parts[1].split(STOP_TOKEN)[0].strip()
        else:
            logging.warning(f"'{ANSWER_TOKEN}' not found in model output: '{text}'")
            pred = text.strip()
        preds.append(pred)
    return preds


def build_prompts_for_batch(
    batch: Dict[str, List],
    args: argparse.Namespace,
    tokenizer: Any,
) -> List[str]:
    """Build prompts for the non-fine-tuned baseline model.

    Args:
        batch: Batch dictionary from the HuggingFace Dataset.
        args: Runtime arguments.
        tokenizer: Tokenizer used to render the chat template.

    Returns:
        List of rendered prompts (strings) matching the API experiment format.

    Raises:
        ValueError: If the tokenizer does not support chat template rendering.
    """
    system = args.system
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Expected a chat model tokenizer with apply_chat_template()")

    prompts: List[str] = []
    for user_content in batch["input"]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    return prompts


def extract_predictions_from_prompted_model(texts: List[str]) -> List[str]:
    """Extract predictions from the answer format of a prompted model.
    Expected format:
        "### Answer: term' output.
    
    Args:
        texts: Model outputs/completions.

    Returns:
        List of cleaned prediction strings.
    """
    preds: List[str] = []
    for text in texts:
        lower = text.lower()
        if "### answer:" in lower:
            # Take the first answer line only.
            tail = text[lower.index("### answer:") + len("### answer:"):]
            pred = tail.strip().splitlines()[0].strip() if tail.strip() else ""
        else:
            logging.warning(f"Answer format not found in model output: '{text}'")
            pred = text.strip()
        preds.append(pred)
    return preds


def _norm(s: Any) -> str:
    """Normalize a string by lowercasing and collapsing whitespace.

    Args:
        s: Value to normalize (cast to string if needed)

    Returns:
        The lowercased, whitespace-collapsed string
    """
    return " ".join(str(s).strip().lower().split())


def evaluate_against_targets(
    preds: List[str],
    targets: List[List[str]],
) -> List[int]:
    """Evaluate the model output by comparing each prediction to its accepted targets.

    A prediction is correct if its normalized form matches any of the accepted
    target strings for that episode (case- and whitespace-insensitive).

    Args:
        preds: The model predictions
        targets: The accepted target strings for each episode (one list per prediction)

    Returns:
        A list of 1s and 0s, where 1 indicates that the model prediction is correct and 0 otherwise
    """
    out: List[int] = []
    for pred, tgt in zip(preds, targets):
        gold = tgt if isinstance(tgt, list) else [tgt]
        gold_set = {_norm(t) for t in gold}
        out.append(1 if _norm(pred) in gold_set else 0)
    return out


def save_results(
    folder: str,
    save_file: str,
    overall_accuracy: float,
    args: argparse.Namespace,
) -> None:
    """Save evaluation results to a CSV file.

    Args:
        folder: Directory to save results
        save_file: Name of the CSV file
        accuracy_dict: Dictionary containing accuracy metrics
        args: Runtime arguments
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, save_file)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("model,seed,ft_type,model_type,dataset,subsample_train,accuracy\n")
    with open(path, "a") as f:
        f.write(
            f"{args.model},{args.seed},{args.ft_type},{args.test_model_type},"
            f"{args.dataset},{args.subsample_train},{overall_accuracy}\n"
        )


def save_breakdown(
    folder: str,
    args: argparse.Namespace,
    overall: float,
    per_relation: Dict[str, float],
    errors: List[Dict[str, Any]],
) -> None:
    """Save per-relation accuracy and the list of wrong predictions to a JSON file.

    Args:
        folder: Directory to save results
        args: Runtime arguments
        overall: Overall accuracy across all test episodes
        per_relation: Mapping from relation name to its accuracy
        errors: List of wrong predictions, one dict per misclassified episode
    """
    log_dir = os.path.join(folder, "errors")
    os.makedirs(log_dir, exist_ok=True)
    json_file = os.path.join(
        log_dir, f"{args.model_name}_eval_{args.dataset}.json"
    )
    payload = {
        "overall_accuracy": overall,
        "per_relation_accuracy": per_relation,
        "n_errors": len(errors),
        "errors": errors,
    }
    with open(json_file, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def test_loop(
    model: Any,
    tokenizer: Any,
    test_data: Dataset,
    args: argparse.Namespace,
):
    """Run evaluation loop on test data.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for processing text
        test_data: Dataset containing test examples
        args: Runtime arguments and configuration
    
    Returns:
        Dictionary containing accuracy metrics (core and by type/length)
    """
    per_relation: Dict[str, List[int]] = defaultdict(list)
    all_correct: List[int] = []
    errors: List[Dict[str, Any]] = []

    total = max(1, len(test_data) // args.batch_size)
    for batch in tqdm(batches(test_data, args.batch_size), desc="Test", leave=False, total=total):
        if args.ft_type == "none":
            prompts = build_prompts_for_batch(batch, args, tokenizer)
            completions = get_model_completions_from_prompts(model, tokenizer, prompts, args.max_new_tokens)
            preds = extract_predictions_from_prompted_model(completions)
        else:
            texts = get_model_predictions(model, tokenizer, batch, args.max_new_tokens)
            preds = extract_predictions(texts)

        correct = evaluate_against_targets(preds, batch["target"])
        all_correct += correct

        for c, rel in zip(correct, batch["relation"]):
            per_relation[rel].append(c)

        for i, c in enumerate(correct):
            if not c:
                errors.append({
                    "episode_id": batch["episode_id"][i],
                    "relation": batch["relation"][i],
                    "query": batch["query"][i],
                    "prediction": preds[i],
                    "target": batch["target"][i],
                })

    overall = round(np.mean(all_correct) * 100, 2) if all_correct else 0.0
    per_relation_acc = {
        rel: round(np.mean(v) * 100, 2) for rel, v in sorted(per_relation.items())
    }
    return overall, per_relation_acc, errors


def main(args: argparse.Namespace) -> None:
    """Main execution function.
    
    Args:
        args: Runtime arguments containing model and training configuration
    """
    set_seed(args.seed)

    # Build splits; we only use the (held-out-relation) test split here.
    _, _, test = get_dataset(
        args.dataset,
        train_path=args.train_path,
        test_path=args.test_path,
        dev_per_relation=0,
        subsample_train=args.subsample_train,
        seed=args.seed,
        print_info=False,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        padding_side="left",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Model
    if args.ft_type == "none":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        model.to(args.device)
        model.eval()
    else:
        model = load_best_trained_model(
            model_id=args.model_id,
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            save_model_dir=args.save_model_dir,
            device=args.device,
            ft_type=args.ft_type,
        )

    # Evaluate
    overall, per_relation, errors = test_loop(model, tokenizer, test, args)
    tqdm.write(f"Accuracy = {overall}")

    save_folder = os.path.join("results", "analogy")
    save_results(save_folder, "results.csv", overall, args)
    save_breakdown(save_folder, args, overall, per_relation, errors)


if __name__ == "__main__":
    args = setup_args()
    main(args)
