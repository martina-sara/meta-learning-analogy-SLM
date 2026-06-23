import os
import argparse
import logging
import json
from collections import defaultdict
from typing import List, Dict, Any
from datetime import datetime
import time

import numpy as np
from tqdm import tqdm
from datasets import Dataset
from openai import AzureOpenAI

from dataset import get_dataset, ANSWER_TOKEN, STOP_TOKEN
from prompts import get_system_prompt
from utils import set_seed


def setup_args() -> argparse.Namespace:
    """Setup and return command line arguments.

    Returns:
        Namespace object containing all runtime arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--azure_endpoint", type=str, required=True,
        help="Azure OpenAI API endpoint."
    )
    parser.add_argument(
        "--api_version", type=str, default="2025-03-01-preview",
        help="Azure OpenAI API version."
    )
    parser.add_argument(
        "--model", type=str, default="o3-mini",
        help="Name of the commercial model to evaluate (label used in results)."
    )
    parser.add_argument(
        "--deployment", type=str, default="o3-mini-b",
        help="Name of the AzureOpenAI deployment to call."
    )
    parser.add_argument("--seed", type=int, default=1048, help="Random seed for reproducibility.")
    parser.add_argument(
        "--dataset", type=str, default="meta", choices=["base", "meta"],
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
        "--max_retries", type=int, default=5,
        help="Maximum number of retries for API calls."
    )
    parser.add_argument(
        "--retry_delay", type=int, default=3,
        help="Delay between retries in seconds."
    )

    args = parser.parse_args()

    # Derived attributes
    args.ft_type = "api"
    args.model_name = f"{args.model}_{args.ft_type}_{args.dataset}"

    # System prompt (shared with the local prompting baseline in test.py)
    args.system = get_system_prompt(args.dataset)
    return args


def extract_predictions(text: str) -> str:
    """Extract the term from a model's '### Answer: term' output.

    Args:
        text: Raw model output text

    Returns:
        Cleaned prediction string
    """
    lower = text.lower()
    if "### answer:" in lower:
        tail = text[lower.index("### answer:") + len("### answer:"):]
        pred = tail.strip().splitlines()[0].strip() if tail.strip() else ""
    else:
        logging.warning(f"Answer format not found in model output: '{text}'")
        pred = text.strip()
    return pred


def _norm(s: Any) -> str:
    """Normalize a string by lowercasing and collapsing whitespace.

    Args:
        s: Value to normalize (cast to string if needed)

    Returns:
        The lowercased, whitespace-collapsed string
    """
    return " ".join(str(s).strip().lower().split())


def evaluate_against_targets(preds: List[str], targets: List[List[str]]) -> List[int]:
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
    """Append the headline accuracy to a CSV summary.

    Args:
        folder: Directory to save results
        save_file: Name of the CSV file
        overall_accuracy: Overall accuracy across all test episodes
        args: Runtime arguments
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, save_file)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("model,seed,ft_type,setting,accuracy\n")
    with open(path, "a") as f:
        f.write(f"{args.model},{args.seed},{args.ft_type},{args.dataset},{overall_accuracy}\n")


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
    json_file = os.path.join(log_dir, f"{args.model_name}.json")
    payload = {
        "overall_accuracy": overall,
        "per_relation_accuracy": per_relation,
        "n_errors": len(errors),
        "errors": errors,
    }
    with open(json_file, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def prepare_batch_data(test_data: Dataset, args: argparse.Namespace) -> str:
    """Convert dataset to the JSONL format required for batch processing.

    Args:
        test_data: Dataset containing test examples to process
        args: Runtime arguments containing model configuration

    Returns:
        str: Path to the created batch file
    """
    os.makedirs("data", exist_ok=True)
    batch_file = f"data/batch_{args.dataset}.jsonl"

    with open(batch_file, "w") as f:
        for idx, item in enumerate(test_data):
            batch_item = {
                "custom_id": f"task-{idx}",
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": args.deployment,
                    "messages": [
                        {"role": "system", "content": args.system},
                        {"role": "user", "content": item["input"]},
                    ],
                },
            }
            f.write(json.dumps(batch_item) + "\n")

    return batch_file


def monitor_batch_status(client: AzureOpenAI, batch_id: str) -> str:
    """Monitor the status of a batch job until completion.

    Args:
        client: Azure OpenAI client instance
        batch_id: ID of the batch job to monitor

    Returns:
        str: Final status of the batch job

    Raises:
        Exception: If batch processing fails
    """
    status = "none"
    last_status = status

    while status not in ("completed", "failed", "canceled"):
        time.sleep(30)
        batch_response = client.batches.retrieve(batch_id)
        status = batch_response.status

        if status != last_status:
            print(f"{datetime.now()} Batch Id: {batch_id}, Status: {status}")
            last_status = status

    if status == "failed":
        for error in batch_response.errors.data:
            logging.error(f"Error code {error.code} Message {error.message}")
        raise Exception("Batch processing failed")

    return status


def process_batch_results(
    results_file: str,
    test_data: Dataset,
    args: argparse.Namespace,
):
    """Process batch results from LLM responses and compute accuracy metrics.

    Extracts a prediction per episode from the raw responses, scores each against
    its accepted target list, and aggregates overall and per-relation accuracy.

    Args:
        results_file: Path to the file containing raw LLM responses
        test_data: Dataset containing test examples and ground truth
        args: Command line arguments

    Returns:
        Tuple of (overall_accuracy, per_relation_accuracy, errors)
    """
    with open(results_file, "r") as f:
        raw_responses = f.read().strip().split("\n")

    # Align predictions back to dataset order via custom_id.
    predictions = ["None"] * len(test_data)
    for raw_response in raw_responses:
        if not raw_response:
            continue
        json_response = json.loads(raw_response)
        idx = int(json_response["custom_id"].split("-")[1])
        try:
            text_response = json_response["response"]["body"]["choices"][0]["message"]["content"]
        except (KeyError, TypeError):
            logging.warning(f"No answer found in response for id {idx}")
            text_response = "None"
        predictions[idx] = extract_predictions(text_response)

    per_relation: Dict[str, List[int]] = defaultdict(list)
    all_correct: List[int] = []
    errors: List[Dict[str, Any]] = []

    for pred, item in zip(predictions, test_data):
        correct = evaluate_against_targets([pred], [item["target"]])[0]
        all_correct.append(correct)
        per_relation[item["relation"]].append(correct)
        if not correct:
            errors.append({
                "episode_id": item["episode_id"],
                "relation": item["relation"],
                "query": item["query"],
                "prediction": pred,
                "target": item["target"],
            })

    overall = round(np.mean(all_correct) * 100, 2) if all_correct else 0.0
    per_relation_acc = {
        rel: round(np.mean(v) * 100, 2) for rel, v in sorted(per_relation.items())
    }
    return overall, per_relation_acc, errors


def batch_inference(client: AzureOpenAI, test_data: Dataset, args: argparse.Namespace):
    """Execute the complete batch inference workflow.

    Args:
        client: Azure OpenAI client instance
        test_data: Dataset containing test examples
        args: Runtime arguments containing configuration

    Returns:
        Tuple of (overall_accuracy, per_relation_accuracy, errors)

    Raises:
        Exception: If batch processing fails or output is missing
    """
    # Prepare batch data
    batch_file = prepare_batch_data(test_data, args)

    # Upload file
    file = client.files.create(
        file=open(batch_file, "rb"),
        purpose="batch",
        extra_body={"expires_after": {"seconds": 1209600, "anchor": "created_at"}},
    )

    # Create batch job
    batch_response = client.batches.create(
        input_file_id=file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    # Monitor progress
    status = monitor_batch_status(client, batch_response.id)
    batch_response = client.batches.retrieve(batch_response.id)

    if status == "completed":
        output_file_id = batch_response.output_file_id
        if output_file_id:
            file_response = client.files.content(output_file_id)
            output_dir = os.path.join("results", "analogy", "api")
            os.makedirs(output_dir, exist_ok=True)
            results_file = os.path.join(output_dir, f"outputs_{args.model_name}.jsonl")
            with open(results_file, "w") as f:
                f.write(file_response.text)

            return process_batch_results(results_file, test_data, args)

    raise Exception("No output file found")


def main(args: argparse.Namespace) -> None:
    """Main execution function.

    Args:
        args: Runtime arguments containing model and API configuration
    """
    set_seed(args.seed)

    # Build splits; only the held-out-relation test split is evaluated.
    _, _, test = get_dataset(
        args.dataset,
        train_path=args.train_path,
        test_path=args.test_path,
        dev_per_relation=0,
        subsample_train=None,
        seed=args.seed,
        print_info=False,
    )

    # Initialize client
    client = AzureOpenAI(
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    # Run batch inference
    overall, per_relation, errors = batch_inference(client, test, args)

    # Save results
    tqdm.write(f"Accuracy = {overall}")
    save_folder = os.path.join("results", "analogy")
    save_results(save_folder, "results_api.csv", overall, args)
    save_breakdown(save_folder, args, overall, per_relation, errors)


if __name__ == "__main__":
    args = setup_args()
    main(args)
