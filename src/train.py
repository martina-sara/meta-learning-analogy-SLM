import os
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar

from dataset import get_dataset, COLUMNS, ANSWER_TOKEN
from utils import (
    set_seed,
    load_model,
    get_model_id,
    mask_labels_for_completion,
)
from test import get_model_predictions, extract_predictions, evaluate_against_targets


COMPLETION_BOUNDARY = f" {ANSWER_TOKEN}"


def setup_accelerator() -> Accelerator:
    """Initialize and return the Accelerator object for distributed training.

    Returns:
        Accelerator: Configured accelerator instance
    """
    accelerator = Accelerator()
    return accelerator


def setup_args(accelerator: Accelerator) -> argparse.Namespace:
    """Set up and parse command line arguments.
    
    Args:
        accelerator: The Accelerator instance for getting device information

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    # Saving dirs
    parser.add_argument("--cache_dir", type=str, help="Directory to store cached models.")
    parser.add_argument(
        "--save_model_dir", type=str, default="analogy-llms",
        help="Directory to save trained models."
    )
    parser.add_argument("--seed", type=int, default=1048, help="Random seed for reproducibility.")
    # Model
    parser.add_argument("--model", type=str, default="qwen2.5-7b", help="Name of the model to use.")
    parser.add_argument(
        "--ft_type", type=str, default="full", choices=["full", "lora"],
        help="Type of fine-tuning to perform (full or lora)."
    )
    parser.add_argument(
        "--precision", type=str, default="bfloat16", choices=["bfloat16", "int8", "int4"],
        help="Precision for model training (bfloat16, int8, or int4)."
    )
    # Dataset / condition
    parser.add_argument(
        "--dataset", type=str, default="base", choices=["base", "meta"],
        help="Training condition: 'base' (query only) or 'meta' (study + query in context)."
    )
    parser.add_argument(
        "--train_path", type=str, default="data/train_episodes.jsonl",
        help="Path to the training-episode .jsonl file."
    )
    parser.add_argument(
        "--test_path", type=str, default="data/test_episodes.jsonl",
        help="Path to the test-episode .jsonl file (disjoint relations)."
    )
    parser.add_argument(
        "--dev_per_relation", type=int, default=2,
        help="Episodes per relation held out of train as an in-distribution dev set."
    )
    parser.add_argument(
        "--subsample_train", type=int, default=None,
        help="Cap on training episodes per relation (low-data regime). None = use all."
    )
    # Optimizer args
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for the scheduler.")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Ratio of total steps used as warmup.")
    # Training args
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs.")
    parser.add_argument("--val_per_epoch", type=int, default=10, help="Number of validation steps per epoch.")
    parser.add_argument("--seq_len", type=int, default=2048, help="Maximum sequence length for the model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--val_batch_size", type=int, default=64, help="Batch size for validation.")
    # Log validation
    parser.add_argument("--log", action=argparse.BooleanOptionalAction, help="Whether to log validation predictions.")
    args = parser.parse_args()

    # Derived arguments
    args.device = accelerator.device
    args.model_id = get_model_id(args.model)
    args.model_name = f"{args.model}_{args.ft_type}_{args.dataset}_seed_{args.seed}"
    if args.subsample_train:
        args.model_name += f"_{args.subsample_train}"

    return args


def setup_tokenizer(args: argparse.Namespace) -> Any:
    """Initialize and configure the tokenizer for the specified model.
    
    Args:
        args: Parsed command line arguments

    Returns:
        Any: Configured tokenizer for the model
    """
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        padding_side="left",
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.seq_len
    return tokenizer


def preprocess_function_lm(
    examples: Dict[str, List[str]],
    return_tensors: Optional[str] = None
) -> Dict[str, List[int]]:
    """Preprocess examples for language modeling.
    
    Args:
        examples: Dictionary containing input and output text pairs

    Returns:
        Dict[str, List[int]]: Tokenized and processed inputs
    """
    strings = [i+o for i,o in zip(examples["input"], examples["output"])]
    return tokenizer(strings, padding=True, truncation=True, max_length=args.seq_len, return_tensors=return_tensors)


def batches(lst: List[Any], n: int) -> List[Any]:
    """Split a list into batches of size n.
    
    Args:
        lst: List to be batched
        n: Batch size

    Yields:
        List[Any]: Batch of elements from the input list
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def log_predictions(
    preds: List[str],
    queries: List[str],
    targets: List[List[str]],
    texts: List[str],
) -> None:
    """Log detailed information about predictions for debugging.
    
    Args:
        preds: List of model predictions
        queries: List of query hypotheses
        targets: List of ground truth outputs
        texts: List of raw model output texts
    """
    for p, q, t, raw in zip(preds, queries, targets, texts):
        tqdm.write("-" * 50)
        tqdm.write("RAW OUTPUT")
        tqdm.write(raw)
        tqdm.write("QUERY")
        tqdm.write(str(q))
        tqdm.write("PREDICTED:")
        tqdm.write(p)
        tqdm.write("TARGET(S):")
        tqdm.write(str(t))


def log_training_details(
    file_path: str,
    epoch: int,
    step: int,
    val_loss: float,
    val_accuracy: float,
) -> None:
    """Log training details to a file.
        
    Args:
        file_path: Path to the log file
        epoch: Current epoch number
        step: Current iteration number
        val_loss: Validation loss
        val_accuracy: Validation accuracy
    """
    tqdm.write("epoch = {}\t|\titer = {}\t|\tval_loss = {}\t|\teval/acc = {}".format(
        epoch, step, val_loss, val_accuracy))
    with open(file_path, "a") as f:
        f.write(f"{epoch},{step},{val_loss},{val_accuracy}\n")


def compute_validation_loss(
    model: Any,
    tokenizer: Any,
    batch: Dict[str, Any],
) -> float:
    """Compute validation loss for a batch (loss only on the answer span).
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        batch: Dictionary containing input and output text pairs

    Returns:
        float: Validation loss for the batch
    """
    model_inputs = preprocess_function_lm(batch, return_tensors="pt")
    labels = model_inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    model_inputs = mask_labels_for_completion(model_inputs, COMPLETION_BOUNDARY, tokenizer)

    for k, v in model_inputs.items():
        if not isinstance(v, torch.Tensor):
            model_inputs[k] = torch.tensor(v).to(args.device)
        else:
            model_inputs[k] = v.to(args.device)

    outputs = model(**model_inputs)
    return outputs.loss.item()


@torch.inference_mode()
def validation_loop(
    model: Any,
    tokenizer: Any,
    val_data: Dataset,
) -> (float, float):
    """Run validation loop to evaluate model performance. 
    Returns both validation accuracy and loss.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        val_data: Validation dataset

    Returns:
        float: Validation accuracy
        float: Validation loss
    """
    if len(val_data) == 0:
        return 0.0, 0.0

    accuracy: List[int] = []
    loss_total = 0.0
    count = 0

    with accelerator.autocast():
        dev_bar = tqdm(total=max(1, len(val_data) // args.val_batch_size), desc="Validation", leave=False)
        for batch in batches(val_data, args.val_batch_size):
            # Generation budget = longest gold output in the batch.
            max_new_tokens = max(len(tokenizer.encode(o)) for o in batch["output"])

            texts = get_model_predictions(model, tokenizer, batch, max_new_tokens)
            preds = extract_predictions(texts)
            accuracy += evaluate_against_targets(preds, batch["target"])

            loss = compute_validation_loss(model, tokenizer, batch)
            loss_total += loss
            count += 1

            if args.log:
                log_predictions(preds, batch["query"], batch["target"], texts)

            dev_bar.update(1)

    avg_accuracy = round(np.mean(accuracy) * 100, 2) if accuracy else 0.0
    avg_loss = loss_total / count if count > 0 else 0.0
    return avg_accuracy, avg_loss


def calculate_warmup_steps(total_steps: int, warmup_steps: int, warmup_ratio: float) -> int:
    """Calculate the number of warmup steps based on either explicit steps or ratio.
    
    Args:
        total_steps: Total number of training steps
        warmup_steps: Explicit number of warmup steps (takes precedence if > 0)
        warmup_ratio: Ratio of total steps to use for warmup
        
    Returns:
        int: Number of warmup steps to use
    """
    if warmup_steps > 0:
        return warmup_steps
    return int(total_steps * warmup_ratio)


def main() -> None:
    """Main training function that handles the complete training pipeline."""
    set_seed(args.seed)

    # Save paths
    save_dir = os.path.join(args.save_model_dir, args.model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Train log dir and file
    log_dir = os.path.join("results", "analogy", "train_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{args.model_name}.csv")
    with open(log_file_path, "w") as f:
        f.write("epoch,iter,val_loss,eval/acc\n")

    # Load dataset
    if not accelerator.is_main_process:
        disable_progress_bar()
    print_info = accelerator.is_main_process
    train, dev, test = get_dataset(
        args.dataset,
        train_path=args.train_path,
        test_path=args.test_path,
        dev_per_relation=args.dev_per_relation,
        subsample_train=args.subsample_train,
        seed=args.seed,
        print_info=print_info,
    )

    # Tokenize the training split.
    train_tokenized = train.map(
        preprocess_function_lm,
        batched=True,
        remove_columns=COLUMNS,
    )
    train_tokenized.set_format("torch")

    # Model
    model = load_model(args.model_id, args.cache_dir, device=args.device, ft_type=args.ft_type, precision=args.precision)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Dataloader
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(train_tokenized, collate_fn=data_collator, batch_size=args.batch_size)

    # Training parameters
    epoch_len = len(train_dataloader)
    total_steps = epoch_len * args.epochs
    save_every_iter = epoch_len // args.val_per_epoch

    num_warmup_steps = calculate_warmup_steps(epoch_len, args.warmup_steps, args.warmup_ratio)
    
    # Create scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    # Accelerator wrap
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

    # Training loop
    model.train()
    best_accuracy = -1.0
    best_epoch = 0
    best_iter = 0
    best_saved = False

    if accelerator.is_main_process:
        epoch_bar = tqdm(total=args.epochs, desc="Epoch", leave=False)

    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            step_bar = tqdm(total=epoch_len, desc=f"Training epoch {epoch}", leave=False)

        for step, batch in enumerate(train_dataloader):
            # Mask everything up to and including <ANSWER>; loss only on the answer.
            batch = mask_labels_for_completion(batch, COMPLETION_BOUNDARY, tokenizer)
            batch = {k: v.to(args.device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Periodic validation + best-model checkpointing.
            if step % save_every_iter == 0:
                if accelerator.is_main_process and len(dev) > 0:
                    unwrapped_model = accelerator.unwrap_model(model)
                    val_accuracy, val_loss = validation_loop(unwrapped_model, tokenizer, dev)
                    global_step = step + epoch * epoch_len

                    log_training_details(log_file_path, epoch, global_step, val_loss, val_accuracy)

                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_iter = global_step
                        best_epoch = epoch
                        save_path = os.path.join(save_dir, f"{args.model_name}_best")
                        unwrapped_model.save_pretrained(
                            save_path,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                        )
                        best_saved = True

                accelerator.wait_for_everyone()

            model.train()

            if accelerator.is_main_process:
                step_bar.update(1)

        if accelerator.is_main_process:
            epoch_bar.update(1)

    # Guarantee a checkpoint exists for testing even if there is no dev set
    # (dev_per_relation == 0) or dev accuracy never improved above the init value.
    if accelerator.is_main_process and not best_saved:
        unwrapped_model = accelerator.unwrap_model(model)
        save_path = os.path.join(save_dir, f"{args.model_name}_best")
        unwrapped_model.save_pretrained(
            save_path,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        tqdm.write("No dev improvement recorded; saved final-epoch model as best.")

    if accelerator.is_main_process:
        tqdm.write("-" * 50)
        tqdm.write("BEST MODEL:")
        tqdm.write("epoch = {}\t|\titer = {}\t|\tdev/acc = {}".format(best_epoch, best_iter, best_accuracy))


if __name__ == "__main__":
    accelerator = setup_accelerator()
    args = setup_args(accelerator)
    tokenizer = setup_tokenizer(args)
    main()
