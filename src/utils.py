import os
import logging
import glob
import random
from typing import List, Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from transformers import BitsAndBytesConfig

from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig


def set_seed(seed: int) -> None:
    """
    Set the seed for random number generation in torch, numpy, and random libraries.

    Args:
        seed: The seed value to set for random number generation.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_model_id(model: str) -> str:
    """
    Get the model id from the model name.

    Args:
        model: The simplified model name.

    Returns:
        The hf model id.
    """
    id_dict = {
        "qwen-1.5b" : "Qwen/Qwen2.5-1.5B",
        "qwen-3b" : "Qwen/Qwen2.5-3B",
        "qwen-7b" :  "Qwen/Qwen2.5-7B",
    }
    return id_dict[model]


def load_model(
    model_id: str,
    cache_dir: str,
    device: str = "cuda",
    ft_type: str = "full",
    precision: str = "bfloat16"
) -> Any:
    """
    Load the pre-trained model and adapter in case of LoRA fine-tuning.

    Args:
        model_id: The model id.
        cache_dir: The cache directory.
        device: The device to use.
        ft_type: The fine-tuning type.
        precision: The precision format ("half", "int8", "int4").

    Returns:
        The pre-trained model.
    """
    # Prepare quantization config based on precision
    quantization_config = None
    dtype = None
    
    if precision == "int8":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif precision == "int4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif precision == "bfloat16":
        dtype = torch.bfloat16
    
    # Load the pre-trained model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        attn_implementation="sdpa",
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=dtype
    )
    
    if ft_type == "lora":
        # Prepare quantized model
        if precision in ["int4", "int8"]:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        # Define the LoraConfig
        target_modules = (
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        task_type = "CAUSAL_LM"
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            bias="none",
            target_modules=target_modules,
            task_type=task_type,
        )
        # Load adapter layers
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if ft_type == "full":
        ft_dir = os.path.join(save_model_dir, model_name)
        ft_path = os.path.join(ft_dir, f"{model_name}_best")
        if not os.path.exists(ft_path):
            raise FileNotFoundError(
                f"No checkpoint at {ft_path}. Check --save_model_dir / --model / "
                f"--test_model_type / --seed match the training run."
            )
        model = AutoModelForCausalLM.from_pretrained(
            ft_path,
            attn_implementation="eager",
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()

    else:
        raise ValueError("ft_type must be 'full' or 'lora'")

    if precision not in ["int4", "int8"]:
        model.to(device)
    
    return model


def load_best_trained_model(
    model_id: str = None,
    model_name: str = None,
    cache_dir: str = None,
    save_model_dir: str = None,
    device: str = "cuda",
    ft_type: str = "full"
) -> Any:
    """
    Load the best trained model for evaluation.

    Args:
        model_id: The model id.
        model_name: The model name.
        cache_dir: The cache directory.
        save_model_dir: The save model directory.
        device: The device to use.
        ft_type: The fine-tuning type.
    
    Returns:
        The best trained model.
    """

    if ft_type == "full":
        ft_dir = os.path.join(save_model_dir, model_name)
        ft_path = os.path.join(ft_dir, f"{model_name}_best")
        model = AutoModelForCausalLM.from_pretrained(
            ft_path,
            attn_implementation="eager",
            trust_remote_code=True,
        )
        # Move model to device and eval mode
        model.to(device)
        model.eval()

    elif ft_type == "lora":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            attn_implementation="eager",
            trust_remote_code=True,
        )
        # Define the adapter path and load adapter
        adapter_dir = os.path.join(save_model_dir, model_name)
        adapter_path = os.path.join(adapter_dir, f"{model_name}_best")
        if not os.path.exists(adapter_path):
            print("Best model not found!")
            subfolders = sorted(glob.glob(os.path.join(adapter_dir, '*/')))
            idx_sub = int(len(subfolders) * 0.7)
            adapter_path = subfolders[idx_sub]
            print(f"Loading {adapter_path} instead")
        model.load_adapter(adapter_path)
        # Move model to device and eval mode
        model.to(device)
        model.eval()

    else:
        raise ValueError("ft_type must be 'full' or 'lora'")
    
    return model


def mask_labels_for_completion(batch, completion_start_text, tokenizer):
    """
    Masks labels in a batch up to (and including) the first occurrence of a tokenized prompt text.
    
    Given a batch with 'labels' tensor, this function tokenizes the completion_start_text and searches
    for this token sequence in each sample's labels. All tokens preceding and including the found sequence
    are set to -100 to ignore them in loss calculation. If the prompt is not found, a warning is logged.
    
    Args:
        batch: Dictionary containing a 'labels' key with a torch.Tensor of shape (batch_size, sequence_length).
        completion_start_text: Text whose tokenized form marks the start of the completion.
        tokenizer: A tokenizer with a __call__ method returning a dict with 'input_ids' (e.g., HuggingFace tokenizer).
    
    Returns:
        dict: The modified batch with masked labels. Samples where the prompt isn't found remain unchanged.
    """
    def find_subsequence(sequence, subsequence):
        """Helper to find the start index of a contiguous subsequence within a sequence, returns -1 if not found."""
        sub_len = len(subsequence)
        seq_len = len(sequence)
        for i in range(seq_len - sub_len + 1):
            if sequence[i:i+sub_len] == subsequence:
                return i
        return -1

    # Tokenize the prompt without special tokens
    prompt_ids = tokenizer(completion_start_text, add_special_tokens=False)['input_ids']
    
    # Clone labels to avoid modifying the original tensor
    new_labels = batch['labels'].clone()
    
    for i in range(new_labels.shape[0]):
        label_seq = new_labels[i].tolist()
        match_idx = find_subsequence(label_seq, prompt_ids)
        
        if match_idx == -1:
            logging.warning(f"Completion IDs '{prompt_ids}' not found in sample {label_seq}. Labels unchanged.")
            continue
        
        # Mask all tokens up to and including the prompt
        mask_end = match_idx + len(prompt_ids)
        new_labels[i, :mask_end] = -100
    
    batch['labels'] = new_labels
    return batch


def evaluate_model_output(y_preds: List[str], y_trues: List[str]) -> List[int]:
    """
    Evaluate the model output by comparing the predicted and true labels. The evaluation 
    is performed by turning the model predictions and true predictions into a set of
    formulas and comparing these two sets.

    Args:
        y_preds: The model predictions.
        y_trues: The true labels.
    
    Returns:
        A list of 1s and 0s, where 1 indicates that the model prediction is correct and 0 otherwise.
    """
    return [1 if set(y_true.lower().split(", ")) == set(y_pred.lower().split(", ")) else 0 for y_true, y_pred in zip(y_trues, y_preds)]

