"""Shared helpers: model name mapping and tokenizer/pad handling."""

MODEL_MAP = {
    "Llama2-7B": "meta-llama/Llama-2-7b-hf",
    "Llama2-13B": "meta-llama/Llama-2-13b-hf",
    "Llama3-8B": "meta-llama/Meta-Llama-3-8B",
    "Llama3-3B": "meta-llama/Llama-3.2-3B",
    "Llama3.2-3B": "meta-llama/Llama-3.2-3B",
    "Mistral-7B": "mistralai/Mistral-7B-v0.1",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
}

MODULE_MAP = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "u": "up_proj",
              "d": "down_proj", "o": "o_proj", "g": "gate_proj"}


def resolve_model(name: str) -> str:
    """Map a short name to an HF id; pass raw HF ids through unchanged."""
    return MODEL_MAP.get(name, name)


def model_tag(model_name: str) -> str:
    """Model name as used in dataset folder names: short names unchanged, HF ids / paths -> last component."""
    import os
    return os.path.basename(model_name.rstrip("/\\"))


def train_data_paths(data_dir: str, task: str, model_name: str, max_length: int):
    """(train_path, validation_path_or_None) for a training task; shared by train.py and the bench probe."""
    import os
    model_name = model_tag(model_name)
    if task == "GSM8k":
        return os.path.join(data_dir, "GSM8k_processed/train"), None
    if task == "GSM8k_lori":
        return os.path.join(data_dir, f"gsm8k_lori_shuffle_{model_name}_special_token"), None
    if task == "metamath":
        return os.path.join(data_dir, f"metamath/train_all_{max_length}_OnlyOutput"), None
    if task == "commonsense":
        return (os.path.join(data_dir, f"commonsense_170k_new/train_all_{max_length}_OnlyOutput_{model_name}"),
                os.path.join(data_dir, "dataset_commonsense/combined_val_0.1_512"))
    if task == "commonsense_new":
        return os.path.join(data_dir, f"commonsense_170k_new/train_val120_{max_length}_OnlyOutput_{model_name}"), None
    raise ValueError(f"Unknown task: {task}")


def parse_target_modules(spec: str) -> list:
    return [MODULE_MAP[c] for c in spec if c in MODULE_MAP]


def setup_pad_token(tokenizer, model=None):
    """Give the tokenizer a pad token without changing the vocabulary.

    Upstream added a new '<PAD>' token and resized the embeddings. That shrinks Qwen's padded
    embedding matrix, and on Llama it adds an lm_head row that recent transformers initialise
    *randomly* (mean_resizing), so train and eval processes see a different extra logit.
    Padding is always masked (attention_mask 0, labels -100), so any existing token works:
    keep the model's pad token, else reuse <unk> (Llama-2, id 0 as in LLM-Adapters), else EOS (Llama-3).
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token is not None else tokenizer.eos_token
    if model is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
    return tokenizer
