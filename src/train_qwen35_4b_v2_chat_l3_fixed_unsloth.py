"""
Train Qwen3.5-4B with Unsloth on Level-3 perfume-review chat data.

The module uses Unsloth's full Qwen3.5 vision-model loader and gradient
checkpointing while fine-tuning language layers only. It retains the verified
assistant-only labels, padding-safe evaluation, early stopping, and full-model
merge checks from the Transformers reference implementation. It depends on
Unsloth, PyTorch, Transformers, Datasets, and PEFT.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Unsloth must patch Transformers before model classes are imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault(
    "UNSLOTH_COMPILE_LOCATION",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "unsloth_compiled_cache_qwen35"),
)

import torch
from unsloth import FastVisionModel
from datasets import Dataset, load_dataset
from transformers import (
    AutoConfig,
    EarlyStoppingCallback,
    GenerationConfig,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ========== Configuration ==========

# ------ Reproducible training defaults ------
RANDOM_SEED = 3407
DEFAULT_MAX_SEQUENCE_LENGTH = 2048
DEFAULT_MAX_STEPS = 2000
DEFAULT_EVALUATION_STEPS = 100
DEFAULT_LEARNING_RATE = 1.5e-5
DEFAULT_WARMUP_STEPS = 50
DEFAULT_TRAIN_BATCH_SIZE = 1
DEFAULT_EVALUATION_BATCH_SIZE = 4
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_EARLY_STOPPING_PATIENCE = 4

# ------ LoRA coverage for Qwen3.5 hybrid text layers ------
LORA_TARGET_MODULES = [
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ========== Command-Line Arguments ==========


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for full training and smoke testing.

    The parser exposes paths and high-impact optimization settings while keeping
    architecture-sensitive choices inside this module.

    Returns:
        argparse.ArgumentParser: Configured parser used by ``main``.

    Raises:
        RuntimeError: This function does not normally raise an exception.
    """

    # 步骤1: Resolve defaults relative to the script so the server copy is portable.
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=project_root / "model/Qwen3.5-4b/unsloth/Qwen3.5-4B")
    parser.add_argument("--train-file", type=Path, default=project_root / "data/qwen_finetune_notes_review_v2_strict_chat_train.jsonl")
    parser.add_argument("--eval-file", type=Path, default=project_root / "data/qwen_finetune_notes_review_v2_strict_chat_val.jsonl")
    parser.add_argument("--output-dir", type=Path, default=project_root / "output/perfume_qwen35_4b_v2_chat_l3_fixed_unsloth")
    parser.add_argument("--final-dir", type=Path, default=project_root / "output/finetuned_qwen35_4b_v2_chat_l3_fixed_unsloth_merged")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--eval-steps", type=int, default=DEFAULT_EVALUATION_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=1)
    return parser


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Validate paths and numeric settings before allocating GPU memory.

    Args:
        arguments (argparse.Namespace): Values produced by ``build_argument_parser``
            and supplied by the command-line caller.

    Returns:
        None: Validation succeeds without changing the arguments.

    Raises:
        ValueError: If a numeric training setting is non-positive.
        FileNotFoundError: If the model directory or a dataset file is missing.
    """

    # 步骤1: Reject invalid optimization settings before model loading.
    if arguments.max_steps <= 0 or arguments.eval_steps <= 0 or arguments.learning_rate <= 0 or arguments.smoke_steps <= 0:
        raise ValueError(f"Argument validation failed: positive values required, context: {vars(arguments)}")

    # 步骤2: Fail with explicit context when required external inputs are absent.
    for input_path in (arguments.model_dir, arguments.train_file, arguments.eval_file):
        if not input_path.exists():
            raise FileNotFoundError(f"Input validation failed: path does not exist, context: path={input_path}")


# ========== Dataset Preparation ==========


def is_level_three_example(example: Dict[str, Any]) -> bool:
    """Return whether one dataset row is labeled as scent-information Level 3.

    Args:
        example (Dict[str, Any]): Raw JSONL row supplied by Hugging Face Datasets;
            metadata may be a dictionary or a serialized JSON object.

    Returns:
        bool: ``True`` only when ``scent_information_level`` equals integer 3.

    Raises:
        ValueError: If serialized metadata cannot be decoded as JSON.
    """

    # 步骤1: Normalize metadata because Datasets may preserve it as serialized text.
    metadata = example.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError as exception:
            raise ValueError(f"Metadata decoding failed: {exception}, context: metadata={metadata[:200]}") from exception

    # 步骤2: Keep only the evidence-rich subset requested by the experiment.
    return metadata.get("scent_information_level") == 3


def validate_messages(messages: Any) -> List[Dict[str, str]]:
    """Validate and normalize one system-user-assistant message sequence.

    Args:
        messages (Any): ``messages`` field read from one chat-format dataset row.

    Returns:
        List[Dict[str, str]]: Validated messages ending in exactly one assistant
        answer; typical roles are system, user, and assistant.

    Raises:
        ValueError: If messages are missing, malformed, empty, or do not end with
        a non-empty assistant response.
    """

    # 步骤1: Check the outer sequence before indexing its final message.
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"Message validation failed: expected a message list, context: messages={messages}")

    # 步骤2: Validate every role/content pair to prevent silent template corruption.
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
            raise ValueError(f"Message validation failed: invalid role/content, context: index={message_index}, message={message}")

    # 步骤3: Assistant-only masking requires the reference answer to be last.
    if messages[-1]["role"] != "assistant" or not messages[-1]["content"].strip():
        raise ValueError(f"Message validation failed: final assistant answer required, context: final_message={messages[-1]}")
    return messages


def render_chat_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Dict[str, str]],
    is_full_conversation: bool,
) -> List[int]:
    """Render a Qwen3.5 chat sequence directly into token IDs.

    Args:
        tokenizer (PreTrainedTokenizerBase): Nested tokenizer returned by
            ``load_unsloth_assets``.
        messages (Sequence[Dict[str, str]]): Validated chat messages from one row.
        is_full_conversation (bool): ``True`` includes the assistant reference;
            ``False`` renders the generation prompt preceding that reference.

    Returns:
        List[int]: Chat-template token IDs without padding, usually 200-1200 IDs.

    Raises:
        RuntimeError: If the tokenizer does not return a list of integer token IDs.
    """

    # 步骤1: The direct keyword is required; nesting it in chat_template_kwargs is ignored.
    rendered_token_ids = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=not is_full_conversation,
        enable_thinking=False,
    )

    # 步骤2: Transformers 5.x may wrap a single sequence in BatchEncoding.
    if hasattr(rendered_token_ids, "input_ids"):
        rendered_token_ids = rendered_token_ids.input_ids
    elif isinstance(rendered_token_ids, dict) and "input_ids" in rendered_token_ids:
        rendered_token_ids = rendered_token_ids["input_ids"]
    if isinstance(rendered_token_ids, list) and len(rendered_token_ids) == 1 and isinstance(rendered_token_ids[0], list):
        rendered_token_ids = rendered_token_ids[0]

    # 步骤3: Validate the normalized tokenizer output before constructing labels.
    if not isinstance(rendered_token_ids, list) or not all(isinstance(token_id, int) for token_id in rendered_token_ids):
        raise RuntimeError(f"Chat rendering failed: token ID list required, context: output_type={type(rendered_token_ids)}")
    return rendered_token_ids


def tokenize_chat_example(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    maximum_sequence_length: int,
) -> Dict[str, List[int]]:
    """Create input IDs and assistant-only labels for one chat row.

    Args:
        example (Dict[str, Any]): Level-3 row supplied by Hugging Face Datasets.
        tokenizer (PreTrainedTokenizerBase): Qwen3.5 tokenizer loaded in ``main``.
        maximum_sequence_length (int): Hard safety limit selected by the training
            configuration; current data was audited below 1200 tokens.

    Returns:
        Dict[str, List[int]]: ``input_ids``, ``attention_mask``, and ``labels``;
        prompt labels are -100 and assistant labels equal their input token IDs.

    Raises:
        ValueError: If messages are malformed, exceed the configured limit, do
        not share an exact prompt prefix, or contain no supervised token.
    """

    # 步骤1: Render full and prompt sequences from the same validated messages.
    messages = validate_messages(example.get("messages"))
    full_token_ids = render_chat_token_ids(tokenizer, messages, is_full_conversation=True)
    prompt_token_ids = render_chat_token_ids(tokenizer, messages[:-1], is_full_conversation=False)

    # 步骤2: Reject truncation because losing assistant tokens silently corrupts loss.
    if len(full_token_ids) > maximum_sequence_length:
        raise ValueError(f"Sequence validation failed: sequence exceeds limit, context: length={len(full_token_ids)}, limit={maximum_sequence_length}")
    if full_token_ids[: len(prompt_token_ids)] != prompt_token_ids:
        raise ValueError(f"Sequence validation failed: prompt is not an exact prefix, context: full_length={len(full_token_ids)}, prompt_length={len(prompt_token_ids)}")

    # 步骤3: Mask the prompt while retaining every assistant and end token.
    labels = [-100] * len(prompt_token_ids) + full_token_ids[len(prompt_token_ids) :]
    if not any(label != -100 for label in labels):
        raise ValueError(f"Label validation failed: no supervised token, context: sequence_length={len(full_token_ids)}")
    return {"input_ids": full_token_ids, "attention_mask": [1] * len(full_token_ids), "labels": labels}


def load_level_three_dataset(dataset_path: Path, split_name: str) -> Dataset:
    """Load and filter one JSONL split to Level-3 examples.

    Args:
        dataset_path (Path): Train or validation JSONL path supplied by CLI.
        split_name (str): Human-readable split label supplied by ``prepare_datasets``.

    Returns:
        Dataset: Filtered Hugging Face dataset retaining original columns.

    Raises:
        RuntimeError: If JSONL loading or filtering fails.
    """

    # 步骤1: Wrap external file I/O so the failing split and path are visible.
    try:
        loaded_dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        filtered_dataset = loaded_dataset.filter(is_level_three_example, desc=f"Filtering {split_name} Level-3 rows")
    except Exception as exception:
        raise RuntimeError(f"Dataset loading failed: {exception}, context: split={split_name}, path={dataset_path}") from exception

    # 步骤2: Empty splits should fail before model training begins.
    if len(filtered_dataset) == 0:
        raise ValueError(f"Dataset validation failed: no Level-3 rows, context: split={split_name}, path={dataset_path}")
    return filtered_dataset


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    maximum_sequence_length: int,
    split_name: str,
) -> Dataset:
    """Tokenize a filtered split and remove raw chat columns.

    Args:
        dataset (Dataset): Level-3 split produced by ``load_level_three_dataset``.
        tokenizer (PreTrainedTokenizerBase): Base tokenizer loaded in ``main``.
        maximum_sequence_length (int): Maximum accepted full-chat token length.
        split_name (str): Label used in progress output and exception context.

    Returns:
        Dataset: Tokenized split containing only model input fields.

    Raises:
        RuntimeError: If chat rendering or dataset mapping fails.
    """

    # 步骤1: Map serially so malformed row exceptions retain deterministic context.
    try:
        return dataset.map(
            lambda example: tokenize_chat_example(example, tokenizer, maximum_sequence_length),
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {split_name} chat rows",
        )
    except Exception as exception:
        raise RuntimeError(f"Dataset tokenization failed: {exception}, context: split={split_name}, rows={len(dataset)}") from exception


def prepare_datasets(
    train_path: Path,
    evaluation_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    is_smoke_test: bool,
) -> tuple[Dataset, Dataset]:
    """Load, optionally subset, and tokenize training and validation data.

    Args:
        train_path (Path): Level-mixed training JSONL path supplied by CLI.
        evaluation_path (Path): Level-mixed validation JSONL path supplied by CLI.
        tokenizer (PreTrainedTokenizerBase): Configured Qwen3.5 tokenizer.
        is_smoke_test (bool): CLI flag selecting 8 train and 4 validation rows.

    Returns:
        tuple[Dataset, Dataset]: Tokenized train and validation datasets.

    Raises:
        RuntimeError: If loading, filtering, or tokenization fails.
    """

    # 步骤1: Load complete Level-3 splits before optional smoke-test subsetting.
    training_dataset = load_level_three_dataset(train_path, "train")
    evaluation_dataset = load_level_three_dataset(evaluation_path, "validation")
    if is_smoke_test:
        training_dataset = training_dataset.select(range(min(8, len(training_dataset))))
        evaluation_dataset = evaluation_dataset.select(range(min(4, len(evaluation_dataset))))

    # 步骤2: Render exact chat tokens and assistant-only labels.
    tokenized_training_dataset = tokenize_dataset(training_dataset, tokenizer, DEFAULT_MAX_SEQUENCE_LENGTH, "train")
    tokenized_evaluation_dataset = tokenize_dataset(evaluation_dataset, tokenizer, DEFAULT_MAX_SEQUENCE_LENGTH, "validation")
    return tokenized_training_dataset, tokenized_evaluation_dataset


# ========== Padding-Safe Data Collator ==========


@dataclass
class AssistantOnlyDataCollator:
    """Right-pad model inputs and keep assistant labels at identical positions.

    Attributes:
        tokenizer (PreTrainedTokenizerBase): Tokenizer configured with right-side
            padding by ``load_unsloth_assets`` and passed from ``main``.
        pad_to_multiple_of (int): Tensor-width alignment used for GPU efficiency.
    """

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate tokenized examples into a verified right-padded batch.

        Args:
            features (List[Dict[str, Any]]): Variable-length rows supplied by the
                Trainer data loader after dataset tokenization.

        Returns:
            Dict[str, torch.Tensor]: Padded input IDs, attention masks, and labels.

        Raises:
            ValueError: If the batch is empty or tokenizer padding is not right-sided.
            RuntimeError: If supervised labels do not align with input token IDs.
        """

        # 步骤1: Refuse the left-padding state that invalidated the previous eval loss.
        if not features or self.tokenizer.padding_side != "right":
            raise ValueError(f"Batch collation failed: non-empty right-padded batch required, context: features={len(features)}, padding_side={self.tokenizer.padding_side}")

        # 步骤2: Let the tokenizer pad model inputs and attention masks on the right.
        model_features = [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in features]
        batch = self.tokenizer.pad(model_features, padding=True, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt")

        # 步骤3: Pad labels on the same side and verify every supervised position.
        batch_width = int(batch["input_ids"].shape[1])
        padded_labels = [item["labels"] + [-100] * (batch_width - len(item["labels"])) for item in features]
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        supervised_mask = batch["labels"] != -100
        if not torch.equal(batch["labels"][supervised_mask], batch["input_ids"][supervised_mask]):
            raise RuntimeError(f"Batch alignment failed: labels differ from input IDs, context: batch_size={len(features)}, width={batch_width}")
        return batch


def verify_collator(
    data_collator: AssistantOnlyDataCollator,
    evaluation_dataset: Dataset,
) -> None:
    """Run a pre-training batch alignment check on variable-length examples.

    Args:
        data_collator (AssistantOnlyDataCollator): Collator configured in ``main``.
        evaluation_dataset (Dataset): Tokenized validation split from which four
            variable-length rows are selected.

    Returns:
        None: Successful collation and alignment complete without side effects.

    Raises:
        RuntimeError: If no supervised tokens survive collation.
        ValueError: Propagated when collator preconditions fail.
    """

    # 步骤1: Select rows with different lengths to force actual padding.
    sorted_indices = sorted(range(len(evaluation_dataset)), key=lambda index: len(evaluation_dataset[index]["input_ids"]))
    candidate_indices = sorted(set([sorted_indices[0], sorted_indices[-1], *sorted_indices[1:3]]))
    collated_batch = data_collator([evaluation_dataset[index] for index in candidate_indices])

    # 步骤2: Confirm the diagnostic batch contains supervised assistant tokens.
    supervised_token_count = int((collated_batch["labels"] != -100).sum().item())
    if supervised_token_count <= 0:
        raise RuntimeError(f"Collator verification failed: no supervised tokens, context: indices={candidate_indices}")
    print(f"Collator verification passed: batch={len(candidate_indices)}, supervised_tokens={supervised_token_count}")


# ========== Model and LoRA Setup ==========


def load_unsloth_assets(
    model_directory: Path,
) -> tuple[torch.nn.Module, Any, PreTrainedTokenizerBase]:
    """Load the full Qwen3.5 model, processor, and nested tokenizer via Unsloth.

    Args:
        model_directory (Path): Local full Qwen3.5 checkpoint supplied by CLI.

    Returns:
        tuple[torch.nn.Module, Any, PreTrainedTokenizerBase]: Unsloth-patched full
        model, Qwen processor used for merged saving, and its nested text tokenizer.

    Raises:
        RuntimeError: If local Unsloth model loading fails.
        ValueError: If the full architecture, nested tokenizer, or pad token is absent.
    """

    # 步骤1: Load BF16 LoRA weights locally; official guidance discourages QLoRA.
    try:
        model, processor = FastVisionModel.from_pretrained(
            model_name=str(model_directory),
            max_seq_length=DEFAULT_MAX_SEQUENCE_LENGTH,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
            use_gradient_checkpointing="unsloth",
            local_files_only=True,
        )
    except Exception as exception:
        raise RuntimeError(f"Unsloth model loading failed: {exception}, context: model_directory={model_directory}") from exception

    # 步骤2: Preserve the full VLM while using its nested tokenizer for text batches.
    if model.config.model_type != "qwen3_5" or not hasattr(processor, "tokenizer"):
        raise ValueError(f"Unsloth asset validation failed: full model and nested tokenizer required, context: model_type={model.config.model_type}, processor={type(processor)}")
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError(f"Tokenizer validation failed: pad token unavailable, context: model_directory={model_directory}")

    # 步骤3: Disable both cache layers before gradient-checkpointed training.
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    return model, processor, tokenizer


def count_target_modules(model: torch.nn.Module) -> Dict[str, int]:
    """Count model modules matched by each configured LoRA suffix.

    Args:
        model (torch.nn.Module): Full base model returned by ``load_unsloth_assets``.

    Returns:
        Dict[str, int]: Mapping from target suffix to positive match count.

    Raises:
        ValueError: If any configured target suffix matches no module.
    """

    # 步骤1: Count exact final path components to avoid ambiguous substring matches.
    module_names = [module_name for module_name, _ in model.named_modules()]
    target_counts = {target: sum(module_name.split(".")[-1] == target for module_name in module_names) for target in LORA_TARGET_MODULES}

    # 步骤2: Every requested attention and MLP family must exist before PEFT wrapping.
    missing_targets = [target for target, count in target_counts.items() if count == 0]
    if missing_targets:
        raise ValueError(f"LoRA target validation failed: modules not found, context: missing_targets={missing_targets}")
    return target_counts


def attach_lora(model: torch.nn.Module) -> torch.nn.Module:
    """Attach Unsloth LoRA adapters to all Qwen3.5 text layer types.

    Args:
        model (torch.nn.Module): Full Unsloth-patched model from ``load_unsloth_assets``.

    Returns:
        torch.nn.Module: PEFT-wrapped causal language model ready for Trainer.

    Raises:
        RuntimeError: If PEFT adapter creation fails.
    """

    # 步骤1: Report module coverage before mutating the model with PEFT wrappers.
    target_counts = count_target_modules(model)
    print(f"LoRA target coverage: {target_counts}")
    # 步骤2: Freeze vision and adapt both Qwen3.5 attention families plus text MLPs.
    try:
        adapted_model = FastVisionModel.get_peft_model(
            model,
            r=16,
            target_modules=LORA_TARGET_MODULES,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            use_gradient_checkpointing="unsloth",
            random_state=RANDOM_SEED,
            max_seq_length=DEFAULT_MAX_SEQUENCE_LENGTH,
        )
    except Exception as exception:
        raise RuntimeError(f"LoRA attachment failed: {exception}, context: targets={LORA_TARGET_MODULES}") from exception
    FastVisionModel.for_training(adapted_model, use_gradient_checkpointing=True)
    adapted_model.print_trainable_parameters()
    return adapted_model


# ========== Trainer Configuration ==========


def build_training_arguments(arguments: argparse.Namespace) -> TrainingArguments:
    """Create corrected Trainer settings for full training or a bounded smoke test.

    Args:
        arguments (argparse.Namespace): Validated command-line values from ``main``.

    Returns:
        TrainingArguments: Hugging Face Trainer configuration.

    Raises:
        ValueError: Propagated by Transformers when argument combinations conflict.
    """

    # 步骤1: Smoke mode evaluates every update but writes no checkpoint.
    maximum_steps = arguments.smoke_steps if arguments.smoke_test else arguments.max_steps
    evaluation_steps = 1 if arguments.smoke_test else arguments.eval_steps
    output_directory = Path("/tmp/qwen35_4b_chat_l3_fixed_smoke") if arguments.smoke_test else arguments.output_dir

    # 步骤2: Full mode saves every evaluation and restores the lowest corrected loss.
    return TrainingArguments(
        output_dir=str(output_directory),
        per_device_train_batch_size=DEFAULT_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=DEFAULT_EVALUATION_BATCH_SIZE,
        gradient_accumulation_steps=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=maximum_steps,
        learning_rate=arguments.learning_rate,
        warmup_steps=0 if arguments.smoke_test else DEFAULT_WARMUP_STEPS,
        weight_decay=0.01,
        optim="adamw_torch_fused",
        bf16=True,
        logging_steps=1 if arguments.smoke_test else 10,
        eval_strategy="steps",
        eval_steps=evaluation_steps,
        save_strategy="no" if arguments.smoke_test else "steps",
        save_steps=evaluation_steps,
        save_total_limit=5,
        load_best_model_at_end=not arguments.smoke_test,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0 if arguments.smoke_test else 2,
        remove_unused_columns=False,
        report_to="none",
        seed=RANDOM_SEED,
    )


def build_trainer(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    training_dataset: Dataset,
    evaluation_dataset: Dataset,
    training_arguments: TrainingArguments,
    is_smoke_test: bool,
) -> Trainer:
    """Construct the Trainer with corrected collation and optional early stopping.

    Args:
        model (torch.nn.Module): PEFT-wrapped full Qwen3.5 model.
        tokenizer (PreTrainedTokenizerBase): Right-padding tokenizer from ``main``.
        training_dataset (Dataset): Tokenized Level-3 training rows.
        evaluation_dataset (Dataset): Tokenized Level-3 validation rows.
        training_arguments (TrainingArguments): Settings from ``build_training_arguments``.
        is_smoke_test (bool): Whether to omit early stopping during the bounded test.

    Returns:
        Trainer: Ready-to-run Hugging Face Trainer instance.

    Raises:
        ValueError: If Trainer rejects a supplied dataset or argument combination.
    """

    # 步骤1: Use patience across four evaluations to stop sustained overfitting.
    callbacks = [] if is_smoke_test else [EarlyStoppingCallback(early_stopping_patience=DEFAULT_EARLY_STOPPING_PATIENCE, early_stopping_threshold=0.002)]

    # 步骤2: Inject the verified assistant-only collator into both train and eval.
    return Trainer(
        model=model,
        args=training_arguments,
        train_dataset=training_dataset,
        eval_dataset=evaluation_dataset,
        data_collator=AssistantOnlyDataCollator(tokenizer),
        callbacks=callbacks,
    )


# ========== Merge and Save ==========


def build_generation_config(
    merged_model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
) -> GenerationConfig:
    """Build generation settings that stop on both Qwen end-token variants.

    Args:
        merged_model (torch.nn.Module): LoRA-merged full Qwen3.5 model.
        tokenizer (PreTrainedTokenizerBase): Tokenizer whose EOS is ``<|im_end|>``.

    Returns:
        GenerationConfig: Saveable configuration with both EOS IDs and pad ID.

    Raises:
        ValueError: If no EOS token ID can be resolved.
    """

    # 步骤1: Preserve the base end-of-text ID and add the chat-template end ID.
    generation_configuration = GenerationConfig.from_model_config(merged_model.config)
    configured_eos = generation_configuration.eos_token_id
    eos_token_ids = list(configured_eos) if isinstance(configured_eos, list) else [configured_eos]
    text_config_eos = getattr(getattr(merged_model.config, "text_config", None), "eos_token_id", None)
    eos_token_ids = sorted({token_id for token_id in [*eos_token_ids, text_config_eos, tokenizer.eos_token_id] if token_id is not None})
    if not eos_token_ids:
        raise ValueError("Generation config failed: no EOS token IDs, context: model and tokenizer EOS are None")

    # 步骤2: Explicit IDs prevent valid JSON from continuing after <|im_end|>.
    generation_configuration.eos_token_id = eos_token_ids
    generation_configuration.pad_token_id = tokenizer.pad_token_id
    return generation_configuration


def verify_full_model_config(model: torch.nn.Module) -> None:
    """Verify that LoRA merging retained the full Qwen3.5 outer architecture.

    Args:
        model (torch.nn.Module): Merged model returned by PEFT ``merge_and_unload``.

    Returns:
        None: Validation succeeds without modifying the model.

    Raises:
        RuntimeError: If the model was reduced to ``qwen3_5_text`` or another type.
    """

    # 步骤1: The previous AutoModelForCausalLM merge failed this exact invariant.
    model_type = getattr(model.config, "model_type", None)
    architectures = getattr(model.config, "architectures", None)
    if model_type != "qwen3_5" or "Qwen3_5ForConditionalGeneration" not in (architectures or []):
        raise RuntimeError(f"Merged config validation failed: full architecture required, context: model_type={model_type}, architectures={architectures}")


def merge_and_save_model(
    trainer: Trainer,
    processor: Any,
    tokenizer: PreTrainedTokenizerBase,
    final_directory: Path,
    is_smoke_test: bool,
) -> None:
    """Merge the active best adapter, verify config, and optionally save artifacts.

    Args:
        trainer (Trainer): Completed Trainer whose model contains the active adapter.
        processor (Any): Qwen3VLProcessor returned by Unsloth and used for export.
        tokenizer (PreTrainedTokenizerBase): Nested text tokenizer used for EOS IDs.
        final_directory (Path): CLI-selected destination for normal training output.
        is_smoke_test (bool): When true, verify merge in memory and write nothing.

    Returns:
        None: Merge and optional save complete successfully.

    Raises:
        RuntimeError: If merging, saving, or post-save config validation fails.
    """

    # 步骤1: Smoke mode verifies an in-memory PEFT merge without writing 9 GB.
    if is_smoke_test:
        try:
            merged_model = trainer.model.merge_and_unload()
            verify_full_model_config(merged_model)
        except Exception as exception:
            raise RuntimeError(f"Model merge failed: {exception}, context: final_directory={final_directory}") from exception
        print("Smoke-test merge verification passed; no model files were saved.")
        return

    # 步骤2: Use Unsloth's Qwen3.5 VLM-aware merged-16bit export path.
    try:
        final_directory.mkdir(parents=True, exist_ok=True)
        verify_full_model_config(trainer.model)
        trainer.model.config.use_cache = True
        trainer.model.config.text_config.use_cache = True
        trainer.model.save_pretrained_merged(
            str(final_directory),
            processor,
            save_method="merged_16bit",
        )
        build_generation_config(trainer.model, tokenizer).save_pretrained(final_directory)
        saved_config = AutoConfig.from_pretrained(final_directory, local_files_only=True)
    except Exception as exception:
        raise RuntimeError(f"Model saving failed: {exception}, context: final_directory={final_directory}") from exception

    # 步骤3: Re-read disk config so deployment cannot inherit a text-only artifact.
    if saved_config.model_type != "qwen3_5":
        raise RuntimeError(f"Saved config validation failed: expected qwen3_5, context: model_type={saved_config.model_type}, directory={final_directory}")
    print(f"Merged full model saved to {final_directory}")


# ========== Runtime Cleanup ==========


def release_gpu_memory() -> None:
    """Release Python and CUDA caches after training or smoke testing.

    Returns:
        None: Garbage collection and CUDA cache cleanup are completed.

    Raises:
        RuntimeError: This function does not normally raise an exception.
    """

    # 步骤1: Collect unreachable tensors before asking CUDA to release cached blocks.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ========== Main Workflow ==========


def main() -> None:
    """Run corrected Unsloth Qwen3.5 training or a bounded smoke test.

    Command-line arguments are parsed from the current process. The function
    loads external model/data files, trains through Hugging Face Trainer, and in
    normal mode saves a merged full-architecture model.

    Returns:
        None: The requested workflow finishes and all GPU caches are released.

    Raises:
        ValueError: If arguments, data, labels, padding, or model type are invalid.
        RuntimeError: If external loading, training, merging, or saving fails.
    """

    # 步骤1: Validate paths, then load Unsloth's full model and nested tokenizer.
    arguments = build_argument_parser().parse_args()
    validate_arguments(arguments)
    set_seed(RANDOM_SEED)
    base_model, processor, tokenizer = load_unsloth_assets(arguments.model_dir)

    # 步骤2: Prepare exact chat labels and verify padding before attaching LoRA.
    training_dataset, evaluation_dataset = prepare_datasets(arguments.train_file, arguments.eval_file, tokenizer, arguments.smoke_test)
    data_collator = AssistantOnlyDataCollator(tokenizer)
    verify_collator(data_collator, evaluation_dataset)
    print(f"Prepared datasets: train={len(training_dataset)}, validation={len(evaluation_dataset)}")

    # 步骤3: Attach complete text-only LoRA coverage and run the Trainer.
    adapted_model = attach_lora(base_model)
    tokenizer.padding_side = "right"
    verify_collator(data_collator, evaluation_dataset)
    training_arguments = build_training_arguments(arguments)
    trainer = build_trainer(adapted_model, tokenizer, training_dataset, evaluation_dataset, training_arguments, arguments.smoke_test)
    try:
        trainer.train(resume_from_checkpoint=arguments.resume_from_checkpoint)
        merge_and_save_model(trainer, processor, tokenizer, arguments.final_dir, arguments.smoke_test)
    except Exception as exception:
        raise RuntimeError(f"Training workflow failed: {exception}, context: smoke_test={arguments.smoke_test}, output_dir={training_arguments.output_dir}") from exception
    finally:
        release_gpu_memory()

    # 步骤4: Emit an unambiguous completion marker for shell automation.
    print("Smoke test completed successfully." if arguments.smoke_test else "Training and merge completed successfully.")


if __name__ == "__main__":
    main()
