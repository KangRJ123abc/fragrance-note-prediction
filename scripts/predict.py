"""Generate a first-pass fragrance-note pyramid for one review.

This release helper reuses the frozen dataset prompt and vLLM decoding settings.
It optionally downloads a pinned model; local directories work before publication.
Dependencies are vLLM, Transformers, and the adjacent model download helpers.
The historical 24-case length-fallback pipeline is not implemented by this helper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from download_models import obtain_model
from model_files import REPOSITORY_ROOT, verify_files


# ========== Prompt and generation ==========

def build_messages(review: str) -> list[dict[str, str]]:
    """Build the original system and user messages without reference answers.

    Args:
        review (str): Nonempty anonymized review supplied directly by the caller.
    Returns:
        list[dict[str, str]]: Two messages in system then user order.
    Raises:
        ValueError: Review contains no non-whitespace text.
        ImportError: The local dataset prompt module is missing.
    """
    # 步骤1: Validate external text before allocating GPU memory.
    if not review.strip():
        raise ValueError("构建提示失败: 评论为空, 上下文: review length=0")
    # 步骤2: Reuse exact frozen templates rather than creating a new task prompt.
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    from build_review_note_dataset_v2 import SYSTEM_INSTRUCTION, USER_TASK_TEMPLATE
    return [{"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": USER_TASK_TEMPLATE.format(review=review).strip()}]


def generate_prediction(model_directory: Path, messages: list[dict[str, str]]) -> str:
    """Load one local model and generate its unmodified first-pass response.

    Args:
        model_directory (Path): Verified artifact directory from CLI or downloader.
        messages (list[dict[str, str]]): System/user prompt created by build_messages.
    Returns:
        str: Raw generated text; valid JSON is not guaranteed.
    Raises:
        RuntimeError: Inference dependencies, tokenizer, model loading or generation fail.
    """
    # 步骤1: Delay optional GPU imports so --help remains lightweight.
    try:
        from vllm import LLM, SamplingParams
        engine = LLM(model=str(model_directory), dtype="bfloat16", max_model_len=4096,
                     gpu_memory_utilization=0.82, seed=3407, trust_remote_code=False)
        tokenizer = engine.get_tokenizer()
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        parameters = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=256,
                                    seed=3407, stop=["<|endoftext|>", "<|im_end|>"])
        response = engine.generate([prompt], parameters)[0]
    except Exception as error:
        raise RuntimeError(f"生成香调失败: 请检查兼容推理环境与模型, 上下文: model={model_directory.name}") from error
    return response.outputs[0].text


def main() -> None:
    """Resolve a model and explicitly run inference for one CLI review.

    Args:
        None: Review, model role and optional local path are supplied by CLI.
    Returns:
        None: Prints the model's raw response without altering it.
    Raises:
        ValueError: Review, model provenance, or download settings are invalid.
        RuntimeError: Artifact access, download, or inference fails.
    """
    # 步骤1: Require user text; an empty invocation never downloads or infers.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True)
    parser.add_argument("--role", choices=("base", "finetuned"), default="finetuned")
    parser.add_argument("--model-dir", type=Path)
    arguments = parser.parse_args()
    messages = build_messages(arguments.review)
    # 步骤2: Explicit local files and downloaded files use the same frozen manifest.
    directory = arguments.model_dir.resolve() if arguments.model_dir else obtain_model(arguments.role)
    if arguments.model_dir:
        manifest = REPOSITORY_ROOT / f"configs/{'base' if arguments.role == 'base' else 'finetuned'}_model_files.json"
        verify_files(directory, manifest)
    print(generate_prediction(directory, messages))


if __name__ == "__main__":
    # 步骤1: Example: python scripts/predict.py --model-dir ../modelscope --review "Citrus and woods."
    main()
