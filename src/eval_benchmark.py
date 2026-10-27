"""
统一评估 Qwen3.5-4B 基座模型与当前 700-step 微调模型的通用能力。

本模块使用独立的 EvalScope 1.4.2 客户端环境，通过 OpenAI 兼容接口调用
vLLM。脚本按模型顺序启动和关闭服务，确保单张 24 GB GPU 同时只驻留一个
模型；两种模型共享完全相同的数据集、few-shot、解码与答案抽取协议。
外部依赖包括 EvalScope 1.4.2、可运行 Qwen3.5 的 vLLM 环境、模型目录和网络
可访问的 ModelScope 数据集仓库。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ========== 配置 ==========

# ------ 可复现评测默认值 ------
EVALSCOPE_REQUIRED_VERSION = "1.4.2"
RANDOM_SEED = 42
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
# MMLU 5-shot 的最长已审计输入为 3,200 token，需与 1,024 token 输出预算同时容纳。
DEFAULT_MAX_MODEL_LENGTH = 8192
DEFAULT_MAX_TOKENS = 1024
DEFAULT_EVAL_BATCH_SIZE = 16
DEFAULT_MAX_NUM_SEQUENCES = 32
DEFAULT_GPU_MEMORY_UTILIZATION = 0.82
DEFAULT_SERVER_START_TIMEOUT_SECONDS = 600
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class BenchmarkSpec:
    """描述一项通用基准的固定评测协议。

    Attributes:
        name (str): EvalScope 注册的数据集名称。
        full_limit (int): 正式评测中每个子集最多使用的样本数。
        few_shot_num (int): 提示中使用的固定示例数量。
        smoke_subset (str): 冒烟测试中唯一启用的代表性子集。
    """

    name: str
    full_limit: int
    few_shot_num: int
    smoke_subset: str


@dataclass(frozen=True)
class ModelTarget:
    """保存一个待评估模型的角色、服务名和本地路径。

    Attributes:
        role (str): 模型在配对比较中的角色，取 ``base`` 或 ``finetuned``。
        served_name (str): vLLM OpenAI 接口公开的模型标识。
        model_path (Path): 服务器上可直接加载的 Hugging Face 模型目录。
    """

    role: str
    served_name: str
    model_path: Path


# ------ 与论文旧实验一致的六项协议 ------
BENCHMARK_SPECS: dict[str, BenchmarkSpec] = {
    "ceval": BenchmarkSpec("ceval", 100, 0, "computer_network"),
    "cmmlu": BenchmarkSpec("cmmlu", 100, 0, "agronomy"),
    "mmlu": BenchmarkSpec("mmlu", 100, 5, "abstract_algebra"),
    "gsm8k": BenchmarkSpec("gsm8k", 300, 5, "main"),
    "hellaswag": BenchmarkSpec("hellaswag", 500, 0, "default"),
    "truthful_qa": BenchmarkSpec("truthful_qa", 200, 0, "multiple_choice"),
}


# ========== 命令行参数 ==========


def build_argument_parser() -> argparse.ArgumentParser:
    """构造通用基准评测的命令行解析器。

    解析器将模型路径、运行模式、数据集范围和 vLLM 资源参数全部暴露为显式
    参数，避免通过改源码切换模型或协议。调用方通常是服务器终端或自动化脚本。

    Returns:
        argparse.ArgumentParser: 已注册全部参数和默认值的解析器。

    Raises:
        RuntimeError: 正常情况下不会抛出；仅 argparse 内部异常时可能出现。
    """

    # 步骤1: 以脚本所在项目为默认根目录，保证从任意工作目录启动均一致。
    default_project_root = Path(__file__).resolve().parent.parent
    default_run_name = datetime.now().strftime("%Y%m%d_%H%M%S_qwen35_4b_general")

    # 步骤2: 注册会影响评测语义、输出位置或资源占用的参数。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--targets", nargs="+", choices=("base", "finetuned"), default=["base", "finetuned"])
    parser.add_argument("--datasets", nargs="+", choices=tuple(BENCHMARK_SPECS), default=list(BENCHMARK_SPECS))
    parser.add_argument("--project-root", type=Path, default=default_project_root)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--finetuned-model", type=Path)
    parser.add_argument("--run-dir", type=Path, default=default_project_root / "src/outputs/general_benchmarks" / default_run_name)
    parser.add_argument("--vllm-executable", type=Path, required=True, help="Explicit vLLM executable in the separate inference environment.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQUENCES)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LENGTH)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--server-start-timeout", type=int, default=DEFAULT_SERVER_START_TIMEOUT_SECONDS)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT_SECONDS)
    parser.add_argument("--force", action="store_true", help="重新执行已有报告的数据集。")
    return parser


def resolve_and_validate_arguments(arguments: argparse.Namespace) -> argparse.Namespace:
    """解析默认路径并验证评测参数的合法性。

    Args:
        arguments (argparse.Namespace): ``build_argument_parser`` 产生的原始参数，
            由命令行调用者传入。

    Returns:
        argparse.Namespace: 已补全绝对模型路径和运行目录的参数对象。

    Raises:
        ValueError: 端口、批大小、长度、超时或显存比例超出合法范围时抛出。
        FileNotFoundError: vLLM 可执行文件或本次选中的模型目录不存在时抛出。
    """

    # 步骤1: 根据项目根目录补全两个固定模型入口。
    project_root = arguments.project_root.resolve()
    arguments.project_root = project_root
    arguments.base_model = (arguments.base_model or project_root / "model/Qwen3.5-4b/unsloth/Qwen3.5-4B").resolve()
    arguments.finetuned_model = (
        arguments.finetuned_model
        or project_root / "output/finetuned_qwen35_4b_v2_chat_l3_fixed_unsloth_merged"
    ).resolve()
    arguments.run_dir = arguments.run_dir.resolve()
    arguments.vllm_executable = arguments.vllm_executable.resolve()

    # 步骤2: 检查会造成请求失败或资源配置无效的数值参数。
    positive_values = {
        "port": arguments.port,
        "eval_batch_size": arguments.eval_batch_size,
        "max_num_seqs": arguments.max_num_seqs,
        "max_model_len": arguments.max_model_len,
        "max_tokens": arguments.max_tokens,
        "server_start_timeout": arguments.server_start_timeout,
        "request_timeout": arguments.request_timeout,
    }
    invalid_values = {name: value for name, value in positive_values.items() if value <= 0}
    if invalid_values or arguments.port > 65535:
        raise ValueError(f"参数合法性检查失败: 数值参数无效, 上下文: invalid_values={invalid_values}, port={arguments.port}")
    if not 0.1 <= arguments.gpu_memory_utilization <= 0.95:
        raise ValueError(
            f"参数合法性检查失败: gpu_memory_utilization 超出范围, "
            f"上下文: value={arguments.gpu_memory_utilization}"
        )

    # 步骤3: 只验证本次确实会加载的模型，便于单模型诊断。
    required_paths = [arguments.vllm_executable]
    if "base" in arguments.targets:
        required_paths.append(arguments.base_model)
    if "finetuned" in arguments.targets:
        required_paths.append(arguments.finetuned_model)
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"输入路径检查失败: 路径不存在, 上下文: path={required_path}")
    return arguments


# ========== 文件与日志 ==========


def configure_logging(run_directory: Path) -> logging.Logger:
    """创建同时写入控制台和运行日志的记录器。

    Args:
        run_directory (Path): 本次运行的根目录，由命令行参数解析得到。

    Returns:
        logging.Logger: 名为 ``general_benchmark`` 的已配置记录器。

    Raises:
        RuntimeError: 输出目录或日志文件无法创建时抛出。
    """

    # 步骤1: 创建运行目录，确保后续清单与模型日志有稳定位置。
    try:
        run_directory.mkdir(parents=True, exist_ok=True)
    except OSError as exception:
        raise RuntimeError(f"运行目录创建失败: {exception}, 上下文: run_directory={run_directory}") from exception

    # 步骤2: 重建处理器，避免测试或重复调用 main 时产生重复日志。
    logger = logging.getLogger("general_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    try:
        file_handler = logging.FileHandler(run_directory / "runner.log", encoding="utf-8")
    except OSError as exception:
        raise RuntimeError(f"日志文件创建失败: {exception}, 上下文: run_directory={run_directory}") from exception
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def write_json_file(output_path: Path, payload: Any) -> None:
    """以 UTF-8 和原子替换方式写入结构化 JSON 文件。

    Args:
        output_path (Path): 待写入的目标路径，由运行目录派生。
        payload (Any): 可被 ``json.dump`` 序列化的清单、状态或汇总对象。

    Returns:
        None: 成功后目标文件包含完整 JSON 内容。

    Raises:
        RuntimeError: 目录创建、序列化、临时文件写入或替换失败时抛出。
    """

    # 步骤1: 先写同目录临时文件，防止进程中断留下半个 JSON。
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with temporary_path.open("w", encoding="utf-8") as output_handle:
            json.dump(payload, output_handle, ensure_ascii=False, indent=2)
            output_handle.write("\n")
        os.replace(temporary_path, output_path)
    except (OSError, TypeError, ValueError) as exception:
        raise RuntimeError(f"JSON 文件写入失败: {exception}, 上下文: output_path={output_path}") from exception


def append_json_line(output_path: Path, payload: dict[str, Any]) -> None:
    """向运行事件日志追加一条 JSONL 记录。

    Args:
        output_path (Path): 事件日志路径，由本次运行目录产生。
        payload (dict[str, Any]): 包含模型、数据集、时间和状态的单条事件。

    Returns:
        None: 成功后事件以一行 JSON 追加到文件末尾。

    Raises:
        RuntimeError: 目录创建、JSON 编码或文件追加失败时抛出。
    """

    # 步骤1: 每个状态立即落盘，保证异常退出后仍能审计已完成工作。
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as output_handle:
            output_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except (OSError, TypeError, ValueError) as exception:
        raise RuntimeError(f"事件日志追加失败: {exception}, 上下文: output_path={output_path}") from exception


# ========== API 与 vLLM 生命周期 ==========


def build_api_base_url(host: str, port: int) -> str:
    """构造 EvalScope 使用的 OpenAI 兼容 API 根地址。

    Args:
        host (str): vLLM 监听主机，由 ``--host`` 传入。
        port (int): vLLM TCP 端口，由 ``--port`` 传入。

    Returns:
        str: 形如 ``http://127.0.0.1:8000/v1`` 的地址。

    Raises:
        ValueError: 主机字符串为空时抛出。
    """

    # 步骤1: 评测客户端和服务同机运行，仅允许明确的非空主机。
    normalized_host = host.strip()
    if not normalized_host:
        raise ValueError(f"API 地址构造失败: host 为空, 上下文: host={host!r}")
    return f"http://{normalized_host}:{port}/v1"


def query_served_model_names(api_base_url: str, timeout_seconds: float) -> list[str]:
    """读取 OpenAI ``/models`` 接口公开的模型名称。

    Args:
        api_base_url (str): 由 ``build_api_base_url`` 生成的 vLLM 根地址。
        timeout_seconds (float): 单次健康检查超时，由启动等待逻辑传入。

    Returns:
        list[str]: 接口响应 ``data`` 中所有非空模型 id。

    Raises:
        RuntimeError: HTTP、网络、JSON 解码或响应结构异常时抛出。
    """

    # 步骤1: 通过标准模型列表验证服务身份，而不是只检查端口是否开放。
    request = Request(f"{api_base_url.rstrip('/')}/models", method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exception:
        raise RuntimeError(f"模型列表请求失败: {exception}, 上下文: api_base_url={api_base_url}") from exception
    model_rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(model_rows, list):
        raise RuntimeError(f"模型列表响应解析失败: data 不是列表, 上下文: payload={payload}")
    return [str(row.get("id")) for row in model_rows if isinstance(row, dict) and row.get("id")]


def ensure_api_port_is_free(api_base_url: str) -> None:
    """拒绝在目标端口已有 OpenAI 服务时启动新模型。

    Args:
        api_base_url (str): 本次计划使用的 vLLM API 根地址。

    Returns:
        None: 无服务响应时允许后续启动。

    Raises:
        RuntimeError: 端口上已有可响应的 OpenAI 兼容服务时抛出。
    """

    # 步骤1: 网络失败代表端口可用于本次受管服务；成功响应则可能评错模型。
    try:
        existing_models = query_served_model_names(api_base_url, timeout_seconds=1.5)
    except RuntimeError:
        return
    raise RuntimeError(f"vLLM 启动前检查失败: API 端口已被占用, 上下文: api={api_base_url}, models={existing_models}")


def build_vllm_command(arguments: argparse.Namespace, target: ModelTarget) -> list[str]:
    """构造固定解码环境下的 vLLM 服务命令。

    Args:
        arguments (argparse.Namespace): 已验证的资源和网络参数。
        target (ModelTarget): 当前即将加载的基座或微调模型。

    Returns:
        list[str]: 可直接传给 ``subprocess.Popen`` 的参数列表。

    Raises:
        RuntimeError: 正常构造不会抛出；路径问题已在参数验证阶段处理。
    """

    # 步骤1: 显式禁用模型目录中的 generation_config，保证两者仅权重不同。
    return [
        str(arguments.vllm_executable),
        "serve",
        str(target.model_path),
        "--served-model-name",
        target.served_name,
        "--host",
        arguments.host,
        "--port",
        str(arguments.port),
        "--dtype",
        "bfloat16",
        "--generation-config",
        "vllm",
        "--max-model-len",
        str(arguments.max_model_len),
        "--gpu-memory-utilization",
        str(arguments.gpu_memory_utilization),
        "--max-num-seqs",
        str(arguments.max_num_seqs),
        "--enable-prefix-caching",
        "--trust-remote-code",
    ]


def build_vllm_environment(vllm_executable: Path) -> dict[str, str]:
    """构造优先解析 vLLM 环境二进制工具的子进程环境。

    Args:
        vllm_executable (Path): 已验证的 vLLM CLI 路径，由 ``--vllm-executable``
            传入，其父目录同时包含 ninja 等运行时 JIT 工具。

    Returns:
        dict[str, str]: 复制自当前进程且已将 vLLM ``bin`` 放在 PATH 首位的环境。

    Raises:
        RuntimeError: 正常构造不会抛出；可执行文件存在性已在参数验证阶段检查。
    """

    # 步骤1: 独立 EvalScope 环境启动 vLLM 时，shebang 不会自动切换子进程 PATH。
    child_environment = os.environ.copy()
    existing_path = child_environment.get("PATH", "")
    child_environment["PATH"] = os.pathsep.join([str(vllm_executable.parent), existing_path])
    return child_environment


def start_vllm_server(
    arguments: argparse.Namespace,
    target: ModelTarget,
    api_base_url: str,
    logger: logging.Logger,
) -> tuple[subprocess.Popen[bytes], Any]:
    """启动 vLLM 并等待目标模型出现在 ``/models`` 中。

    Args:
        arguments (argparse.Namespace): 已验证的启动参数和运行目录。
        target (ModelTarget): 当前待部署模型的路径与服务名。
        api_base_url (str): 健康检查使用的 OpenAI API 根地址。
        logger (logging.Logger): 由 ``configure_logging`` 创建的运行记录器。

    Returns:
        tuple[subprocess.Popen[bytes], Any]: vLLM 进程和保持打开的二进制日志句柄。

    Raises:
        RuntimeError: 日志创建、进程启动、提前退出或等待超时时抛出。
    """

    # 步骤1: 在独立进程组中启动服务，以便结束时清理其工作进程。
    ensure_api_port_is_free(api_base_url)
    server_log_path = arguments.run_dir / "vllm_logs" / f"{target.role}.log"
    try:
        server_log_path.parent.mkdir(parents=True, exist_ok=True)
        server_log_handle = server_log_path.open("ab", buffering=0)
        server_process = subprocess.Popen(
            build_vllm_command(arguments, target),
            stdout=server_log_handle,
            stderr=subprocess.STDOUT,
            env=build_vllm_environment(arguments.vllm_executable),
            start_new_session=True,
        )
    except OSError as exception:
        raise RuntimeError(f"vLLM 进程启动失败: {exception}, 上下文: target={target.role}, log={server_log_path}") from exception

    # 步骤2: 轮询模型身份；Qwen3.5 首次加载耗时较长，不能用固定睡眠判断。
    deadline = time.monotonic() + arguments.server_start_timeout
    logger.info("正在启动 %s: model=%s, log=%s", target.role, target.model_path, server_log_path)
    while time.monotonic() < deadline:
        if server_process.poll() is not None:
            server_log_handle.close()
            raise RuntimeError(
                f"vLLM 启动失败: 服务进程提前退出, 上下文: target={target.role}, "
                f"return_code={server_process.returncode}, log={server_log_path}"
            )
        try:
            served_names = query_served_model_names(api_base_url, timeout_seconds=2.0)
            if target.served_name in served_names:
                logger.info("vLLM 已就绪: target=%s, served_name=%s", target.role, target.served_name)
                return server_process, server_log_handle
        except RuntimeError:
            time.sleep(2.0)
    stop_vllm_server(server_process, server_log_handle, logger)
    raise RuntimeError(
        f"vLLM 启动超时: 未发现目标模型, 上下文: target={target.role}, "
        f"timeout={arguments.server_start_timeout}, log={server_log_path}"
    )


def stop_vllm_server(server_process: subprocess.Popen[bytes], server_log_handle: Any, logger: logging.Logger) -> None:
    """终止 vLLM 主进程组并关闭服务日志。

    Args:
        server_process (subprocess.Popen[bytes]): ``start_vllm_server`` 返回的进程。
        server_log_handle (Any): 与该进程绑定的二进制日志文件句柄。
        logger (logging.Logger): 当前运行记录器，用于报告退出状态。

    Returns:
        None: 返回时主进程已退出，日志句柄已关闭。

    Raises:
        RuntimeError: 进程组无法在 SIGTERM 和 SIGKILL 后停止时抛出。
    """

    # 步骤1: 优先优雅停止整个进程组，使 vLLM worker 正常释放共享内存和显存。
    try:
        if server_process.poll() is None:
            os.killpg(server_process.pid, signal.SIGTERM)
            server_process.wait(timeout=45)
    except (OSError, subprocess.TimeoutExpired):
        # 步骤2: 超时后强制终止，防止下一模型因残留 worker 发生 OOM。
        try:
            os.killpg(server_process.pid, signal.SIGKILL)
            server_process.wait(timeout=15)
        except (OSError, subprocess.TimeoutExpired) as exception:
            raise RuntimeError(
                f"vLLM 进程停止失败: {exception}, 上下文: pid={server_process.pid}"
            ) from exception
    finally:
        try:
            server_log_handle.close()
        except OSError:
            pass
    logger.info("vLLM 已停止: pid=%s, return_code=%s", server_process.pid, server_process.returncode)


def probe_chat_completion(api_base_url: str, target: ModelTarget, timeout_seconds: int) -> dict[str, Any]:
    """发送一条非计分请求，确认聊天模板和非思考模式可用。

    Args:
        api_base_url (str): 已就绪的 vLLM OpenAI API 根地址。
        target (ModelTarget): 请求中的模型服务名来源。
        timeout_seconds (int): HTTP 请求超时，由 ``--request-timeout`` 提供。

    Returns:
        dict[str, Any]: 包含回复文本、finish reason 和 token 用量的诊断信息。

    Raises:
        RuntimeError: 请求、JSON 解码或响应字段提取失败时抛出。
    """

    # 步骤1: 显式传递 enable_thinking=False，验证 EvalScope 后续依赖的服务能力。
    request_payload = {
        "model": target.served_name,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = Request(
        f"{api_base_url.rstrip('/')}/chat/completions",
        data=json.dumps(request_payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        first_choice = response_payload["choices"][0]
        return {
            "content": first_choice["message"].get("content"),
            "reasoning_content": first_choice["message"].get("reasoning_content"),
            "finish_reason": first_choice.get("finish_reason"),
            "usage": response_payload.get("usage"),
        }
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exception:
        raise RuntimeError(
            f"聊天接口探测失败: {exception}, 上下文: target={target.role}, api={api_base_url}"
        ) from exception


# ========== EvalScope 任务 ==========


def verify_evalscope_version() -> str:
    """导入 EvalScope 并确认版本与论文协议一致。

    Returns:
        str: 实际导入的 EvalScope 版本，正常为 ``1.4.2``。

    Raises:
        RuntimeError: EvalScope 不可导入或版本不是要求值时抛出。
    """

    # 步骤1: 延迟导入，使 ``--help`` 和静态检查不依赖服务器环境。
    try:
        import evalscope
    except ImportError as exception:
        raise RuntimeError(f"EvalScope 导入失败: {exception}, 上下文: required={EVALSCOPE_REQUIRED_VERSION}") from exception
    actual_version = str(getattr(evalscope, "__version__", "unknown"))
    if actual_version != EVALSCOPE_REQUIRED_VERSION:
        raise RuntimeError(
            f"EvalScope 版本检查失败: 版本不一致, "
            f"上下文: required={EVALSCOPE_REQUIRED_VERSION}, actual={actual_version}"
        )
    return actual_version


def build_dataset_arguments(specification: BenchmarkSpec, mode: str) -> dict[str, dict[str, Any]]:
    """构造单个数据集的 few-shot 与冒烟子集参数。

    Args:
        specification (BenchmarkSpec): ``BENCHMARK_SPECS`` 中的固定协议项。
        mode (str): ``smoke`` 或 ``full``，由命令行调用者选择。

    Returns:
        dict[str, dict[str, Any]]: 可直接传给 EvalScope ``dataset_args`` 的映射。

    Raises:
        ValueError: mode 不是支持值时抛出。
    """

    # 步骤1: 固定 few-shot 顺序，避免两个模型抽到不同示例。
    if mode not in {"smoke", "full"}:
        raise ValueError(f"数据集参数构造失败: mode 无效, 上下文: mode={mode}")
    options: dict[str, Any] = {
        "few_shot_num": specification.few_shot_num,
        "few_shot_random": False,
    }

    # 步骤2: 冒烟阶段仅取一个子集，确保六项协议能在几分钟内完成验证。
    if mode == "smoke":
        options["subset_list"] = [specification.smoke_subset]
    return {specification.name: options}


def locate_report_file(work_directory: Path, served_name: str, dataset_name: str) -> Path:
    """定位 EvalScope 为单模型单数据集生成的正式报告。

    Args:
        work_directory (Path): 传给 TaskConfig 的独立输出目录。
        served_name (str): 报告目录使用的模型服务名。
        dataset_name (str): 报告文件使用的数据集注册名。

    Returns:
        Path: 预期的 ``reports/<model>/<dataset>.json`` 路径。

    Raises:
        RuntimeError: 预期路径不存在且递归搜索也不能唯一定位报告时抛出。
    """

    # 步骤1: 优先使用 EvalScope 1.4.2 的固定目录结构。
    expected_path = work_directory / "reports" / served_name / f"{dataset_name}.json"
    if expected_path.exists():
        return expected_path

    # 步骤2: 保留唯一回退搜索，以兼容模型名被框架清理的情况但拒绝歧义。
    candidate_paths = list(work_directory.glob(f"**/reports/**/{dataset_name}.json"))
    if len(candidate_paths) != 1:
        raise RuntimeError(
            f"EvalScope 报告定位失败: 候选数量不是 1, "
            f"上下文: work_directory={work_directory}, candidates={candidate_paths}"
        )
    return candidate_paths[0]


def parse_evalscope_report(report_path: Path, target_role: str) -> dict[str, Any]:
    """提取 EvalScope 报告的总体分数、样本数和指标名。

    Args:
        report_path (Path): ``locate_report_file`` 返回的报告 JSON。
        target_role (str): 当前模型角色，用于汇总和错误上下文。

    Returns:
        dict[str, Any]: 标准化后的 role、dataset、score、num、metric 与路径。

    Raises:
        RuntimeError: 文件读取、JSON 解码或关键字段结构异常时抛出。
    """

    # 步骤1: 读取顶层报告并取第一个总体指标，与旧论文结果口径保持一致。
    try:
        with report_path.open("r", encoding="utf-8") as report_handle:
            report_payload = json.load(report_handle)
        metrics = report_payload["metrics"]
        first_metric = metrics[0]
        return {
            "role": target_role,
            "model_id": report_payload.get("model_name"),
            "dataset": str(report_payload["dataset_name"]),
            "score": float(report_payload["score"]),
            "num": int(first_metric["num"]),
            "metric": str(first_metric["name"]),
            "report_path": str(report_path),
        }
    except (OSError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exception:
        raise RuntimeError(f"EvalScope 报告解析失败: {exception}, 上下文: report_path={report_path}") from exception


def run_evalscope_dataset(
    arguments: argparse.Namespace,
    target: ModelTarget,
    specification: BenchmarkSpec,
    api_base_url: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    """执行或复用一个模型在一个数据集上的 EvalScope 评测。

    Args:
        arguments (argparse.Namespace): 运行模式、批大小、输出目录和解码参数。
        target (ModelTarget): 当前已部署模型的角色、服务名和路径。
        specification (BenchmarkSpec): 当前数据集的固定协议。
        api_base_url (str): vLLM 的 OpenAI API 根地址。
        logger (logging.Logger): 运行记录器，用于显示数据集级进度。

    Returns:
        dict[str, Any]: 从正式报告标准化得到的数据集汇总。

    Raises:
        RuntimeError: EvalScope 导入、任务运行或报告定位解析失败时抛出。
    """

    # 步骤1: 为每个模型和数据集使用独立目录，支持安全续跑和原始回答审计。
    work_directory = arguments.run_dir / "evalscope" / target.role / specification.name
    expected_report = work_directory / "reports" / target.served_name / f"{specification.name}.json"
    if expected_report.exists() and not arguments.force:
        logger.info("复用已有报告: target=%s, dataset=%s", target.role, specification.name)
        return parse_evalscope_report(expected_report, target.role)

    # 步骤2: 构造与旧论文实验一致的确定性生成和 few-shot 参数。
    try:
        from evalscope import TaskConfig, run_task
    except ImportError as exception:
        raise RuntimeError(f"EvalScope 任务导入失败: {exception}, 上下文: dataset={specification.name}") from exception
    sample_limit = 2 if arguments.mode == "smoke" else specification.full_limit
    task_configuration = TaskConfig(
        model=target.served_name,
        model_id=target.served_name,
        datasets=[specification.name],
        dataset_args=build_dataset_arguments(specification, arguments.mode),
        limit=sample_limit,
        generation_config={
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": arguments.max_tokens,
            "timeout": float(arguments.request_timeout),
            "chat_template_kwargs": {"enable_thinking": False},
        },
        eval_type="openai_api",
        api_url=api_base_url,
        api_key="EMPTY",
        eval_batch_size=arguments.eval_batch_size,
        work_dir=str(work_directory),
        no_timestamp=True,
        seed=RANDOM_SEED,
    )

    # 步骤3: 让 EvalScope 保留 predictions、reviews、config 和 report 全套证据。
    logger.info(
        "开始数据集: target=%s, dataset=%s, limit=%s, few_shot=%s",
        target.role,
        specification.name,
        sample_limit,
        specification.few_shot_num,
    )
    try:
        run_task(task_configuration)
    except Exception as exception:
        raise RuntimeError(
            f"EvalScope 数据集执行失败: {exception}, "
            f"上下文: target={target.role}, dataset={specification.name}, work_directory={work_directory}"
        ) from exception
    report_path = locate_report_file(work_directory, target.served_name, specification.name)
    summary = parse_evalscope_report(report_path, target.role)
    logger.info(
        "完成数据集: target=%s, dataset=%s, score=%.4f, num=%s",
        target.role,
        specification.name,
        summary["score"],
        summary["num"],
    )
    return summary


# ========== 运行编排与汇总 ==========


def build_model_targets(arguments: argparse.Namespace) -> list[ModelTarget]:
    """按命令行顺序构造本次待评估模型列表。

    Args:
        arguments (argparse.Namespace): 包含 ``targets`` 与两个已解析模型路径。

    Returns:
        list[ModelTarget]: 与 ``--targets`` 顺序一致的模型目标列表。

    Raises:
        ValueError: 出现参数解析器之外的未知角色时抛出。
    """

    # 步骤1: 使用稳定服务名，让两个模型的报告目录可以直接区分。
    target_by_role = {
        "base": ModelTarget("base", "Qwen3.5-4B-Base", arguments.base_model),
        "finetuned": ModelTarget("finetuned", "Qwen3.5-4B-Finetuned-Step700", arguments.finetuned_model),
    }
    try:
        return [target_by_role[role] for role in arguments.targets]
    except KeyError as exception:
        raise ValueError(f"模型目标构造失败: 未知角色, 上下文: targets={arguments.targets}") from exception


def build_run_manifest(
    arguments: argparse.Namespace,
    targets: Sequence[ModelTarget],
    evalscope_version: str,
) -> dict[str, Any]:
    """生成记录完整协议与运行环境的可复现清单。

    Args:
        arguments (argparse.Namespace): 已验证的全部命令行配置。
        targets (Sequence[ModelTarget]): 本次实际会部署的模型列表。
        evalscope_version (str): ``verify_evalscope_version`` 验证后的版本。

    Returns:
        dict[str, Any]: 可写入 ``manifest.json`` 的结构化配置快照。

    Raises:
        RuntimeError: 正常构造不会抛出；序列化异常由写文件函数处理。
    """

    # 步骤1: 同时保存正式限制与当前模式，防止 smoke 结果被误认为完整结果。
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": arguments.mode,
        "status": "running",
        "evalscope_version": evalscope_version,
        "seed": RANDOM_SEED,
        "targets": [
            {"role": target.role, "served_name": target.served_name, "model_path": str(target.model_path)}
            for target in targets
        ],
        "datasets": [asdict(BENCHMARK_SPECS[name]) for name in arguments.datasets],
        "generation_config": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": arguments.max_tokens,
            "enable_thinking": False,
        },
        "runtime": {
            "api_base_url": build_api_base_url(arguments.host, arguments.port),
            "eval_batch_size": arguments.eval_batch_size,
            "max_num_seqs": arguments.max_num_seqs,
            "max_model_len": arguments.max_model_len,
            "gpu_memory_utilization": arguments.gpu_memory_utilization,
            "vllm_executable": str(arguments.vllm_executable),
        },
    }


def collect_existing_summaries(arguments: argparse.Namespace, targets: Sequence[ModelTarget]) -> list[dict[str, Any]]:
    """扫描本次运行目录中已经生成的 EvalScope 报告。

    Args:
        arguments (argparse.Namespace): 提供运行目录和数据集选择。
        targets (Sequence[ModelTarget]): 提供角色及服务名对应关系。

    Returns:
        list[dict[str, Any]]: 所有可成功解析的模型数据集汇总。

    Raises:
        RuntimeError: 发现报告但其 JSON 内容损坏或结构异常时抛出。
    """

    # 步骤1: 按目标和数据集固定顺序读取，保证 comparison.json 稳定可比较。
    summaries: list[dict[str, Any]] = []
    for target in targets:
        for dataset_name in arguments.datasets:
            report_path = (
                arguments.run_dir
                / "evalscope"
                / target.role
                / dataset_name
                / "reports"
                / target.served_name
                / f"{dataset_name}.json"
            )
            if report_path.exists():
                summaries.append(parse_evalscope_report(report_path, target.role))
    return summaries


def build_comparison(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """将两模型报告按数据集配对并计算微调前后差值。

    Args:
        summaries (Sequence[dict[str, Any]]): ``collect_existing_summaries`` 的结果。

    Returns:
        list[dict[str, Any]]: 每个数据集的 base、finetuned 与绝对分数变化。

    Raises:
        RuntimeError: 正常汇总不会抛出；缺失模型时对应分数字段为 ``None``。
    """

    # 步骤1: 先按数据集和角色建立索引，允许单模型运行也生成阶段性汇总。
    summary_index = {(item["dataset"], item["role"]): item for item in summaries}
    comparison_rows: list[dict[str, Any]] = []
    dataset_names = sorted({item["dataset"] for item in summaries}, key=list(BENCHMARK_SPECS).index)
    for dataset_name in dataset_names:
        base_summary = summary_index.get((dataset_name, "base"))
        finetuned_summary = summary_index.get((dataset_name, "finetuned"))
        base_score = base_summary["score"] if base_summary else None
        finetuned_score = finetuned_summary["score"] if finetuned_summary else None
        score_delta = finetuned_score - base_score if base_score is not None and finetuned_score is not None else None
        comparison_rows.append(
            {
                "dataset": dataset_name,
                "base_score": base_score,
                "finetuned_score": finetuned_score,
                "delta": score_delta,
                "base_num": base_summary["num"] if base_summary else None,
                "finetuned_num": finetuned_summary["num"] if finetuned_summary else None,
            }
        )
    return comparison_rows


def persist_progress(arguments: argparse.Namespace, targets: Sequence[ModelTarget]) -> list[dict[str, Any]]:
    """重建并写入当前运行的报告索引和配对比较。

    Args:
        arguments (argparse.Namespace): 提供运行目录与数据集范围。
        targets (Sequence[ModelTarget]): 提供待扫描的模型目标。

    Returns:
        list[dict[str, Any]]: 当前已完成报告的标准化汇总。

    Raises:
        RuntimeError: 报告解析或 JSON 汇总文件写入失败时抛出。
    """

    # 步骤1: 每完成一个数据集就刷新两个文件，使长时间全量评测可随时查看。
    summaries = collect_existing_summaries(arguments, targets)
    write_json_file(arguments.run_dir / "summary.json", summaries)
    write_json_file(arguments.run_dir / "comparison.json", build_comparison(summaries))
    return summaries


def evaluate_model_target(
    arguments: argparse.Namespace,
    target: ModelTarget,
    all_targets: Sequence[ModelTarget],
    api_base_url: str,
    logger: logging.Logger,
) -> None:
    """部署一个模型并依次执行用户选择的所有基准。

    Args:
        arguments (argparse.Namespace): 已验证的运行、模型和数据集参数。
        target (ModelTarget): 当前需要部署和评测的单个模型。
        all_targets (Sequence[ModelTarget]): 用于生成跨模型阶段性汇总的完整列表。
        api_base_url (str): 本次受管 vLLM 服务地址。
        logger (logging.Logger): 运行记录器。

    Returns:
        None: 所选数据集完成后结果已写入各自目录和总汇总。

    Raises:
        RuntimeError: 服务启动、探测、EvalScope 执行或服务清理失败时抛出。
    """

    # 步骤1: 模型生命周期由 try/finally 包裹，任何数据集失败都释放 GPU。
    server_process, server_log_handle = start_vllm_server(arguments, target, api_base_url, logger)
    try:
        probe_result = probe_chat_completion(api_base_url, target, arguments.request_timeout)
        write_json_file(arguments.run_dir / "probes" / f"{target.role}.json", probe_result)
        logger.info("聊天探测完成: target=%s, content=%r", target.role, probe_result.get("content"))

        # 步骤2: 按清单顺序执行并在每项结束后记录事件和汇总。
        for dataset_name in arguments.datasets:
            started_at = datetime.now().astimezone().isoformat(timespec="seconds")
            summary = run_evalscope_dataset(arguments, target, BENCHMARK_SPECS[dataset_name], api_base_url, logger)
            append_json_line(
                arguments.run_dir / "events.jsonl",
                {
                    "status": "completed",
                    "started_at": started_at,
                    "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    **summary,
                },
            )
            persist_progress(arguments, all_targets)
    finally:
        stop_vllm_server(server_process, server_log_handle, logger)


def run_benchmarks(arguments: argparse.Namespace, logger: logging.Logger) -> None:
    """执行完整的多模型通用基准运行并维护清单状态。

    Args:
        arguments (argparse.Namespace): ``resolve_and_validate_arguments`` 的结果。
        logger (logging.Logger): 已绑定本次运行目录的日志记录器。

    Returns:
        None: 成功时清单状态更新为 ``completed`` 并输出最终比较。

    Raises:
        RuntimeError: 版本、服务、数据集或结果持久化任一步失败时抛出。
    """

    # 步骤1: 在启动 GPU 任务前锁定框架版本和完整运行清单。
    evalscope_version = verify_evalscope_version()
    targets = build_model_targets(arguments)
    manifest = build_run_manifest(arguments, targets, evalscope_version)
    write_json_file(arguments.run_dir / "manifest.json", manifest)
    api_base_url = build_api_base_url(arguments.host, arguments.port)
    logger.info("运行开始: mode=%s, targets=%s, datasets=%s", arguments.mode, arguments.targets, arguments.datasets)

    # 步骤2: 串行部署模型，保证单卡显存与请求协议对称。
    try:
        for target in targets:
            evaluate_model_target(arguments, target, targets, api_base_url, logger)
        summaries = persist_progress(arguments, targets)
        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        manifest["completed_reports"] = len(summaries)
        write_json_file(arguments.run_dir / "manifest.json", manifest)
    except Exception:
        manifest["status"] = "failed"
        manifest["failed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        write_json_file(arguments.run_dir / "manifest.json", manifest)
        raise
    logger.info("运行完成: reports=%s, comparison=%s", len(summaries), arguments.run_dir / "comparison.json")


def main(command_line_arguments: Sequence[str] | None = None) -> int:
    """解析命令行并执行 Qwen3.5 通用基准比较。

    Args:
        command_line_arguments (Sequence[str] | None): 测试可显式传入的参数序列；
            ``None`` 时由 Python 从终端 ``sys.argv`` 读取。

    Returns:
        int: 成功为 0；可预期的评测或输入异常为 1。

    Raises:
        RuntimeError: 已在本函数转为日志和返回码，不向命令行继续传播。
        ValueError: 已在本函数转为日志和返回码，不向命令行继续传播。
    """

    # 步骤1: 参数验证失败时仍向终端给出带上下文的明确错误。
    parser = build_argument_parser()
    try:
        arguments = resolve_and_validate_arguments(parser.parse_args(command_line_arguments))
        logger = configure_logging(arguments.run_dir)
        run_benchmarks(arguments, logger)
    except (FileNotFoundError, RuntimeError, ValueError) as exception:
        logging.getLogger("general_benchmark").exception("通用基准评测失败: %s", exception)
        return 1
    return 0


if __name__ == "__main__":
    # 用法示例: python src/eval_benchmark.py --mode smoke --targets base finetuned
    raise SystemExit(main())
