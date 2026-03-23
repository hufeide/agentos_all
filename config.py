"""
AgentOS 配置模块
支持从环境变量加载配置，提供配置验证和默认值管理
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentOSConfig:
    """AgentOS 配置数据类"""

    # LLM 配置
    vllm_url: str = "http://192.168.1.210:19000"
    model_name: str = "Qwen3Qoder"

    # 日志配置
    log_level: str = "INFO"

    # 搜索配置
    search_engine: str = "tavily"
    tavily_api_key: str = "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"

    # 工作配置
    worker_count: int = 2
    skills_dir: str = "skills"

    # 超时配置（秒）
    global_timeout: float = 300.0
    llm_call_timeout: float = 60.0
    step_execution_timeout: float = 120.0

    # 事件总线配置
    bus_max_queue_size: int = 1000
    event_handler_retry_enabled: bool = False
    event_handler_max_retries: int = 3
    dead_letter_queue_enabled: bool = False

    # 清理配置（秒）
    projection_cleanup_delay: int = 300
    cleanup_check_interval: int = 60

    # 重试策略配置
    max_step_retries: int = 3
    claim_retry_base: float = 0.1
    claim_retry_max: float = 0.3

    @classmethod
    def from_env(cls) -> 'AgentOSConfig':
        """从环境变量加载配置"""
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if log_level not in valid_log_levels:
            log_level = "INFO"

        worker_count_str = os.environ.get("WORKER_COUNT", "2")
        try:
            worker_count = int(worker_count_str)
            worker_count = max(1, min(worker_count, 10))  # 限制范围
        except ValueError:
            worker_count = 2

        return cls(
            vllm_url=os.environ.get("VLLM_URL", "http://192.168.1.210:19000"),
            model_name=os.environ.get("MODEL_NAME", "Qwen3Qoder"),
            log_level=log_level,
            tavily_api_key=os.environ.get("TAVILY_API_KEY", "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"),
            worker_count=worker_count,
            skills_dir=os.environ.get("SKILLS_DIR", "skills"),
            global_timeout=float(os.environ.get("GLOBAL_TIMEOUT", "300")),
            llm_call_timeout=float(os.environ.get("LLM_CALL_TIMEOUT", "60")),
            step_execution_timeout=float(os.environ.get("STEP_EXECUTION_TIMEOUT", "120")),
            bus_max_queue_size=int(os.environ.get("BUS_MAX_QUEUE_SIZE", "1000")),
            projection_cleanup_delay=int(os.environ.get("PROJECTION_CLEANUP_DELAY", "300")),
            cleanup_check_interval=int(os.environ.get("CLEANUP_CHECK_INTERVAL", "60")),
            event_handler_retry_enabled=os.environ.get("EVENT_HANDLER_RETRY_ENABLED", "false").lower() == "true",
            event_handler_max_retries=int(os.environ.get("EVENT_HANDLER_MAX_RETRIES", "3")),
            dead_letter_queue_enabled=os.environ.get("DEAD_LETTER_QUEUE_ENABLED", "false").lower() == "true",
            max_step_retries=int(os.environ.get("MAX_STEP_RETRIES", "3")),
            claim_retry_base=float(os.environ.get("CLAIM_RETRY_BASE", "0.1")),
            claim_retry_max=float(os.environ.get("CLAIM_RETRY_MAX", "0.3")),
        )


# 全局配置实例
config = AgentOSConfig.from_env()
