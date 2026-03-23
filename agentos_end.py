import asyncio
import uuid
import time
import logging
import json
import os
from enum import Enum
from typing import Dict, Any, List, Set, Optional, Callable, Protocol, runtime_checkable, Tuple
from dataclasses import dataclass, field
from openai import AsyncOpenAI
import httpx

# 尝试导入 PyYAML 用于动态 Skill 的 frontmatter 解析
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# =========================
# 配置管理
# =========================
class ConfigManager:
    def __init__(self):
        self._config = None
        try:
            from config import config
            self._config = config
        except ImportError:
            pass

        # 先从 self._config 读取，再 fallback 到环境变量
        self.MAX_STEP_RETRIES = self._get_config_or_env("MAX_STEP_RETRIES", int, 3)
        self.GLOBAL_TIMEOUT = self._get_config_or_env("GLOBAL_TIMEOUT", int, 300)
        self.STEP_EXECUTION_TIMEOUT = self._get_config_or_env("STEP_EXECUTION_TIMEOUT", int, 120)
        self.STEP_READY_TIMEOUT = self._get_config_or_env("STEP_READY_TIMEOUT", float, 5.0)
        self.QUEUE_TIMEOUT = self._get_config_or_env("QUEUE_TIMEOUT", float, 1.0)
        self.LLM_CALL_TIMEOUT = self._get_config_or_env("LLM_CALL_TIMEOUT", float, 60.0)
        self.LLM_MAX_TOKENS = self._get_config_or_env("LLM_MAX_TOKENS", int, 6000)
        self.LLM_TEMPERATURE = self._get_config_or_env("LLM_TEMPERATURE", float, 0.3)
        self.EVENT_BUS_MAX_QUEUE_SIZE = self._get_config_or_env("EVENT_BUS_MAX_QUEUE_SIZE", int, 1000)
        self.DEAD_LETTER_QUEUE_ENABLED = self._get_config_or_env("DEAD_LETTER_QUEUE_ENABLED", bool, False)
        # UUID_TRUNC_LENGTH 已弃用：不再使用此配置
        self.STEP_OUTPUT_SUMMARY_LEN = self._get_config_or_env("STEP_OUTPUT_SUMMARY_LEN", int, 200)
        # 关键配置项：必须有非空值，否则在初始化时提前报错
        self.VLLM_URL = self._get_config_or_env("VLLM_URL", str, "http://192.168.1.159:19000")
        self.MODEL_NAME = self._get_config_or_env("MODEL_NAME", str, "Qwen3Coder")
        self.LOG_LEVEL = self._get_config_or_env("LOG_LEVEL", str, "INFO")
        self.TAVILY_API_KEY = self._get_config_or_env("TAVILY_API_KEY", str, "")
        self.WORKER_COUNT = self._get_config_or_env("WORKER_COUNT", int, 2)
        self.SKILLS_DIR = self._get_config_or_env("SKILLS_DIR", str, "skills")

        # 关键变量校验：启动时检查必需配置
        self._validate_required_config()

    def _get_config_or_env(self, key: str, value_type: type, default: Any) -> Any:
        """优先从 self._config 读取，再 fallback 到环境变量，最后使用默认值"""
        # 1. 优先从 self._config 读取
        if self._config is not None:
            config_val = getattr(self._config, key, None)
            if config_val is not None:
                return config_val

        # 2. fallback 到环境变量
        env_val = os.getenv(key)
        if env_val is not None and env_val != "":
            try:
                if value_type == int:
                    return int(env_val)
                elif value_type == float:
                    return float(env_val)
                elif value_type == bool:
                    return env_val.lower() == "true"
                else:  # str
                    return env_val
            except (ValueError, AttributeError):
                pass  # 类型转换失败，使用默认值

        # 3. 使用默认值
        return default

    def _validate_required_config(self):
        """校验必需配置项，如果为空则提前报错"""
        required_configs = {
            "VLLM_URL": self.VLLM_URL,
            "MODEL_NAME": self.MODEL_NAME,
        }
        missing = [k for k, v in required_configs.items() if not v or v.strip() == ""]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"必需配置项缺失: {missing_str}。请在 config.py 中设置或设置环境变量。")

config_manager = ConfigManager()
AgentOSDefaults = config_manager
VLLM_URL = config_manager.VLLM_URL
MODEL_NAME = config_manager.MODEL_NAME
LOG_LEVEL = config_manager.LOG_LEVEL
TAVILY_API_KEY = config_manager.TAVILY_API_KEY

# =========================
# 日志
# =========================
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("ProductionAgentOS")

# =========================
# Protocol 定义
# =========================
@runtime_checkable
class SearchToolProtocol(Protocol):
    async def search(self, query: str, **kwargs) -> Dict[str, Any]: ...
    async def search_finance(self, query: str, **kwargs) -> Dict[str, Any]: ...
    def format_results(self, results: Dict[str, Any]) -> str: ...
    async def close(self) -> None: ...

@runtime_checkable
class ToolExecutorProtocol(Protocol):
    async def execute(self, tool_name: str, args: Dict[str, Any]) -> Any: ...
    def get_available_tools(self) -> List[Dict[str, Any]]: ...

@runtime_checkable
class SkillExecutorProtocol(Protocol):
    async def execute(self, skill_name: str, args: Dict[str, Any]) -> Any: ...
    def get_available_skills(self) -> List[Dict[str, Any]]: ...

# =========================
# 核心事件模型
# =========================
class EventType(str, Enum):
    TASK_SUBMITTED = "task_submitted"
    STEP_READY = "step_ready"
    STEP_CLAIMED = "step_claimed"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"

@dataclass(frozen=True)
class Event:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    step_id: Optional[str] = None
    event_type: EventType = EventType.STEP_READY
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

# =========================
# 事件总线（已修复：使用dict存储订阅者 + 安全 shutdown + dead_letter 消费）
# =========================
class EventBus:
    def __init__(self, max_queue_size: int = None):
        if max_queue_size is None:
            max_queue_size = config_manager.EVENT_BUS_MAX_QUEUE_SIZE
        self._subscribers: Dict[int, Callable] = {}  # 修复：使用dict而不是list
        self._next_subscription_id = 0
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._dead_letter_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._dead_letter_enabled = config_manager.DEAD_LETTER_QUEUE_ENABLED
        self._dead_letter_flush_task = None
        self._worker_task = asyncio.create_task(self._process_queue())
        self._stop_event = asyncio.Event()
        self._max_queue_size = max_queue_size

        # 启动 dead_letter 队列的后台消费任务（定期 flush 到日志）
        if self._dead_letter_enabled:
            self._dead_letter_flush_task = asyncio.create_task(self._flush_dead_letter_loop())

    def subscribe(self, handler: Callable) -> int:
        subscription_id = self._next_subscription_id
        self._subscribers[subscription_id] = handler
        self._next_subscription_id += 1
        return subscription_id

    def unsubscribe(self, subscription_id: int) -> bool:
        # 修复：直接删除，而不是设置为None
        if subscription_id in self._subscribers:
            del self._subscribers[subscription_id]
            return True
        return False

    async def publish(self, event: Event, block: bool = True, timeout: Optional[float] = None) -> bool:
        try:
            if block:
                if timeout is None:
                    await self._queue.put(event)
                else:
                    await asyncio.wait_for(self._queue.put(event), timeout=timeout)
            else:
                self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            if self._dead_letter_enabled:
                try:
                    await self._dead_letter_queue.put((event, "queue_full"))
                except asyncio.QueueFull:
                    pass
            return False
        except asyncio.TimeoutError:
            return False

    async def _flush_dead_letter_loop(self):
        """后台任务：定期 flush dead_letter_queue 到日志"""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(5.0)  # 每5秒 flush 一次
                while not self._dead_letter_queue.empty():
                    try:
                        item = await asyncio.wait_for(self._dead_letter_queue.get(), timeout=1.0)
                        event, reason = item if isinstance(item, tuple) else (item, "unknown")
                        logger.error(f"[DeadLetter] Event {event.event_id} dropped ({reason}): {event}")
                    except asyncio.TimeoutError:
                        break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"[DeadLetter] Flush loop error")  # 修复：添加 logger.exception

    async def _process_queue(self):
        while not self._stop_event.is_set():
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=config_manager.QUEUE_TIMEOUT)
            except asyncio.TimeoutError:
                continue
            if event is None:
                break
            # 修复：遍历副本，防止删除时的并发问题
            for handler in list(self._subscribers.values()):
                if handler is None:
                    continue
                try:
                    await handler(event)
                except Exception as e:
                    logger.exception(f"Handler {handler.__name__ if hasattr(handler, '__name__') else handler} failed: {e}")  # 修复：添加 logger.exception
            self._queue.task_done()

    async def shutdown(self):
        """安全关闭：先停止接收事件 -> drain queue -> flush dead_letter -> cancel worker"""
        logger.info("[EventBus] Shutdown initiated...")
        self._stop_event.set()

        # 1. 清空订阅者，防止内存泄漏和新事件分发
        self._subscribers.clear()

        # 2. drain queue：等待队列中的事件处理完成
        try:
            await asyncio.wait_for(self._queue.join(), timeout=2.0)
            logger.info("[EventBus] Main queue drained")
        except asyncio.TimeoutError:
            logger.warning("[EventBus] Main queue drain timeout, some events may be lost")

        # 3. flush dead_letter queue 到日志
        if self._dead_letter_enabled and self._dead_letter_flush_task:
            try:
                # 给最后一次 flush 机会
                await asyncio.sleep(1.0)
                self._dead_letter_flush_task.cancel()
                try:
                    await self._dead_letter_flush_task
                except asyncio.CancelledError:
                    pass
            except Exception as e:
                logger.exception(f"[EventBus] Error shutting down dead_letter flush task")

        # 关闭剩余的 dead_letter 队列
        try:
            if not self._dead_letter_queue.empty():
                count = 0
                while not self._dead_letter_queue.empty():
                    try:
                        item = self._dead_letter_queue.get_nowait()
                        event, reason = item if isinstance(item, tuple) else (item, "unknown")
                        logger.error(f"[DeadLetter] Unprocessed event {event.event_id} ({reason})")
                        count += 1
                    except asyncio.QueueEmpty:
                        break
                if count > 0:
                    logger.warning(f"[EventBus] {count} events lost in dead_letter queue")
        except Exception:
            pass

        # 4. 等待 worker 完成
        try:
            await asyncio.wait_for(self._worker_task, timeout=2.0)
            logger.info("[EventBus] Worker task completed")
        except asyncio.TimeoutError:
            logger.warning("[EventBus] Worker task timeout, cancelling...")
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("[EventBus] Shutdown completed")

# =========================
# 工具注册中心
# =========================
class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_schemas: Dict[str, Dict] = {}
        self.tool_descriptions: Dict[str, str] = {}

    def register(self, name: str, func: Callable, description: str, params_schema: Optional[Dict] = None):
        self.tools[name] = func
        self.tool_schemas[name] = {"type": "function", "function": {"name": name, "description": description, "parameters": params_schema or {"type": "object", "properties": {}}}}
        self.tool_descriptions[name] = description

    def get_available_tools(self) -> List[Dict]:
        return list(self.tool_schemas.values())

    async def execute(self, tool_name: str, args: Dict[str, Any]) -> Any:
        if tool_name not in self.tools:
            return f"错误：工具 '{tool_name}' 不存在"
        try:
            result = await self.tools[tool_name](**args)
            return result
        except Exception as e:
            return f"错误：工具 '{tool_name}' 执行失败: {str(e)}"

# =========================
# Skill 注册中心
# =========================
class SkillRegistry:
    def __init__(self):
        self.skills: Dict[str, Any] = {}
        self.skill_schemas: Dict[str, Dict] = {}

    def register(self, name: str, skill: Any, description: str = "", params_schema: Optional[Dict] = None):
        self.skills[name] = skill
        self.skill_schemas[name] = {"type": "skill", "name": name, "description": description, "parameters": params_schema or {"type": "object", "properties": {}}}

    def get_available_skills(self) -> List[Dict]:
        return list(self.skill_schemas.values())

    async def execute(self, skill_name: str, args: Dict[str, Any]) -> Any:
        if skill_name not in self.skills:
            return f"错误：Skill '{skill_name}' 不存在"
        try:
            result = await self.skills[skill_name].run(**args)
            return result
        except Exception as e:
            return f"错误：Skill '{skill_name}' 执行失败: {str(e)}"

# =========================
# 动态加载 Skills
# =========================
def load_skills_from_directory(skills_dir: str = "skills") -> Dict[str, Any]:
    """从指定目录加载所有 Skills"""
    loaded_skills = {}
    if not os.path.exists(skills_dir):
        return loaded_skills
    # 支持打包部署：优先使用 __file__，否则使用当前工作目录
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    skills_path = os.path.join(script_dir, skills_dir)
    if not os.path.exists(skills_path):
        return loaded_skills
    for filename in os.listdir(skills_path):
        if filename.endswith('.md') and filename not in ('README.md', 'config.yaml'):
            skill_name = filename[:-3]
            file_path = os.path.join(skills_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            meta = _parse_skill_frontmatter(content)
            loaded_skills[skill_name] = DynamicSkill(name=skill_name, file_path=file_path, content=content, meta=meta)
    return loaded_skills

def _parse_skill_frontmatter(content: str) -> Dict[str, Any]:
    """使用 PyYAML 解析 frontmatter，支持复杂 YAML 结构"""
    meta = {}
    if not content.startswith('---'):
        return meta
    try:
        parts = content.split('---', 2)
        if len(parts) >= 3:
            yaml_content = parts[1].strip()
            # 使用 PyYAML 解析
            if HAS_YAML:
                meta = yaml.safe_load(yaml_content) or {}
            else:
                # fallback 到简单解析
                for line in yaml_content.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        meta[key.strip()] = value.strip().strip('"').strip("'")
    except Exception as e:
        logger.warning(f"Failed to parse frontmatter: {e}")
    return meta

class DynamicSkill:
    def __init__(self, name: str, file_path: str, content: str, meta: Dict[str, Any]):
        self.name = name
        self.file_path = file_path
        self.content = content
        self.meta = meta
        self.description = meta.get('description', '')
        # 提取可执行代码（如果存在）
        self.code = self._extract_code(content)

    def _extract_code(self, content: str) -> Optional[str]:
        """提取 frontmatter 之后的代码块"""
        if not content.startswith('---'):
            return None
        parts = content.split('---', 2)
        if len(parts) >= 3:
            body = parts[2].strip()
            # 移除 Markdown 格式标记
            if body.startswith('```'):
                lines = body.split('\n')
                # 找到结尾的 ```
                for i, line in enumerate(lines[1:], 1):
                    if line.strip() == '```':
                        return '\n'.join(lines[1:i])
            return body
        return None

    async def run(self, **kwargs: Any) -> Dict[str, Any]:
        """执行 Skill，支持执行代码或返回元数据"""
        result = {
            "skill_name": self.name,
            "description": self.description,
            "meta": self.meta,
            "content_length": len(self.content),
            "success": True
        }
        # 如果有可执行代码，尝试执行
        if self.code:
            try:
                # 创建安全的执行环境
                exec_globals = {
                    "__builtins__": {
                        "print": print,
                        "len": len,
                        "str": str,
                        "int": int,
                        "float": float,
                        "list": list,
                        "dict": dict,
                        "range": range
                    },
                    "kwargs": kwargs,
                    "skill_meta": self.meta
                }
                exec_locals = {}
                exec(self.code, exec_globals, exec_locals)
                # 如果代码返回了值，使用它
                if "result" in exec_locals:
                    result["exec_result"] = exec_locals["result"]
                elif "output" in exec_locals:
                    result["exec_result"] = exec_locals["output"]
                else:
                    result["exec_result"] = "Code executed successfully"
            except Exception as e:
                result["exec_result"] = f"Code execution error: {str(e)}"
                result["success"] = False
        return result

# =========================
# Tavily 搜索工具
# =========================
class TavilySearchTool:
    def __init__(self, api_key: str = TAVILY_API_KEY):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)

    async def search(self, query: str, max_results: int = 5, search_depth: str = "advanced", include_answer: bool = False, include_raw_content: bool = False, include_images: bool = False) -> Dict[str, Any]:
        if not self.api_key:
            return self._mock_search(query, max_results)
        try:
            payload = {"api_key": self.api_key, "query": query, "max_results": max_results, "search_depth": search_depth, "include_answer": include_answer, "include_raw_content": include_raw_content, "include_images": include_images}
            response = await self.client.post("https://api.tavily.com/search", json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            result = response.json()
            return result
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return self._mock_search(query, max_results)

    def format_results(self, results: Dict[str, Any]) -> str:
        try:
            if not results or "results" not in results:
                return "未找到相关搜索结果"
            formatted = []
            answer = results.get("answer")
            if answer:
                formatted.append(f"AI 摘要:\n{answer}\n")
            search_results = results.get("results", [])
            if not isinstance(search_results, list):
                search_results = []
            for i, result in enumerate(search_results[:10], 1):
                if not isinstance(result, dict):
                    continue
                title = result.get("title", "无标题")
                url = result.get("url", "")
                content = result.get("content", "")
                score = result.get("score", 0)
                formatted.append(f"{i}. **{title}**")
                formatted.append(f"   URL: {url}")
                formatted.append(f"   相关性：{score:.2f}")
                if content:
                    formatted.append(f"   摘要：{content[:200]}...")
                formatted.append("")
            return "\n".join(formatted)
        except Exception as e:
            logger.error(f"format_results error: {e}")
            return "搜索结果格式化失败"

    def _mock_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        return {"query": query, "results": [{"title": f"关于 '{query}' 的分析", "url": "https://example.com", "content": "模拟结果...", "score": 0.95}], "answer": None, "images": []}

    async def close(self):
        if hasattr(self, 'client') and self.client:
            await self.client.aclose()
            self.client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# =========================
# v3 严格事件驱动框架（已修复）
# =========================
"""
核心原则（已修复）：
1. ExecutionEngine 是唯一调度核心
2. Worker 无脑执行（不决定下一步，不修改 DAG）
3. 状态驱动（只有 Engine 决定谁 READY）
4. Single Source of Truth：Engine 拥有所有状态修改权
"""

class StepState(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class StepV3:
    step_id: str
    step_type: str
    depends_on: List[str] = field(default_factory=list)
    max_retries: int = 3
    timeout: float = 60.0
    parallel: bool = True
    input_data: Dict[str, Any] = field(default_factory=dict)
    output: Any = None
    error: Optional[str] = None
    status: StepState = StepState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.step_id:
            self.step_id = f"step-{uuid.uuid4().hex[:8]}"

class PlanV3:
    def __init__(self, plan_id: str, task: str):
        self.plan_id = plan_id
        self.task = task
        self.steps: Dict[str, StepV3] = {}
        self.status: str = "pending"
        self.created_at = time.time()

    def add_step(self, step: StepV3) -> bool:
        if step.step_id in self.steps:
            return False
        for dep_id in step.depends_on:
            if dep_id not in self.steps:
                logger.warning(f"Step {step.step_id} depends on non-existent {dep_id}")
                return False
        self.steps[step.step_id] = step
        return True

    def get_ready_steps(self) -> List[StepV3]:
        """只有 Engine 可以调用"""
        ready = []
        for step in self.steps.values():
            if step.status != StepState.PENDING:
                continue
            deps_met = all(self.steps[d].status == StepState.COMPLETED for d in step.depends_on)
            if deps_met:
                ready.append(step)
        return ready

    def is_complete(self) -> bool:
        return all(s.status == StepState.COMPLETED for s in self.steps.values())

    def has_failed(self) -> bool:
        return any(s.status == StepState.FAILED for s in self.steps.values())

    def get_step_count_by_state(self, state: StepState) -> int:
        return len([s for s in self.steps.values() if s.status == state])

class StateV3:
    def __init__(self):
        self.steps: Dict[str, Dict[str, Any]] = {}
        self.artifacts: Dict[str, Any] = {}
        self.errors: List[Dict[str, Any]] = []

    def update_step(self, step_id: str, **kwargs):
        if step_id not in self.steps:
            self.steps[step_id] = {"step_id": step_id, "created_at": time.time()}
        self.steps[step_id].update(kwargs)

    def add_artifact(self, key: str, value: Any):
        self.artifacts[key] = value

# =========================
# Worker v3（已修复：完全无脑执行 + 合并重复逻辑）
# =========================
class WorkerV3:
    def __init__(self, worker_id: str, tool_registry: ToolRegistry, skill_registry: SkillRegistry):
        self.worker_id = worker_id
        self.tool_registry = tool_registry
        self.skill_registry = skill_registry
        self.llm: Optional[AsyncOpenAI] = None
        self.search_tool: Optional[TavilySearchTool] = None
        self.bus: Optional[EventBus] = None
        self.plan: Optional[PlanV3] = None
        self.engine: Optional[ExecutionEngineV3] = None
        self._running = False

    def set_llm(self, llm: AsyncOpenAI, search_tool: TavilySearchTool):
        self.llm = llm
        self.search_tool = search_tool

    def set_event_bus(self, bus: EventBus):
        self.bus = bus

    async def on_step_ready(self, event: Event):
        """
        Worker 的事件处理入口 - 只能订阅 STEP_READY 事件
        1. 调用 Engine 的原子接口抢占任务
        2. 执行步骤（带重试）
        3. 发布 STEP_COMPLETED 或 STEP_FAILED（不修改任何状态）
        """
        if event.event_type != EventType.STEP_READY:
            return
        if not self.bus or not self.plan or not self.engine:
            return
        step_id = event.step_id
        if not step_id:
            return
        # 原子CLAIM步骤 - 防止多Worker重复执行
        if not await self.engine.claim_step(step_id):
            return  # 没抢到，退出
        try:
            step = self.plan.steps.get(step_id)
            if not step:
                return
            # 带重试机制执行步骤
            success, result = await self._execute_with_retry(step)
            # 修复：只发布事件，不修改step.status
            # 状态更新完全由 Engine 在 process_completed 中完成
            if success:
                # 打印步骤执行结果
                logger.info(f"[{self.worker_id}] Step {step.step_id[:8]} ({step.step_type}) completed successfully")
                logger.info(f"[{self.worker_id}] Result: {str(result)[:500]}...")
                # 存储到 state.artifacts（Engine 会同步这个状态）
                if self.engine.state:
                    self.engine.state.artifacts[step_id] = result
                evt = Event(
                    event_type=EventType.STEP_COMPLETED,
                    step_id=step_id,
                    payload={"output": result}
                )
                await self.bus.publish(evt)
            else:
                # 打印步骤失败结果
                logger.error(f"[{self.worker_id}] Step {step.step_id[:8]} ({step.step_type}) failed: {str(result)[:200]}")
                evt = Event(
                    event_type=EventType.STEP_FAILED,
                    step_id=step_id,
                    payload={"error": str(result) if result else "Unknown error"}
                )
                await self.bus.publish(evt)
        finally:
            # 释放CLAIM状态
            await self.engine.release_claim(step_id)

    async def _get_dependency_artifacts(self, step: StepV3) -> Dict[str, Any]:
        """
        获取依赖步骤的输出结果。
        优先从 step.input_data 获取，其次从 Engine.state.artifacts 获取。
        """
        artifacts = {}
        for dep_id in step.depends_on:
            # 优先从 step.input_data 获取（由 Engine 在 publish_ready_steps 时填充）
            artifact = step.input_data.get(f"_dep_{dep_id}")
            if artifact is None and self.engine and self.engine.state:
                # fallback 到 Engine 的 state.artifacts
                artifact = self.engine.state.artifacts.get(dep_id)
            if artifact:
                artifacts[f"_dep_{dep_id}"] = artifact
        return artifacts

    async def execute(self, step: StepV3) -> Tuple[bool, Any]:
        """统一的执行入口 - 合并了 _execute_with_retry 的逻辑"""
        logger.info(f"[{self.worker_id}] Executing step {step.step_id[:8]} ({step.step_type})")
        try:
            if step.step_type == "tool":
                result = await self.tool_registry.execute(step.tool_name or "tavily_search", step.tool_args)
            elif step.step_type == "search":
                result = await self._execute_search(step)
            elif step.step_type == "analyze":
                result = await self._execute_analyze(step)
            elif step.step_type == "answer":
                result = await self._execute_answer(step)
            else:
                result = await self._execute_llm(step)
            return (True, result)
        except Exception as e:
            return (False, str(e))

    async def _execute_search(self, step: StepV3) -> str:
        query = step.input_data.get("query", "")
        # 使用已配置的 search_tool 实例（单例，避免重复创建 httpx.AsyncClient）
        if self.search_tool:
            try:
                result = await self.search_tool.search(query=query, max_results=5)
                return self.search_tool.format_results(result)
            except Exception as e:
                logger.exception(f"Search failed for step {step.step_id[:8]}")
                return "搜索执行失败"
        else:
            # 回退方案 - 仅在 search_tool 未初始化时创建临时实例
            temp_search = TavilySearchTool()
            try:
                result = await temp_search.search(query=query, max_results=5)
                return temp_search.format_results(result)
            finally:
                await temp_search.close()

    async def _execute_analyze(self, step: StepV3) -> str:
        """执行分析步骤"""
        context_parts = []
        artifacts = await self._get_dependency_artifacts(step)
        for dep_id in step.depends_on:
            artifact = artifacts.get(f"_dep_{dep_id}")
            if artifact:
                context_parts.append(f"[{dep_id}]: {str(artifact)[:500]}")
        context = "\n\n".join(context_parts)
        prompt = f"分析以下内容：\n{context}\n\n请给出你的分析和见解。"
        if not self.llm:
            return "LLM 未初始化"
        response = await self.llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": "你是一个分析专家。"}, {"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=config_manager.LLM_MAX_TOKENS
        )
        return response.choices[0].message.content

    async def _execute_answer(self, step: StepV3) -> str:
        """执行答案生成步骤"""
        context_parts = []
        artifacts = await self._get_dependency_artifacts(step)
        for dep_id in step.depends_on:
            artifact = artifacts.get(f"_dep_{dep_id}")
            if artifact:
                context_parts.append(f"[{dep_id}]: {str(artifact)[:500]}")
        context = "\n\n".join(context_parts)

        # 获取原始问题（优先从 step.input_data 获取，其次从 plan.task）
        original_task = step.input_data.get("task", step.input_data.get("prompt", ""))
        if not original_task and self.plan:
            original_task = self.plan.task

        # 如果没有有效上下文，根据原始问题决定返回内容
        if not context.strip():
            if original_task:
                # 有原始问题但没有上下文，直接让 LLM 回答简单问题
                prompt = f"用户问题：{original_task}\n\n请直接回答这个问题。"
            else:
                # 完全没有信息，返回提示
                return "请提供具体的上下文或问题内容，我将根据您提供的信息给出最终答案。"
        else:
            # 有上下文时，组合问题和上下文
            prompt = f"""用户问题：{original_task}

相关上下文和分析结果：
{context}

请基于以上信息给出最终答案。"""

        if not self.llm:
            return "LLM 未初始化"
        response = await self.llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": "你是一个回答专家。"}, {"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=config_manager.LLM_MAX_TOKENS
        )
        return response.choices[0].message.content

    async def _execute_llm(self, step: StepV3) -> str:
        """执行通用 LLM 步骤"""
        prompt = step.input_data.get("prompt", step.input_data.get("task", ""))
        if not self.llm:
            return "LLM 未初始化"
        response = await self.llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": "你是一个智能助手。"}, {"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=config_manager.LLM_MAX_TOKENS
        )
        return response.choices[0].message.content

    async def _execute_with_retry(self, step: StepV3) -> Tuple[bool, Any]:
        """
        带重试机制的执行方法 - 现在统一调用 execute
        """
        last_error = None
        for attempt in range(step.max_retries):
            if attempt > 0:
                logger.info(f"[{self.worker_id}] Retry attempt {attempt + 1}/{step.max_retries} for step {step.step_id[:8]}")
                await asyncio.sleep(0.5 * (attempt + 1))  # 指数退避
            try:
                # 统一调用 execute，避免重复逻辑
                result = await self.execute(step)
                if not result[0]:
                    raise Exception(result[1])
                return (True, result[1])
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[{self.worker_id}] Step {step.step_id[:8]} failed on attempt {attempt + 1}: {e}")
                logger.exception(f"Exception details")  # 修复：添加 logger.exception
        return (False, last_error or "Unknown error")

# =========================
# ExecutionEngine v3（已修复：唯一调度核心，Single Source of Truth）
# =========================
class ExecutionEngineV3:
    def __init__(self, tool_registry: ToolRegistry, skill_registry: SkillRegistry):
        self.tool_registry = tool_registry
        self.skill_registry = skill_registry
        self.plan: Optional[PlanV3] = None
        self.state = StateV3()
        self.bus: Optional[EventBus] = None
        self.llm: Optional[AsyncOpenAI] = None
        self.search_tool: Optional[TavilySearchTool] = None
        self._running = False
        self._claim_lock = asyncio.Lock()  # CLAIM 操作的原子锁
        self._claiming_steps: Set[str] = set()  # 正在CLAIM的步骤ID
        # 修复：记录已发布 READY 的步骤，防止重复发布
        self._published_ready_steps: Set[str] = set()
        # 修复：订阅管理器，集中管理事件订阅
        self._subscribed_events: Set[EventType] = set()

    def set_event_bus(self, bus: EventBus):
        self.bus = bus

    def set_llm(self, llm: AsyncOpenAI, search_tool: TavilySearchTool):
        self.llm = llm
        self.search_tool = search_tool

    def set_plan(self, plan: PlanV3):
        self.plan = plan
        # 初始化 published_ready_steps 集合
        self._published_ready_steps.clear()

    async def start(self):
        """启动调度循环 - 唯一调度器"""
        if not self.bus:
            raise RuntimeError("Event bus not set")
        # 注意：事件订阅在 run() 中统一管理，避免重复订阅
        # self.bus.subscribe(self.process_completed)  # 已在 run() 中订阅
        # self.bus.subscribe(self.process_failed)     # 已在 run() 中订阅
        self._running = True
        logger.info("[Engine] Started, publishing initial READY steps...")
        # 发布初始 READY 步骤（设置状态为READY并发布）
        await self._publish_ready_steps()

    async def claim_step(self, step_id: str) -> bool:
        """
        原子操作：CLAIM 一个 READY 步骤
        防止多个 Worker 同时执行同一个步骤
        """
        async with self._claim_lock:
            if step_id in self._claiming_steps:
                return False  # 已经在CLAIM中
            step = self.plan.steps.get(step_id) if self.plan else None
            if step and step.status == StepState.READY:
                step.status = StepState.RUNNING
                step.started_at = time.time()
                self._claiming_steps.add(step_id)
                # 修复：从已发布集合中移除，因为现在正在运行
                self._published_ready_steps.discard(step_id)
                return True
            return False

    async def release_claim(self, step_id: str):
        """释放CLAIM状态"""
        async with self._claim_lock:
            self._claiming_steps.discard(step_id)

    async def _publish_ready_steps(self):
        """
        发布 READY 步骤 - Engine 独有权限
        修复：使用 _published_ready_steps 集合防止重复发布
        """
        if not self.bus or not self.plan:
            return

        ready_steps = self.plan.get_ready_steps()
        for step in ready_steps:
            # 修复：只发布尚未发布的步骤（增量发布）
            if step.step_id not in self._published_ready_steps:
                # 设置为READY状态
                step.status = StepState.READY
                # 构建 input_data，将依赖步骤的输出填充进去
                input_data = dict(step.input_data)  # 复制一份
                for dep_id in step.depends_on:
                    artifact = self.state.artifacts.get(dep_id)
                    if artifact:
                        input_data[f"_dep_{dep_id}"] = artifact
                evt = Event(
                    event_type=EventType.STEP_READY,
                    step_id=step.step_id,
                    payload={"step_type": step.step_type, "step_data": {"input_data": input_data, "tool_name": step.tool_name, "tool_args": step.tool_args}}
                )
                logger.info(f"[Engine] Publishing STEP_READY for {step.step_id[:8]}")
                # 修复：添加到已发布集合
                self._published_ready_steps.add(step.step_id)
                # 修复：直接 await publish，而不是 create_task
                # 这样可以捕获异常并保证顺序
                try:
                    await self.bus.publish(evt)
                except Exception as e:
                    logger.exception(f"[Engine] Failed to publish STEP_READY for {step.step_id[:8]}")

    async def process_completed(self, event: Event):
        """
        处理 STEP_COMPLETED 事件 - 唯一的调度入口
        1. 更新 State（Single Source of Truth）
        2. 找 READY 步骤
        3. 发布 STEP_READY
        """
        if event.event_type != EventType.STEP_COMPLETED or not event.step_id:
            return
        logger.info(f"[Engine] STEP_COMPLETED for {event.step_id[:8]}")
        step = self.plan.steps.get(event.step_id) if self.plan else None
        if step:
            # 修复：完全由 Engine 更新状态，Worker 不得修改
            step.status = StepState.COMPLETED
            step.output = event.payload.get("output")
            step.completed_at = time.time()
            self.state.update_step(event.step_id, status="completed", output=event.payload.get("output"))
        # 修复：发布 READY 步骤
        await self._publish_ready_steps()
        # 修复：如果所有步骤都已完成，发布 TASK_COMPLETED 事件
        if self.plan and self.plan.is_complete():
            logger.info("[Engine] All steps completed!")
            await self.bus.publish(Event(event_type=EventType.TASK_COMPLETED, task_id=self.plan.plan_id, payload={"plan_id": self.plan.plan_id}))

    async def process_failed(self, event: Event):
        """
        处理 STEP_FAILED 事件
        """
        if event.event_type != EventType.STEP_FAILED or not event.step_id:
            return
        logger.warning(f"[Engine] STEP_FAILED for {event.step_id[:8]}")
        step = self.plan.steps.get(event.step_id) if self.plan else None
        if step:
            # 修复：完全由 Engine 更新状态
            step.status = StepState.FAILED
            step.error = event.payload.get("error")
            step.completed_at = time.time()
            self.state.update_step(event.step_id, status="failed", error=event.payload.get("error"))
            self.state.errors.append({
                "step_id": event.step_id,
                "error": event.payload.get("error"),
                "timestamp": time.time()
            })
        # 失败后也触发调度（可能有其他并行步骤可以继续）
        await self._publish_ready_steps()
        # 如果所有步骤都已完成或失败，发布 TASK_COMPLETED/TASK_FAILED
        if self.plan:
            if self.plan.is_complete():
                logger.info("[Engine] All steps completed!")
                await self.bus.publish(Event(event_type=EventType.TASK_COMPLETED, task_id=self.plan.plan_id, payload={"plan_id": self.plan.plan_id}))
            elif self.plan.has_failed():
                # 检查是否所有可能的步骤都失败了（没有可执行的 READY 步骤）
                ready_steps = self.plan.get_ready_steps()
                if not ready_steps and self.plan.get_step_count_by_state(StepState.FAILED) > 0:
                    logger.warning("[Engine] Task failed - some steps failed and no steps are ready")
                    await self.bus.publish(Event(event_type=EventType.TASK_FAILED, task_id=self.plan.plan_id, payload={"plan_id": self.plan.plan_id}))

# =========================
# Planner v3（只生成 DAG）
# =========================
class PlannerV3:
    def __init__(self, llm: AsyncOpenAI):
        self.llm = llm

    async def plan(self, task: str, tools: List[Dict]) -> PlanV3:
        logger.info(f"[PlannerV3] Generating DAG for task: {task[:50]}...")
        tools_desc = "\n".join([f"- {t.get('name')}: {t.get('description')}" for t in tools])
        prompt = f"""任务：{task}

可用工具：
{tools_desc}

请生成一个执行计划（DAG），要求：
1. 每个步骤必须有唯一的 id
2. 正确设置依赖关系（depends_on）
3. 只返回步骤列表

输出格式（JSON）：
{{
  "steps": [
    {{
      "id": "step_id",
      "type": "tool|llm|search|analyze|answer",
      "depends_on": [],
      "input_data": {{}},
      "tool_name": "tool_name",
      "tool_args": {{}}
    }}
  ]
}}"""
        # response_format={"type": "json_object"} 是 OpenAI 的参数
        # 使用时需要确保模型支持，否则会忽略
        try:
            response = await self.llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": "你是一个任务规划器。请将任务分解为可执行的步骤序列（DAG）。"}, {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=6000
            )
        except Exception as e:
            logger.error(f"[PlannerV3] LLM call failed: {e}")
            response = None

        if not response or not response.choices:
            logger.error("[PlannerV3] No response from LLM")
            return PlanV3(plan_id=f"plan-{uuid.uuid4().hex[:8]}", task=task)

        plan_json = self._parse_response(response.choices[0].message.content)
        plan_id = f"plan-{uuid.uuid4().hex[:8]}"
        plan = PlanV3(plan_id=plan_id, task=task)

        # JSON schema 校验
        steps_data = plan_json.get("steps", [])
        for step_data in steps_data:
            # 校验必需字段
            step_id = step_data.get("id")
            step_type = step_data.get("type")

            if not step_id:
                logger.warning(f"[PlannerV3] Skipping step without id: {step_data}")
                continue
            if step_type not in ["tool", "llm", "search", "analyze", "answer"]:
                logger.warning(f"[PlannerV3] Invalid step type '{step_type}' for step {step_id}, using 'llm'")
                step_type = "llm"

            step = StepV3(
                step_id=step_id,
                step_type=step_type,
                depends_on=step_data.get("depends_on", []),
                input_data=step_data.get("input_data", {}),
                tool_name=step_data.get("tool_name"),
                tool_args=step_data.get("tool_args", {}),
                max_retries=step_data.get("max_retries", 3)
            )
            if not plan.add_step(step):
                logger.warning(f"[PlannerV3] Failed to add step {step_id}")

        # 循环依赖检测
        cycle = self._detect_cycle(plan)
        if cycle:
            logger.error(f"[PlannerV3] Cycle detected in plan: {' -> '.join(cycle)}")
            # 移除循环依赖：找到循环中的最后一个步骤，移除其依赖
            cycle_step_id = cycle[-1]
            cycle_step = plan.steps.get(cycle_step_id)
            if cycle_step:
                # 找到循环中的前一个步骤，断开依赖
                cycle_prev = cycle[-2] if len(cycle) > 1 else None
                if cycle_prev and cycle_prev in cycle_step.depends_on:
                    cycle_step.depends_on.remove(cycle_prev)
                    logger.info(f"[PlannerV3] Removed cycle dependency: {cycle_step_id} -> {cycle_prev}")

        plan.status = "planned"
        logger.info(f"[PlannerV3] Generated plan with {len(plan.steps)} steps")
        return plan

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析 LLM 响应为 JSON"""
        try:
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            logger.error(f"[PlannerV3] Failed to parse: {response[:200]}")
            return {"steps": []}

    def _detect_cycle(self, plan: PlanV3) -> Optional[List[str]]:
        """检测 DAG 中的循环依赖（使用 DFS）"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {step_id: WHITE for step_id in plan.steps}
        parent = {}

        def dfs(node: str) -> Optional[List[str]]:
            color[node] = GRAY
            step = plan.steps.get(node)
            if step:
                for dep in step.depends_on:
                    if dep not in color:
                        continue  # 依赖不存在，跳过
                    if color[dep] == GRAY:
                        # 找到循环：从 dep 到 node 的路径
                        cycle = [node]
                        current = node
                        while current != dep:
                            current = parent.get(current)
                            if current is None:
                                break
                            cycle.append(current)
                        cycle.append(dep)
                        return cycle
                    if color[dep] == WHITE:
                        parent[dep] = node
                        result = dfs(dep)
                        if result:
                            return result
            color[node] = BLACK
            return None

        for step_id in plan.steps:
            if color[step_id] == WHITE:
                result = dfs(step_id)
                if result:
                    return result
        return None

# =========================
# ProductionAgentOS v3（统一入口）
# =========================
class ProductionAgentOS_v3:
    def __init__(self, worker_count: int = 2, skills_dir: str = "skills"):
        self.bus = EventBus()
        self.tool_registry = ToolRegistry()
        self.skill_registry = SkillRegistry()
        self.engine: Optional[ExecutionEngineV3] = None
        self.planner: Optional[PlannerV3] = None
        self.llm: Optional[AsyncOpenAI] = None
        self.search_tool: Optional[TavilySearchTool] = None
        self.workers: List[WorkerV3] = []
        self.worker_count = worker_count
        self._completion_event: Optional[asyncio.Event] = None
        self._register_skills()

    def _register_skills(self):
        try:
            skills = load_skills_from_directory("skills")
            for skill_name, skill_obj in skills.items():
                self.skill_registry.register(name=skill_name, skill=skill_obj, description=skill_obj.description)
        except Exception as e:
            logger.error(f"[AgentOS_v3] Failed to load skills: {e}")

    def register_tool(self, name: str, func: Callable, description: str, params_schema: Optional[Dict] = None):
        self.tool_registry.register(name, func, description, params_schema)

    async def _wait_for_completion(self, plan: PlanV3, timeout: float) -> Dict[str, Any]:
        """使用 asyncio.Event 代替轮询，避免浪费 CPU"""
        if not self._completion_event:
            raise RuntimeError("Completion event not initialized")
        try:
            await asyncio.wait_for(self._completion_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        # 返回结果
        if plan.is_complete():
            return {"status": "completed", "plan_id": plan.plan_id, "total_steps": len(plan.steps), "completed_steps": plan.get_step_count_by_state(StepState.COMPLETED)}
        elif plan.has_failed():
            return {"status": "failed", "plan_id": plan.plan_id, "total_steps": len(plan.steps), "completed_steps": plan.get_step_count_by_state(StepState.COMPLETED), "failed_steps": plan.get_step_count_by_state(StepState.FAILED)}
        else:
            return {"status": "timeout", "plan_id": plan.plan_id, "total_steps": len(plan.steps), "completed_steps": plan.get_step_count_by_state(StepState.COMPLETED), "failed_steps": plan.get_step_count_by_state(StepState.FAILED)}

    async def run(self, task: str, timeout: float = 300.0) -> Dict[str, Any]:
        logger.info("=" * 70)
        logger.info("任务启动 (v3 - 严格事件驱动)")
        logger.info("=" * 70)
        self.llm = AsyncOpenAI(base_url=f"{VLLM_URL}/v1", api_key="sk-ignore", timeout=config_manager.LLM_CALL_TIMEOUT)
        self.search_tool = TavilySearchTool()
        self.planner = PlannerV3(self.llm)
        self.engine = ExecutionEngineV3(self.tool_registry, self.skill_registry)
        self.engine.set_event_bus(self.bus)
        self.engine.set_llm(self.llm, self.search_tool)
        # 创建 Completion Event 用于通知完成
        self._completion_event = asyncio.Event()
        # 订阅 TASK_COMPLETED 和 TASK_FAILED 事件来触发 completion event
        async def on_task_completed(event: Event):
            if event.event_type != EventType.TASK_COMPLETED:
                return
            logger.info("[AgentOS] TASK_COMPLETED event received")
            if self._completion_event and not self._completion_event.is_set():
                self._completion_event.set()
        async def on_task_failed(event: Event):
            if event.event_type != EventType.TASK_FAILED:
                return
            logger.info("[AgentOS] TASK_FAILED event received")
            if self._completion_event and not self._completion_event.is_set():
                self._completion_event.set()
        self.bus.subscribe(on_task_completed)
        self.bus.subscribe(on_task_failed)
        # 创建 Workers
        self.workers = [WorkerV3(f"W-{i}", self.tool_registry, self.skill_registry) for i in range(self.worker_count)]
        for w in self.workers:
            w.set_event_bus(self.bus)
            w.set_llm(self.llm, self.search_tool)
            w.engine = self.engine
        # Engine 订阅 STEP_COMPLETED 和 STEP_FAILED 事件
        self.bus.subscribe(self.engine.process_completed)
        self.bus.subscribe(self.engine.process_failed)
        # Workers 订阅 STEP_READY 事件
        for w in self.workers:
            self.bus.subscribe(w.on_step_ready)
        tools = self.tool_registry.get_available_tools()
        plan = await self.planner.plan(task, tools)
        if not plan or not plan.steps:
            logger.error("[AgentOS] Plan is empty, cannot proceed")
            return {"status": "error", "error": "Plan generation failed - no steps produced"}
        self.engine.set_plan(plan)
        # 更新 workers 的 plan 引用（在 set_plan 之后）
        for w in self.workers:
            w.plan = self.engine.plan
        await self.engine.start()
        try:
            result = await self._wait_for_completion(plan, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            return {"status": "timeout", "plan_id": plan.plan_id, "total_steps": len(plan.steps), "completed_steps": plan.get_step_count_by_state(StepState.COMPLETED), "failed_steps": plan.get_step_count_by_state(StepState.FAILED)}
        finally:
            # 统一关闭资源
            logger.info("[AgentOS] Cleaning up resources...")
            if self._completion_event:
                self._completion_event.clear()
                self._completion_event = None
            if self.search_tool:
                try:
                    await self.search_tool.close()
                except Exception as e:
                    logger.exception("[AgentOS] Error closing search_tool")
            if self.llm:
                try:
                    await self.llm.close()
                except Exception as e:
                    logger.exception("[AgentOS] Error closing LLM")
            await self.bus.shutdown()

# =========================
# 运行入口
# =========================
async def main():
    agent = ProductionAgentOS_v3(worker_count=2, skills_dir="skills")
    task = "hello"
    agent.register_tool(name="tavily_search", func=lambda q: None, description="使用 Tavily 进行网络搜索")
    try:
        result = await agent.run(task, timeout=300)
        logger.info("=" * 70)
        logger.info("最终结果 (v3)")
        logger.info("=" * 70)
        logger.info(f"Status: {result.get('status')}")
        logger.info(f"Plan ID: {result.get('plan_id')}")
        logger.info(f"Total Steps: {result.get('total_steps')}")
        logger.info(f"Completed: {result.get('completed_steps')}")
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"错误：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
