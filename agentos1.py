import asyncio
import uuid
import time
import logging
import json
import os
from enum import Enum
from typing import Dict, Any, List, Set, Optional, Callable, Tuple
from dataclasses import dataclass, field
from openai import AsyncOpenAI
from collections import defaultdict

# =========================
# 配置管理
# =========================
class ConfigManager:
    def __init__(self):
        self.MAX_STEP_RETRIES = int(os.getenv("MAX_STEP_RETRIES", "3"))
        self.GLOBAL_TIMEOUT = int(os.getenv("GLOBAL_TIMEOUT", "300"))
        self.STEP_EXECUTION_TIMEOUT = int(os.getenv("STEP_EXECUTION_TIMEOUT", "120"))
        self.EVENT_BUS_MAX_QUEUE_SIZE = int(os.getenv("EVENT_BUS_MAX_QUEUE_SIZE", "1000"))
        self.VLLM_URL = os.getenv("VLLM_URL", "http://192.168.1.210:19000")
        self.MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3Qoder")
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
        self.WORKER_COUNT = int(os.getenv("WORKER_COUNT", "3"))
        self.SKILLS_DIR = os.getenv("SKILLS_DIR", "skills")
        self.PERSIST_DIR = os.getenv("PERSIST_DIR", "storage")

config_manager = ConfigManager()
VLLM_URL = config_manager.VLLM_URL
MODEL_NAME = config_manager.MODEL_NAME

logging.basicConfig(
    level=getattr(logging, config_manager.LOG_LEVEL, logging.INFO),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("AgentOS_v3")

# =========================
# 状态持久化接口
# =========================
class PersistenceManager:
    """处理状态持久化的类"""
    def __init__(self, base_dir: str = config_manager.PERSIST_DIR):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save_state(self, plan_id: str, data: Dict[str, Any]):
        path = os.path.join(self.base_dir, f"{plan_id}.json")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"状态已保存：{path}")
        except Exception as e:
            logger.error(f"持久化失败：{e}")

# =========================
# 事件系统升级
# =========================
class EventType(str, Enum):
    TASK_SUBMITTED = "task_submitted"
    STEP_READY = "step_ready"
    STEP_CLAIMED = "step_claimed"
    STEP_CHUNK = "step_chunk"
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

class EventBus:
    def __init__(self):
        self._subscribers: List[Callable] = []
        self._queue = asyncio.Queue(maxsize=config_manager.EVENT_BUS_MAX_QUEUE_SIZE)
        self._stop_event = asyncio.Event()
        self._worker_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    def subscribe(self, handler: Callable) -> int:
        """订阅事件，返回订阅 ID 以便后续取消"""
        self._subscribers.append(handler)
        return len(self._subscribers) - 1

    def unsubscribe(self, subscription_id: int):
        """取消订阅"""
        if 0 <= subscription_id < len(self._subscribers):
            self._subscribers.pop(subscription_id)

    async def publish(self, event: Event):
        await self._queue.put(event)

    async def start(self):
        """启动事件总线（在事件循环中调用）"""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._process_queue())
            logger.debug("EventBus 已启动")

    async def _process_queue(self):
        while not self._stop_event.is_set():
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                if event is None:
                    break
                # 并发执行所有订阅者
                tasks = []
                for handler in self._subscribers:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(asyncio.create_task(handler(event)))
                    else:
                        # 同步回调封装为协程
                        tasks.append(asyncio.create_task(asyncio.to_thread(handler, event)))
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            logger.error(f"事件处理器 {i} 异常：{result}")
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"事件队列处理异常：{e}")

    async def shutdown(self, wait_queue: bool = True):
        """关闭事件总线

        Args:
            wait_queue: 如果为 True，等待队列中所有事件处理完成后再退出
        """
        if wait_queue:
            # 等待队列中所有事件处理完成
            try:
                await asyncio.wait_for(self._queue.join(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("队列清理超时，强制关闭")

        self._stop_event.set()
        await self._queue.put(None)
        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass
        logger.debug("EventBus 已关闭")

# =========================
# 核心模型 (V3 改良)
# =========================
class StepState(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class StepV3:
    step_id: str
    step_type: str
    depends_on: List[str] = field(default_factory=list)
    max_retries: int = config_manager.MAX_STEP_RETRIES
    retry_count: int = 0
    input_data: Dict[str, Any] = field(default_factory=dict)
    output: Any = None
    error: Optional[str] = None
    status: StepState = StepState.PENDING
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)

class StateV3:
    def __init__(self, plan_id: str, persistence: PersistenceManager):
        self.plan_id = plan_id
        self.persistence = persistence
        self.artifacts: Dict[str, Any] = {}
        self.step_logs: Dict[str, List[str]] = defaultdict(list)

    def add_artifact(self, step_id: str, value: Any):
        self.artifacts[step_id] = value
        self._trigger_persist()

    def add_chunk(self, step_id: str, chunk: str):
        self.step_logs[step_id].append(chunk)

    def _trigger_persist(self):
        data = {
            "plan_id": self.plan_id,
            "artifacts": self.artifacts,
            "timestamp": time.time()
        }
        self.persistence.save_state(self.plan_id, data)

# =========================
# Worker v3 (支持重试与流式)
# =========================
class WorkerV3:
    def __init__(self, worker_id: str, engine: 'ExecutionEngineV3'):
        self.worker_id = worker_id
        self.engine = engine
        self.llm: Optional[AsyncOpenAI] = None
        self.bus: Optional[EventBus] = None

    async def on_step_ready(self, event: Event):
        """处理 STEP_READY 事件的回调"""
        if event.event_type != EventType.STEP_READY:
            return

        step_id = event.step_id

        # 原子抢占
        claimed = await self.engine.claim_step(step_id)
        if not claimed:
            logger.debug(f"[{self.worker_id}] 步骤 {step_id} 已被其他 worker 抢占")
            return

        step = self.engine.plan.steps.get(step_id)
        if step is None:
            logger.error(f"[{self.worker_id}] 步骤 {step_id} 不存在")
            await self.engine.release_claim(step_id)
            return

        try:
            logger.info(f"[{self.worker_id}] 开始执行步骤：{step_id}")
            success, result = await self.execute(step)

            if success:
                await self.bus.publish(Event(
                    event_type=EventType.STEP_COMPLETED,
                    step_id=step_id,
                    payload={"output": result}
                ))
            else:
                await self.bus.publish(Event(
                    event_type=EventType.STEP_FAILED,
                    step_id=step_id,
                    payload={"error": str(result)}
                ))
        except Exception as e:
            logger.error(f"[{self.worker_id}] 步骤 {step_id} 执行异常：{e}")
            await self.bus.publish(Event(
                event_type=EventType.STEP_FAILED,
                step_id=step_id,
                payload={"error": str(e)}
            ))
        finally:
            await self.engine.release_claim(step_id)

    async def execute(self, step: StepV3) -> Tuple[bool, Any]:
        """执行步骤"""
        logger.info(f"[{self.worker_id}] 执行步骤：{step.step_id} (类型：{step.step_type})")
        try:
            if step.step_type in ["llm", "answer", "analyze"]:
                return await self._execute_streaming_llm(step)
            elif step.step_type == "tool":
                return await self._execute_tool(step)
            else:
                return (True, f"步骤 {step.step_id} 执行完成")
        except asyncio.TimeoutError:
            return (False, f"步骤执行超时")
        except Exception as e:
            return (False, str(e))

    async def _execute_streaming_llm(self, step: StepV3) -> Tuple[bool, str]:
        """执行流式 LLM 调用"""
        prompt = step.input_data.get("prompt", "请处理任务")
        full_response = []

        if self.llm is None:
            return (False, "LLM 客户端未初始化")

        try:
            response = await self.llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                timeout=config_manager.STEP_EXECUTION_TIMEOUT
            )

            async for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    full_response.append(content)
                    await self.bus.publish(Event(
                        event_type=EventType.STEP_CHUNK,
                        step_id=step.step_id,
                        payload={"chunk": content}
                    ))

            return (True, "".join(full_response))
        except Exception as e:
            logger.error(f"LLM 调用失败：{e}")
            return (False, str(e))

    async def _execute_tool(self, step: StepV3) -> Tuple[bool, Any]:
        """执行工具调用"""
        # 工具执行逻辑占位
        await asyncio.sleep(0.1)  # 模拟工具调用延迟
        return (True, f"工具 {step.tool_name} 执行结果")

# =========================
# Engine v3 (调度与重试核心)
# =========================
class ExecutionEngineV3:
    def __init__(self, tool_registry: Any, skill_registry: Any):
        self.plan: Optional[Any] = None
        self.state: Optional[StateV3] = None
        self.bus: Optional[EventBus] = None
        self._claim_lock = asyncio.Lock()
        self._claiming_steps: Set[str] = set()
        self._subscription_ids: List[int] = []

    async def claim_step(self, step_id: str) -> bool:
        """原子性地抢占步骤"""
        async with self._claim_lock:
            if step_id in self._claiming_steps:
                return False

            step = self.plan.steps.get(step_id)
            if step and step.status == StepState.READY:
                step.status = StepState.RUNNING
                self._claiming_steps.add(step_id)
                logger.debug(f"步骤 {step_id} 已被抢占")
                return True
            return False

    async def release_claim(self, step_id: str):
        """释放步骤的抢占标记"""
        async with self._claim_lock:
            self._claiming_steps.discard(step_id)
            logger.debug(f"步骤 {step_id} 的抢占标记已释放")

    async def start(self):
        """启动引擎"""
        if self.bus is None:
            raise RuntimeError("EventBus 未设置")

        # 订阅事件处理器
        self._subscription_ids.append(self.bus.subscribe(self.handle_completed))
        self._subscription_ids.append(self.bus.subscribe(self.handle_failed))
        self._subscription_ids.append(self.bus.subscribe(self.handle_chunk))

        # 发布就绪的步骤
        self._publish_ready_steps()

    async def handle_chunk(self, event: Event):
        """处理流式片段事件"""
        if event.event_type == EventType.STEP_CHUNK and event.step_id:
            self.state.add_chunk(event.step_id, event.payload.get("chunk", ""))

    async def handle_completed(self, event: Event):
        """处理步骤完成事件"""
        if event.event_type != EventType.STEP_COMPLETED:
            return

        step = self.plan.steps.get(event.step_id)
        if step is None:
            logger.error(f"步骤 {event.step_id} 不存在")
            return

        step.status = StepState.COMPLETED
        step.output = event.payload.get("output")
        self.state.add_artifact(event.step_id, event.payload.get("output"))

        logger.info(f"步骤 {event.step_id} 完成")

        # 发布新的就绪步骤
        self._publish_ready_steps()

        # 检查是否所有步骤都完成
        if all(s.status == StepState.COMPLETED for s in self.plan.steps.values()):
            logger.info("所有步骤已完成")
            await self.bus.publish(Event(event_type=EventType.TASK_COMPLETED))

    async def handle_failed(self, event: Event):
        """处理步骤失败事件 - 实现重试逻辑"""
        if event.event_type != EventType.STEP_FAILED:
            return

        step = self.plan.steps.get(event.step_id)
        if step is None:
            logger.error(f"步骤 {event.step_id} 不存在")
            return

        step.error = event.payload.get("error", "未知错误")

        # 重试逻辑
        if step.retry_count < step.max_retries:
            step.retry_count += 1
            step.status = StepState.PENDING
            logger.warning(f"步骤 {step.step_id} 失败 (错误：{step.error}), 正在进行第 {step.retry_count} 次重试...")

            # 重置为待处理状态后重新发布
            self._publish_ready_steps()
        else:
            step.status = StepState.FAILED
            logger.error(f"步骤 {step.step_id} 达到最大重试次数 ({step.max_retries}), 任务中止。")
            await self.bus.publish(Event(event_type=EventType.TASK_FAILED))

    def _publish_ready_steps(self):
        """发布就绪的步骤"""
        for step in self.plan.steps.values():
            if step.status == StepState.PENDING:
                # 检查所有依赖是否已完成
                deps_completed = all(
                    self.plan.steps[d].status == StepState.COMPLETED
                    for d in step.depends_on
                    if d in self.plan.steps
                )

                if deps_completed:
                    step.status = StepState.READY
                    # 注入依赖数据
                    for d in step.depends_on:
                        if d in self.state.artifacts:
                            step.input_data[f"_dep_{d}"] = self.state.artifacts[d]

                    # 异步发布事件
                    asyncio.create_task(self.bus.publish(Event(
                        event_type=EventType.STEP_READY,
                        step_id=step.step_id
                    )))

    def cleanup(self):
        """清理资源"""
        for sid in self._subscription_ids:
            if self.bus:
                self.bus.unsubscribe(sid)
        self._subscription_ids.clear()

# =========================
# 系统入口
# =========================
class ProductionAgentOS_v3:
    def __init__(self, worker_count: int = config_manager.WORKER_COUNT):
        self.bus = EventBus()
        self.persistence = PersistenceManager()
        self.engine = ExecutionEngineV3(None, None)
        self.worker_count = worker_count
        self._workers: List[WorkerV3] = []

    async def run(self, task: str, timeout: float = 300.0) -> Dict[str, Any]:
        """运行任务"""
        plan_id = f"plan-{uuid.uuid4().hex[:8]}"
        logger.info(f"开始执行任务：{task}")

        # 初始化状态
        self.engine.state = StateV3(plan_id, self.persistence)
        self.engine.bus = self.bus

        # 启动事件总线
        await self.bus.start()

        # 模拟 Planner 生成步骤
        from dataclasses import make_dataclass
        Plan = make_dataclass("Plan", [("steps", Dict), ("plan_id", str)])
        self.engine.plan = Plan(steps={
            "step-1": StepV3(step_id="step-1", step_type="llm", input_data={"prompt": f"分析任务：{task}"})
        }, plan_id=plan_id)

        # 启动 Workers
        llm_client = AsyncOpenAI(base_url=f"{VLLM_URL}/v1", api_key="sk-none")
        for i in range(self.worker_count):
            w = WorkerV3(f"Worker-{i}", self.engine)
            w.llm = llm_client
            w.bus = self.bus
            self.bus.subscribe(w.on_step_ready)
            self._workers.append(w)

        # 监控前端流式输出
        async def ui_logger(event: Event):
            if event.event_type == EventType.STEP_CHUNK:
                print(f"\r[UI 流式预览] {event.step_id}: {event.payload.get('chunk', '')}", end="", flush=True)
            elif event.event_type == EventType.STEP_COMPLETED:
                print(f"\n[UI] 步骤 {event.step_id} 完成。", flush=True)

        self.bus.subscribe(ui_logger)

        # 启动引擎
        await self.engine.start()

        # 等待完成
        done_event = asyncio.Event()
        async def waiter(event: Event):
            if event.event_type in [EventType.TASK_COMPLETED, EventType.TASK_FAILED]:
                done_event.set()
        self.bus.subscribe(waiter)

        try:
            await asyncio.wait_for(done_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error("任务执行超时")
            return {"error": "任务超时"}

        result = self.engine.state.artifacts.copy()
        logger.info(f"任务完成，结果：{result}")
        return result

    async def shutdown(self):
        """关闭系统"""
        self.engine.cleanup()
        await self.bus.shutdown()

async def main():
    os_sys = ProductionAgentOS_v3()
    try:
        result = await os_sys.run("编写一段关于未来 AI 的短文")
        print("\n最终结果:", json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        await os_sys.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
