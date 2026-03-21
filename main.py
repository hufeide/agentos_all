import asyncio
import uuid
import time
import logging
import json
import os
import re
from enum import Enum
from typing import Dict, Any, List, Set, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from openai import AsyncOpenAI

# =========================
# 配置与日志
# =========================
VLLM_URL = os.environ.get("VLLM_URL", "http://192.168.1.210:19000")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen3Qoder")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()  # 默认 INFO，便于观察关键步骤

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("ProductionAgentOS")

# =========================
# 核心事件模型
# =========================
class EventType(str, Enum):
    TASK_SUBMITTED = "task_submitted"
    DAG_UPDATED = "dag_updated"
    STEP_READY = "step_ready"
    STEP_CLAIMED = "step_claimed"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    STEP_RETRY = "step_retry"
    STEP_DEAD = "step_dead"
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

@dataclass
class StepDefine:
    step_id: str
    step_type: str
    input_data: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    max_retries: int = 3
    version: int = 1

# =========================
# 任务投影（增强版）
# =========================
class TaskProjection:
    def __init__(self, task_id: str, task_text: str):
        self.task_id = task_id
        self.task_text = task_text
        self.steps: Dict[str, StepDefine] = {}
        self.completed: Set[str] = set()
        self.running: Dict[str, Dict] = {}
        self.ready_pool: Set[str] = set()
        self.failed_steps: Set[str] = set()
        self.retry_counts: Dict[str, int] = {}
        self.is_terminal = False
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.history: List[Dict] = []
        self.step_records: List[Dict] = []
        self.phase_counter = 0
        self.step_created_at: Dict[str, float] = {}
        self.step_last_ready_time: Dict[str, float] = {}
        self._step_record_map: Dict[str, Dict] = {}

    def _add_steps(self, steps_data: List[Dict], event_timestamp: float):
        steps_added = 0
        for s_dict in steps_data:
            sd = StepDefine(**s_dict)
            if sd.step_id not in self.steps:
                invalid_deps = [dep for dep in sd.dependencies if dep not in self.steps]
                if invalid_deps:
                    logger.error(f"Step {sd.step_id[:8]} depends on non-existent steps: {invalid_deps}")
                    continue
                self.steps[sd.step_id] = sd
                self.step_created_at[sd.step_id] = event_timestamp
                self.phase_counter += 1
                record = {
                    "phase": self.phase_counter,
                    "step_id": sd.step_id,
                    "step_type": sd.step_type,
                    "created_at": event_timestamp,
                    "input": sd.input_data,
                    "dependencies": sd.dependencies,
                    "status": "created"
                }
                self.step_records.append(record)
                self._step_record_map[sd.step_id] = record
                steps_added += 1
        self._detect_cycle()
        return steps_added

    def _detect_cycle(self):
        graph = {sid: set(step.dependencies) for sid, step in self.steps.items()}
        visited = set()
        stack = set()

        def dfs(node):
            if node in stack:
                raise ValueError(f"循环依赖检测到: {node} -> {stack}")
            if node in visited:
                return
            visited.add(node)
            stack.add(node)
            for dep in graph.get(node, []):
                if dep not in self.steps:
                    continue
                dfs(dep)
            stack.remove(node)

        try:
            for node in graph:
                if node not in visited:
                    dfs(node)
        except ValueError as e:
            logger.error(f"任务 {self.task_id[:8]} 存在循环依赖: {e}")

    def apply(self, event: Event):
        etype = event.event_type
        sid = event.step_id
        payload = event.payload

        logger.debug(f"[Projection] Applying {etype.value} for task {self.task_id[:8]}")

        if etype == EventType.TASK_SUBMITTED:
            self.task_text = payload.get("task", self.task_text)
            initial_steps = payload.get("dag", [])
            if initial_steps:
                added = self._add_steps(initial_steps, event.timestamp)
                logger.info(f"[Projection] Task initialized with {added} initial steps")
            logger.info(f"[Projection] Task initialized: {self.task_text[:50]}...")

        elif etype == EventType.DAG_UPDATED:
            steps_data = payload.get("steps", [])
            added = self._add_steps(steps_data, event.timestamp)
            logger.info(f"[Projection] DAG updated: +{added} steps, total={len(self.steps)}")

        elif etype in (EventType.TASK_COMPLETED, EventType.TASK_FAILED):
            self.is_terminal = True
            self.end_time = event.timestamp
            logger.info(f"[Projection] Task terminated: {etype.value}")

        if not sid:
            return

        if etype in (EventType.STEP_READY, EventType.STEP_RETRY):
            self.ready_pool.add(sid)
            self.step_last_ready_time[sid] = event.timestamp
            logger.debug(f"[Projection] Step {sid[:8]} added to ready_pool")

        elif etype == EventType.STEP_CLAIMED:
            self.ready_pool.discard(sid)
            self.running[sid] = {
                "worker": payload.get("worker", "unknown"),
                "start_time": event.timestamp
            }
            record = self._step_record_map.get(sid)
            if record:
                record["started_at"] = event.timestamp
                record["worker_id"] = payload.get("worker")
                record["status"] = "running"
            logger.debug(f"[Projection] Step {sid[:8]} claimed by {payload.get('worker')}")

        elif etype == EventType.STEP_COMPLETED:
            self.running.pop(sid, None)
            self.completed.add(sid)
            record = self._step_record_map.get(sid)
            if record:
                record["completed_at"] = event.timestamp
                duration = event.timestamp - record.get("started_at", event.timestamp)
                record["duration"] = duration
                record["output"] = payload.get("output", "")[:500]
                record["next_step"] = payload.get("next_step", "done")
                record["is_terminal"] = payload.get("is_terminal", False)
                record["status"] = "completed"
            self.history.append({
                "step_id": sid,
                "step_type": self.steps.get(sid, StepDefine(sid, "unknown", {})).step_type,
                "output": payload.get("output", "")[:200],
                "timestamp": event.timestamp,
                "duration": duration
            })
            logger.info(f"[Projection] Step {sid[:8]} completed ({duration:.1f}s) -> {payload.get('next_step', 'done')}")

        elif etype == EventType.STEP_FAILED:
            self.running.pop(sid, None)
            self.retry_counts[sid] = self.retry_counts.get(sid, 0) + 1
            logger.warning(f"[Projection] Step {sid[:8]} failed, retry={self.retry_counts[sid]}")

        elif etype == EventType.STEP_RETRY:
            self.ready_pool.add(sid)
            self.step_last_ready_time[sid] = event.timestamp
            self.running.pop(sid, None)
            logger.info(f"[Projection] Step {sid[:8]} retried, added back to ready_pool")

        elif etype == EventType.STEP_DEAD:
            self.running.pop(sid, None)
            self.failed_steps.add(sid)
            logger.error(f"[Projection] Step {sid[:8]} marked as DEAD")

    def get_runnable_steps(self) -> List[StepDefine]:
        if self.is_terminal:
            return []
        runnable = []
        for sid, step in self.steps.items():
            if sid in self.completed or sid in self.running or sid in self.ready_pool or sid in self.failed_steps:
                continue
            deps = set(step.dependencies)
            if any(d not in self.steps for d in deps):
                continue
            if any(d in self.failed_steps for d in deps):
                continue
            if deps.issubset(self.completed):
                runnable.append(step)
        return runnable

    def check_deadlock(self, min_age_seconds: float = 0.5) -> bool:
        if self.is_terminal:
            return False
        runnable = self.get_runnable_steps()
        if runnable:
            return False
        incomplete = set(self.steps.keys()) - self.completed - self.failed_steps
        if not incomplete:
            return False
        has_active = len(self.running) > 0 or len(self.ready_pool) > 0
        if not has_active:
            now = time.time()
            fresh_steps = [
                sid for sid in incomplete
                if now - self.step_created_at.get(sid, 0) < min_age_seconds
            ]
            if fresh_steps:
                logger.debug(f"[Projection] Steps just created, skipping deadlock check")
                return False
        deadlocked = not has_active
        if deadlocked:
            logger.error(f"[Projection] DEADLOCK detected: {len(incomplete)} steps incomplete")
        return deadlocked

    def get_status(self) -> Dict:
        return {
            "total": len(self.steps),
            "completed": len(self.completed),
            "running": len(self.running),
            "ready": len(self.ready_pool),
            "failed": len(self.failed_steps),
            "terminal": self.is_terminal
        }

    def print_phase_report(self, step_id: str):
        record = self._step_record_map.get(step_id)
        if not record or record.get("status") != "completed":
            return
        phase = record["phase"]
        step_type = record["step_type"].upper()
        duration = record.get("duration", 0)
        started_at = record.get("started_at")
        completed_at = record.get("completed_at")
        start_str = time.strftime('%H:%M:%S', time.localtime(started_at)) if started_at else "N/A"
        end_str = time.strftime('%H:%M:%S', time.localtime(completed_at)) if completed_at else "N/A"
        print(f"\n{'─'*70}")
        print(f"📍 Phase {phase}: {step_type} ({duration:.1f}s)")
        print(f"{'─'*70}")
        print(f"⏰ 时间: {start_str} → {end_str}")
        print(f"👷 执行者: {record.get('worker_id', 'Unknown')}")
        print(f"🆔 步骤ID: {step_id[:8]}")
        if record.get("dependencies"):
            deps_str = ", ".join([d[:8] for d in record["dependencies"]])
            print(f"🔗 依赖步骤: {deps_str}")
        input_data = record.get("input", {})
        if input_data:
            print(f"\n📥 输入:")
            for key, value in list(input_data.items())[:3]:
                if key != "task":
                    value_str = str(value)[:60]
                    print(f"   • {key}: {value_str}{'...' if len(str(value)) > 60 else ''}")
        output = record.get("output", "")
        print(f"\n📤 输出:")
        if output:
            lines = output.split("\n")
            for i, line in enumerate(lines[:8]):
                if line.strip():
                    print(f"   {line[:120]}")
            if len(lines) > 8:
                print(f"   ... ({len(lines) - 8} more lines)")
        print(f"\n🎯 决策:")
        print(f"   next_step: {record.get('next_step', 'done')}")
        print(f"   is_terminal: {record.get('is_terminal', False)}")
        print(f"{'─'*70}")

    def print_final_summary(self):
        if not self.is_terminal:
            return
        total_duration = (self.end_time or time.time()) - self.start_time
        print(f"\n{'='*70}")
        print(f"📊 执行摘要")
        print(f"{'='*70}")
        print(f"任务ID: {self.task_id[:8]}")
        print(f"总耗时: {total_duration:.1f}s")
        print(f"总步骤数: {len(self.steps)}")
        print(f"成功完成: {len(self.completed)}")
        print(f"失败: {len(self.failed_steps)}")
        print(f"\n⏱️  阶段耗时明细:")
        for record in self.step_records:
            if record.get("duration"):
                bar = "█" * int(record["duration"] / 2)
                print(f"   Phase {record['phase']:2d} [{record['step_type']:10s}]: {record['duration']:5.1f}s {bar}")
        print(f"\n🗺️  执行流程:")
        completed_records = [r for r in self.step_records if r.get("status") == "completed"]
        flow = " → ".join([r["step_type"].upper() for r in completed_records])
        print(f"   {flow}")
        print(f"{'='*70}\n")

# =========================
# 事务性事件存储（增强版）
# =========================
class EventStore:
    def __init__(self, bus: 'EventBus'):
        self._logs = defaultdict(list)
        self._projections: Dict[str, TaskProjection] = {}
        self._bus = bus
        self._task_locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 60

    async def start_cleanup(self):
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(self._cleanup_interval)
            now = time.time()
            to_delete = []
            async with self._global_lock:
                for tid, proj in self._projections.items():
                    if proj.is_terminal and proj.end_time and now - proj.end_time > 300:
                        to_delete.append(tid)
                for tid in to_delete:
                    del self._projections[tid]
                    if tid in self._logs:
                        del self._logs[tid]
                    if tid in self._task_locks:
                        del self._task_locks[tid]
                    logger.info(f"[EventStore] Cleaned up task {tid[:8]}")

    def _get_task_lock(self, task_id: str) -> asyncio.Lock:
        if task_id not in self._task_locks:
            self._task_locks[task_id] = asyncio.Lock()
        return self._task_locks[task_id]

    async def append_and_publish(self, event: Event, max_retries: int = 1):
        tid = event.task_id
        lock = self._get_task_lock(tid)

        async with lock:
            self._logs[tid].append(event)

            async with self._global_lock:
                if event.event_type == EventType.TASK_SUBMITTED and tid not in self._projections:
                    task_text = event.payload.get("task", "")
                    self._projections[tid] = TaskProjection(tid, task_text)

            if tid in self._projections:
                self._projections[tid].apply(event)

            if event.event_type == EventType.STEP_COMPLETED and event.step_id:
                proj = self._projections[tid]
                asyncio.create_task(self._async_print_report(proj, event.step_id))

        for attempt in range(max_retries + 1):
            try:
                await self._bus.publish(event)
                return
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"[EventStore] Publish failed (attempt {attempt + 1}), retrying: {e}")
                    await asyncio.sleep(0.1 * (attempt + 1))
                else:
                    logger.error(f"[EventStore] Publish failed after {max_retries + 1} attempts: {e}")

    async def _async_print_report(self, proj: TaskProjection, step_id: str):
        proj.print_phase_report(step_id)

    async def try_claim_step(self, task_id: str, step_id: str, worker_id: str) -> bool:
        lock = self._get_task_lock(task_id)
        async with lock:
            proj = self._projections.get(task_id)
            if not proj:
                logger.warning(f"[EventStore] Claim failed: projection missing for {task_id[:8]}")
                return False
            if proj.is_terminal:
                logger.debug(f"[EventStore] Claim failed: task {task_id[:8]} is terminal")
                return False
            if step_id not in proj.ready_pool:
                logger.warning(f"[EventStore] Claim failed: step {step_id[:8]} not in ready_pool (current ready: {list(proj.ready_pool)[:5]})")
                return False

            event = Event(
                task_id=task_id,
                step_id=step_id,
                event_type=EventType.STEP_CLAIMED,
                payload={"worker": worker_id}
            )
            self._logs[task_id].append(event)
            proj.apply(event)
            try:
                await self._bus.publish(event)
                logger.info(f"[EventStore] Step {step_id[:8]} claimed by {worker_id}")
                return True
            except Exception as e:
                logger.error(f"[EventStore] Failed to publish STEP_CLAIMED: {e}")
                proj.ready_pool.add(step_id)
                proj.running.pop(step_id, None)
                return False

    def get_projection(self, task_id: str) -> Optional[TaskProjection]:
        return self._projections.get(task_id)

    def get_logs(self, task_id: str) -> List[Event]:
        return self._logs.get(task_id, [])

    async def shutdown(self):
        if self._cleanup_task:
            self._cleanup_task.cancel()
            await asyncio.sleep(0)

# =========================
# 事件总线（异步队列版）
# =========================
class EventBus:
    def __init__(self):
        self._subscribers: List[Callable] = []
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._process_queue())
        self._stop_event = asyncio.Event()

    def subscribe(self, handler: Callable):
        self._subscribers.append(handler)
        logger.info(f"[EventBus] Subscribed {handler.__name__}, total handlers: {len(self._subscribers)}")

    async def publish(self, event: Event):
        logger.info(f"[EventBus] Publishing event {event.event_type.value} for task {event.task_id[:8]}")
        await self._queue.put(event)

    async def _process_queue(self):
        while not self._stop_event.is_set():
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                logger.info(f"[EventBus] Processing event {event.event_type.value} for task {event.task_id[:8]}")
            except asyncio.TimeoutError:
                continue
            coros = [self._safe_run(handler, event) for handler in self._subscribers]
            results = await asyncio.gather(*coros, return_exceptions=True)
            for handler, result in zip(self._subscribers, results):
                if isinstance(result, Exception):
                    logger.error(f"[EventBus] Handler {handler.__name__} failed: {result}")
            self._queue.task_done()

    async def _safe_run(self, handler: Callable, event: Event) -> Optional[Exception]:
        try:
            await handler(event)
            return None
        except Exception as e:
            return e

    async def shutdown(self):
        self._stop_event.set()
        await self._worker_task

# =========================
# 智能调度器（增强版，含去重）
# =========================
class Scheduler:
    def __init__(self, store: EventStore, queue: asyncio.Queue):
        self.store = store
        self.queue = queue
        self._completion_futures: Dict[str, asyncio.Future] = {}
        self._last_dispatch: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._scan_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._pending_steps: Set[str] = set()
        self._state_lock = asyncio.Lock()

    async def start_periodic_scan(self, interval: float = 5.0):
        while not self._stop_event.is_set():
            await asyncio.sleep(interval)
            if self._stop_event.is_set():
                break
            await self._scan_and_retry_ready_steps()

    async def _scan_and_retry_ready_steps(self):
        now = time.time()
        for task_id, proj in self.store._projections.items():
            if proj.is_terminal:
                continue
            for step_id in list(proj.ready_pool):
                last_ready = proj.step_last_ready_time.get(step_id, 0)
                if now - last_ready > 10:
                    logger.info(f"[Scheduler] Re-dispatching step {step_id[:8]} (idle >10s)")
                    ready_event = Event(
                        task_id=task_id,
                        step_id=step_id,
                        event_type=EventType.STEP_READY,
                        payload={"step_type": proj.steps[step_id].step_type}
                    )
                    try:
                        await self.store.append_and_publish(ready_event)
                    except Exception as e:
                        logger.error(f"[Scheduler] Failed to re-dispatch step {step_id[:8]}: {e}")

    async def on_event(self, event: Event):
        logger.info(f"[Scheduler] Received {event.event_type.value} for {event.task_id[:8]}")

        if event.event_type in (EventType.TASK_COMPLETED, EventType.TASK_FAILED):
            fut = self._completion_futures.get(event.task_id)
            if fut and not fut.done():
                fut.set_result(event.event_type)
                proj = self.store.get_projection(event.task_id)
                if proj:
                    proj.print_final_summary()
                async with self._state_lock:
                    self._last_dispatch.pop(event.task_id, None)
            return

        if event.event_type == EventType.STEP_CLAIMED:
            async with self._state_lock:
                self._pending_steps.discard(event.step_id)
            return

        if event.event_type not in (
            EventType.TASK_SUBMITTED,
            EventType.STEP_COMPLETED,
            EventType.STEP_FAILED,
            EventType.STEP_RETRY,
            EventType.DAG_UPDATED,
            EventType.STEP_DEAD,
            EventType.STEP_READY
        ):
            return

        task_lock = self.store._get_task_lock(event.task_id)
        async with task_lock:
            proj = self.store.get_projection(event.task_id)
            if not proj or proj.is_terminal:
                return

            if event.event_type == EventType.STEP_READY:
                step_id = event.step_id
                async with self._state_lock:
                    if step_id in self._pending_steps:
                        logger.debug(f"[Scheduler] Step {step_id[:8]} already pending, skipping")
                        return
                    self._pending_steps.add(step_id)
                await self.queue.put((event.task_id, step_id))
                logger.info(f"[Scheduler] Re-dispatched step {step_id[:8]} from ready event")
                return

            runnable = proj.get_runnable_steps()
            if runnable:
                logger.info(f"[Scheduler] Found {len(runnable)} runnable steps for task {event.task_id[:8]}")
                for step in runnable:
                    async with self._state_lock:
                        if step.step_id in self._pending_steps:
                            continue
                        now = time.time()
                        last = self._last_dispatch[event.task_id].get(step.step_id, 0)
                        if now - last < 1.0:
                            continue
                        self._last_dispatch[event.task_id][step.step_id] = now
                        self._pending_steps.add(step.step_id)

                    ready_event = Event(
                        task_id=event.task_id,
                        step_id=step.step_id,
                        event_type=EventType.STEP_READY,
                        payload={"step_type": step.step_type}
                    )
                    try:
                        await self.store.append_and_publish(ready_event)
                        await self.queue.put((event.task_id, step.step_id))
                        logger.info(f"[Scheduler] ✅ Dispatched step {step.step_id[:8]} ({step.step_type})")
                    except Exception as e:
                        async with self._state_lock:
                            self._pending_steps.discard(step.step_id)
                        logger.error(f"[Scheduler] Failed to dispatch step {step.step_id[:8]}: {e}")
                return

            await self._check_completion(event.task_id, proj)

            if proj.check_deadlock(min_age_seconds=0.5):
                logger.error(f"[Scheduler] Deadlock detected for {event.task_id[:8]}")
                try:
                    await self.store.append_and_publish(Event(
                        task_id=event.task_id,
                        event_type=EventType.TASK_FAILED,
                        payload={"reason": "deadlock"}
                    ))
                except Exception as e:
                    logger.error(f"[Scheduler] Failed to publish deadlock event: {e}")

    async def _check_completion(self, task_id: str, proj: TaskProjection):
        status = proj.get_status()
        if status["total"] > 0 and status["running"] == 0:
            if status["completed"] + status["failed"] >= status["total"]:
                if proj.is_terminal:
                    return
                final_type = EventType.TASK_COMPLETED if status["failed"] == 0 else EventType.TASK_FAILED
                logger.info(f"[Scheduler] All steps done, triggering {final_type.value}")
                try:
                    await self.store.append_and_publish(Event(
                        task_id=task_id,
                        event_type=final_type
                    ))
                except Exception as e:
                    logger.error(f"[Scheduler] Failed to publish completion event: {e}")

    def register_completion(self, task_id: str, future: asyncio.Future):
        self._completion_futures[task_id] = future

    async def shutdown(self):
        self._stop_event.set()
        if self._scan_task:
            self._scan_task.cancel()
            await asyncio.sleep(0)

# =========================
# Worker（增强版，含客户端关闭）
# =========================
class Worker:
    def __init__(self, wid: str, queue: asyncio.Queue, store: EventStore, scheduler: Scheduler):
        self.wid = wid
        self.queue = queue
        self.store = store
        self.scheduler = scheduler
        self.llm = AsyncOpenAI(
            base_url=f"{VLLM_URL}/v1",
            api_key="sk-ignore",
            timeout=120.0
        )
        self._running = True
        self._stop_event = asyncio.Event()
        self._global_semaphore = asyncio.Semaphore(1)

    def stop(self):
        self._running = False
        self._stop_event.set()

    async def run(self):
        logger.info(f"[{self.wid}] Worker started")
        try:
            while self._running:
                try:
                    task_id, step_id = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                    logger.info(f"[{self.wid}] Got task from queue: {step_id[:8]}")
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"[{self.wid}] Error getting from queue: {e}")
                    continue
                try:
                    success = await self._process_task(task_id, step_id)
                finally:
                    self.queue.task_done()
                if not success:
                    logger.warning(f"[{self.wid}] Task processing failed for {step_id[:8]}")
        finally:
            await self.llm.close()
            logger.info(f"[{self.wid}] Worker shut down")

    async def _process_task(self, task_id: str, step_id: str) -> bool:
        try:
            proj = self.store.get_projection(task_id)
            if not proj:
                logger.error(f"[{self.wid}] Task {task_id[:8]} projection not found")
                return False
            if proj.is_terminal:
                logger.debug(f"[{self.wid}] Task {task_id[:8]} is terminal, skipping step {step_id[:8]}")
                return False

            step = proj.steps.get(step_id)
            if not step:
                logger.error(f"[{self.wid}] Step {step_id[:8]} not found in steps")
                return False

            claimed = False
            for attempt in range(3):
                if await self.store.try_claim_step(task_id, step_id, self.wid):
                    claimed = True
                    break
                await asyncio.sleep(0.1 * (attempt + 1))
            if not claimed:
                proj = self.store.get_projection(task_id)
                if proj and step_id in proj.ready_pool:
                    logger.warning(f"[{self.wid}] Failed to claim step {step_id[:8]}, re-queueing")
                    await self.queue.put((task_id, step_id))
                else:
                    logger.warning(f"[{self.wid}] Failed to claim step {step_id[:8]}, step already claimed")
                return False

            logger.info(f"[{self.wid}] 🚀 Executing step {step_id[:8]}")
            start_time = time.time()
            async with self._global_semaphore:
                try:
                    result = await asyncio.wait_for(
                        self._execute_llm(step, proj),
                        timeout=120
                    )
                except asyncio.TimeoutError:
                    logger.error(f"[{self.wid}] Step {step_id[:8]} LLM timeout")
                    raise Exception("LLM timeout")
                except Exception as e:
                    logger.error(f"[{self.wid}] Step {step_id[:8]} LLM execution failed: {e}", exc_info=True)
                    raise

            await self._handle_dag_extension(task_id, step_id, step, result, proj)

            total_time = time.time() - start_time
            logger.info(f"[{self.wid}] ✅ Step {step_id[:8]} total time: {total_time:.1f}s")
            return True

        except Exception as e:
            logger.error(f"[{self.wid}] 💥 Exception processing step {step_id[:8]}: {e}", exc_info=True)

            proj = self.store.get_projection(task_id)
            retries = proj.retry_counts.get(step_id, 0) if proj else 0
            step = proj.steps.get(step_id) if proj else None
            max_retries = step.max_retries if step else 3

            if retries < max_retries:
                await self.store.append_and_publish(Event(
                    task_id=task_id,
                    step_id=step_id,
                    event_type=EventType.STEP_RETRY
                ))
                await self.scheduler.on_event(Event(
                    task_id=task_id,
                    step_id=step_id,
                    event_type=EventType.STEP_RETRY
                ))
            else:
                await self.store.append_and_publish(Event(
                    task_id=task_id,
                    step_id=step_id,
                    event_type=EventType.STEP_DEAD,
                    payload={"error": str(e)}
                ))
            return False

    async def _execute_llm(self, step: StepDefine, proj: TaskProjection) -> Dict:
        dep_results = {}
        for dep_id in step.dependencies:
            record = proj._step_record_map.get(dep_id)
            if record and record.get("status") == "completed":
                dep_results[dep_id] = record.get("output", "")[:800]

        all_completed = []
        for record in proj.step_records:
            if record.get("status") == "completed" and record["step_id"] not in dep_results:
                all_completed.append({
                    "step_id": record["step_id"],
                    "step_type": record["step_type"],
                    "output": record.get("output", "")[:400],
                    "completed_at": record.get("completed_at", 0)
                })
        all_completed.sort(key=lambda x: x["completed_at"], reverse=True)
        recent_context = all_completed[:3]

        context_parts = []
        if dep_results:
            context_parts.append("=== Direct Dependencies ===")
            for dep_id, output in dep_results.items():
                short_id = dep_id[:8]
                context_parts.append(f"[{short_id}]: {output[:300]}")
        if recent_context:
            context_parts.append("\n=== Additional Context ===")
            for ctx in recent_context:
                short_id = ctx["step_id"][:8]
                context_parts.append(f"[{short_id} {ctx['step_type']}]: {ctx['output'][:200]}")
        full_context = "\n".join(context_parts) if context_parts else "No previous context available."

        valid_types = ["think", "analyze", "reflect", "plan", "search", "done"]
        step_type = step.step_type
        if step_type == "think":
            stage_instruction = """- 输出对任务的拆解逻辑和初步思考，至少 200 字。
    - 必须指定至少一个下一步类型（analyze/plan/search等），不能直接结束。"""
        elif step_type == "analyze":
            stage_instruction = """- 提供深度分析，至少 300 字，内容应涵盖数据解读、因果关系、市场影响等。
    - 分析要有层次，可使用小标题或分段。"""
        elif step_type == "plan":
            stage_instruction = """- 输出具体的投资策略或执行计划，至少 200 字。
    - 策略应包含操作方向、风险控制、时机判断等要素。"""
        elif step_type == "search":
            stage_instruction = """- 为后续分析收集和整理关键数据，输出不少于 150 字。
    - 列出需要关注的核心指标、数据来源或当前已知的关键数据点。"""
        elif step_type == "reflect":
            stage_instruction = """- 反思已有分析的不足，指出可能遗漏的因素，并提出改进方向或后续验证点。
    - 输出不少于 150 字。"""
        else:
            stage_instruction = f"- 根据阶段类型提供详细分析或计划，至少 150 字。\n- 严禁输出空的 'output'。"

        prompt = f"""你是一个高级金融分析师。当前任务：{proj.task_text}
当前阶段类型：{step.step_type}

【上下文信息】
{full_context}

【执行指令】
{stage_instruction}
- 严禁输出空的 "output"。
- 如果任务尚未完成，"next_step" 不允许为 "done"，必须从 {valid_types[:-1]} 中选择。

【响应格式】
必须返回如下 JSON 格式，不要包含任何开场白：
{{
    "next_step": "下一个阶段类型",
    "next_steps": ["可选的并行阶段"],
    "output": "此处填写详细的分析内容或计划路线",
    "is_terminal": false
}}"""

        try:
            resp = await self.llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=6000
            )
            content = resp.choices[0].message.content
            logger.debug(f"[{self.wid}] Raw LLM response: {content[:500]}")

            data = {}
            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    try:
                        fixed = json_match.group(1).replace("'", '"')
                        data = json.loads(fixed)
                    except:
                        data = {"output": content}
            else:
                data = {"output": content}

            if not data.get("output") or str(data["output"]).strip() == "":
                data["output"] = content if content.strip() else "模型未生成有效内容"
            if isinstance(data, list):
                data = {"output": json.dumps(data)}
            if not isinstance(data, dict):
                data = {"output": str(data)}

            if not data.get("output") or data["output"].strip() == "":
                data["output"] = content[:800] if content else "无输出"

            data.setdefault("delta", {})
            data.setdefault("next_step", "done")
            data.setdefault("next_steps", [])
            data.setdefault("dependencies", [])
            data.setdefault("is_terminal", data["next_step"] == "done")

            return data
        except Exception as e:
            return {
                "delta": {"error": str(e)},
                "next_step": "done",
                "next_steps": [],
                "dependencies": [],
                "output": f"Error: {str(e)}",
                "is_terminal": True
            }

    async def _handle_dag_extension(self, task_id: str, step_id: str, step: StepDefine, result: Dict, proj: TaskProjection):
        next_steps = result.get("next_steps", [])
        single_next = result.get("next_step", "done")
        if single_next != "done" and single_next not in next_steps:
            next_steps.append(single_next)

        if not next_steps or next_steps == ["done"]:
            await self.store.append_and_publish(Event(
                task_id=task_id,
                step_id=step_id,
                event_type=EventType.STEP_COMPLETED,
                payload={
                    "output": result.get("output", ""),
                    "next_step": "done",
                    "is_terminal": True,
                    "delta": result.get("delta", {})
                }
            ))
            return

        existing_types = {
            s.step_type for s in proj.steps.values()
            if s.step_type not in proj.completed and s.step_type not in proj.failed_steps
        }
        new_steps = []
        for i, next_step_type in enumerate(next_steps):
            if next_step_type == "done":
                continue
            if next_step_type in existing_types:
                logger.debug(f"[{self.wid}] Skipping duplicate step type {next_step_type}")
                continue
            new_step_id = f"step-{uuid.uuid4().hex[:8]}"
            dependencies = result.get("dependencies", [step_id])
            if not dependencies:
                dependencies = [step_id]
            input_data = {"previous": result.get("output", "")[:300]}
            if i > 0:
                input_data["branch"] = i
            new_step = StepDefine(
                step_id=new_step_id,
                step_type=next_step_type,
                input_data=input_data,
                dependencies=dependencies
            )
            new_steps.append(new_step.__dict__)

        if new_steps:
            logger.info(f"[{self.wid}] Creating {len(new_steps)} new steps: {[s['step_type'] for s in new_steps]}")
            await self.store.append_and_publish(Event(
                task_id=task_id,
                event_type=EventType.DAG_UPDATED,
                payload={"steps": new_steps}
            ))

        await self.store.append_and_publish(Event(
            task_id=task_id,
            step_id=step_id,
            event_type=EventType.STEP_COMPLETED,
            payload={
                "output": result.get("output", ""),
                "next_step": next_steps[0] if next_steps else "done",
                "is_terminal": False,
                "delta": result.get("delta", {}),
                "created_steps": [s["step_id"] for s in new_steps]
            }
        ))

# =========================
# 主控（增强版）
# =========================
class ProductionAgentOS:
    def __init__(self, worker_count: int = 2):
        self.bus = EventBus()
        self.store = EventStore(self.bus)
        self.queue = asyncio.Queue()
        self.scheduler = Scheduler(self.store, self.queue)
        self.worker_count = worker_count
        self.workers: List[Worker] = []
        self._global_timeout_task: Optional[asyncio.Task] = None

        self.bus.subscribe(self.scheduler.on_event)
        logger.info(f"🚀 ProductionAgentOS initialized with {worker_count} workers")

    async def run(self, task: str, timeout: float = 30.0) -> Dict:
        task_id = str(uuid.uuid4())
        print(f"\n{'='*70}")
        print(f"🚀 任务启动")
        print(f"{'='*70}")
        print(f"任务ID: {task_id[:8]}")
        print(f"任务描述: {task[:60]}...")
        print(f"开始时间: {time.strftime('%H:%M:%S')}")
        print(f"Workers: {self.worker_count}")
        print(f"{'='*70}\n")

        self.workers = [Worker(f"W-{i}", self.queue, self.store, self.scheduler) for i in range(self.worker_count)]
        worker_tasks = [asyncio.create_task(w.run()) for w in self.workers]

        self.scheduler._scan_task = asyncio.create_task(self.scheduler.start_periodic_scan(interval=5.0))
        await self.store.start_cleanup()

        await asyncio.sleep(0.3)

        completion_future = asyncio.get_running_loop().create_future()
        self.scheduler.register_completion(task_id, completion_future)

        async def global_timeout():
            try:
                await asyncio.sleep(timeout)
                if not completion_future.done():
                    logger.error(f"⏱️ 全局超时 {timeout}s，强制结束任务")
                    await self.store.append_and_publish(Event(
                        task_id=task_id,
                        event_type=EventType.TASK_FAILED,
                        payload={"reason": "global_timeout"}
                    ))
                    for w in self.workers:
                        w.stop()
                    for t in worker_tasks:
                        t.cancel()
            except asyncio.CancelledError:
                pass

        self._global_timeout_task = asyncio.create_task(global_timeout())

        init_step = StepDefine(
            step_id=f"step-{uuid.uuid4().hex[:8]}",
            step_type="think",
            input_data={"task": task, "phase": "initial_analysis"},
            dependencies=[]
        )

        try:
            await self.store.append_and_publish(Event(
                task_id=task_id,
                event_type=EventType.TASK_SUBMITTED,
                payload={
                    "task": task,
                    "dag": [init_step.__dict__]
                }
            ))
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            for t in worker_tasks:
                t.cancel()
            if self._global_timeout_task:
                self._global_timeout_task.cancel()
            return {"task_id": task_id, "status": "submit_failed", "error": str(e)}

        try:
            result_status = await asyncio.wait_for(completion_future, timeout=timeout + 5)
            proj = self.store.get_projection(task_id)
            result = {
                "task_id": task_id,
                "status": result_status.value,
                "completed_steps": len(proj.completed) if proj else 0,
                "total_steps": len(proj.steps) if proj else 0,
                "duration": (proj.end_time - proj.start_time) if proj and proj.end_time else 0
            }
            return result
        except asyncio.TimeoutError:
            logger.error("⏱️ 任务超时")
            proj = self.store.get_projection(task_id)
            if proj:
                logger.error(f"Final status: {proj.get_status()}")
            return {"task_id": task_id, "status": "timeout", "duration": timeout}
        finally:
            if self._global_timeout_task:
                self._global_timeout_task.cancel()
            await self.scheduler.shutdown()
            logger.info("Shutting down workers...")
            for w in self.workers:
                w.stop()
            try:
                await asyncio.wait_for(asyncio.gather(*worker_tasks, return_exceptions=True), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Worker shutdown timeout")
            await self.bus.shutdown()
            await self.store.shutdown()

# =========================
# 运行入口
# =========================
async def main():
    agent = ProductionAgentOS(worker_count=2)
    task = "分析 AH 股溢价对当前市场情绪的影响，并提出投资策略"
    result = await agent.run(task, timeout=300)
    print(f"\n{'='*70}")
    print(f"🏁 最终结果")
    print(f"{'='*70}")
    print(f"Status: {result['status']}")
    print(f"Completed: {result['completed_steps']}/{result['total_steps']} steps")
    print(f"Duration: {result['duration']:.1f}s")
    print(f"{'='*70}")

if __name__ == "__main__":
    asyncio.run(main())
