import asyncio
import uuid
import time
import logging
import json
import os
import re
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Set, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from openai import AsyncOpenAI
import urllib.request
import urllib.error
import httpx

# =========================
# 配置与日志
# =========================
VLLM_URL = os.environ.get("VLLM_URL", "http://192.168.1.210:19000")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen3Qoder")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# 搜索工具配置
SEARCH_ENGINE = os.environ.get("SEARCH_ENGINE", "tavily")  # tavily/bing/google
BING_API_KEY = os.environ.get("BING_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "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")  # Tavily API Key
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("ProductionAgentOS")


# =========================
# Tavily 搜索工具
# =========================
class TavilySearchTool:
    """Tavily 搜索工具，用于实时网络搜索获取信息"""

    def __init__(self, api_key: str = TAVILY_API_KEY):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        include_answer: bool = False,
        include_raw_content: bool = False,
        include_images: bool = False
    ) -> Dict[str, Any]:
        """
        使用 Tavily API 进行搜索

        Args:
            query: 搜索查询
            max_results: 最大结果数
            search_depth: 搜索深度 ("basic" 或 "advanced")
            include_domains: 包含的域名列表
            exclude_domains: 排除的域名列表
            include_answer: 是否包含 AI 答案
            include_raw_content: 是否包含原始内容
            include_images: 是否包含图片

        Returns:
            搜索结果字典
        """
        if not self.api_key:
            logger.warning("Tavily API key not configured, returning mock results")
            return self._mock_search(query, max_results)

        try:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
                "include_images": include_images
            }

            if include_domains:
                payload["include_domains"] = include_domains
            if exclude_domains:
                payload["exclude_domains"] = exclude_domains

            response = await self.client.post(
                TAVILY_SEARCH_URL,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()

            logger.info(f"[Tavily] Search completed, found {len(result.get('results', []))} results")
            return result

        except httpx.TimeoutException:
            logger.error("[Tavily] Search timeout")
            return self._mock_search(query, max_results)
        except Exception as e:
            logger.error(f"[Tavily] Search failed: {e}")
            return self._mock_search(query, max_results)

    async def search_finance(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        金融相关搜索，优先从财经网站获取信息

        Args:
            query: 搜索查询
            max_results: 最大结果数

        Returns:
            搜索结果字典
        """
        finance_domains = [
            "investing.com", "finance.yahoo.com", "bloomberg.com",
            "reuters.com", "cnbc.com", "wallstreetjournal.com",
            "seekingalpha.com", "marketwatch.com", "ft.com"
        ]
        return await self.search(
            query,
            max_results=max_results,
            include_domains=finance_domains,
            search_depth="advanced"
        )

    def format_results(self, results: Dict[str, Any]) -> str:
        """
        格式化搜索结果为可读文本

        Args:
            results: Tavily 搜索结果

        Returns:
            格式化的搜索结果文本
        """
        if not results or "results" not in results:
            return "未找到相关搜索结果"

        formatted = []

        # 添加 AI 答案（如果有）
        if results.get("answer"):
            formatted.append(f"📌 AI 摘要:\n{results['answer']}\n")

        # 添加搜索结果
        for i, result in enumerate(results.get("results", [])[:10], 1):
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

    def _mock_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        模拟搜索结果（用于无 API 配置时）

        Args:
            query: 搜索查询
            max_results: 最大结果数

        Returns:
            模拟的搜索结果
        """
        logger.info(f"[Mock Search] Query: {query}")

        mock_results = [
            {
                "title": f"关于 '{query}' 的最新分析 - 金融新闻",
                "url": "https://example.com/finance-news",
                "content": f"这是关于 {query} 的最新市场分析和报道...",
                "score": 0.95
            },
            {
                "title": f"{query} - 股票行情和数据",
                "url": "https://example.com/stock-data",
                "content": "实时股票数据、历史价格和交易量信息...",
                "score": 0.88
            },
            {
                "title": f"专家解读：{query}",
                "url": "https://example.com/expert-analysis",
                "content": f"行业专家对 {query} 的深度解读和投资建议...",
                "score": 0.82
            },
            {
                "title": f"{query} 市场趋势报告",
                "url": "https://example.com/market-report",
                "content": "最新的市场趋势分析和预测数据...",
                "score": 0.75
            },
            {
                "title": f"{query} 相关新闻汇总",
                "url": "https://example.com/news-summary",
                "content": f"汇总了近期关于 {query} 的重要新闻和事件...",
                "score": 0.70
            }
        ]

        return {
            "query": query,
            "results": mock_results[:max_results],
            "answer": None,
            "images": []
        }

    async def close(self):
        """关闭 HTTP 客户端"""
        await self.client.aclose()


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
# 任务投影
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

    def _add_steps(self, steps_data: List[Dict], event_timestamp: float) -> int:
        steps_added = 0
        for s_dict in steps_data:
            required_fields = ["step_id", "step_type", "input_data"]
            missing = [f for f in required_fields if f not in s_dict]
            if missing:
                logger.error(f"Step missing required fields: {missing}")
                continue
            try:
                sd = StepDefine(**s_dict)
            except TypeError as e:
                logger.error(f"Failed to create StepDefine: {e}")
                continue
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
                raise ValueError(f"循环依赖检测到：{node} -> {stack}")
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
            logger.error(f"任务 {self.task_id[:8]} 存在循环依赖：{e}")

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
            # 新增步骤后，检查是否有就绪的步骤
            if added > 0:
                self._check_ready_steps()

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
            duration = 0
            is_terminal_step = payload.get("is_terminal", False)
            if record:
                record["completed_at"] = event.timestamp
                duration = event.timestamp - record.get("started_at", event.timestamp)
                record["duration"] = duration
                record["output"] = payload.get("output", "")[:500]
                record["next_step"] = payload.get("next_step", "done")
                record["is_terminal"] = is_terminal_step
                record["status"] = "completed"
            self.history.append({
                "step_id": sid,
                "step_type": self.steps.get(sid, StepDefine(sid, "unknown", {})).step_type,
                "output": payload.get("output", "")[:200],
                "timestamp": event.timestamp,
                "duration": duration
            })
            logger.info(f"[Projection] Step {sid[:8]} completed ({duration:.1f}s) -> {payload.get('next_step', 'done')}")
            # 🔧 修复：如果步骤标记为终端步骤，设置任务为终止状态
            if is_terminal_step:
                self.is_terminal = True
                self.end_time = event.timestamp
                logger.info(f"[Projection] Task terminated: STEP_COMPLETED is_terminal=True")
            else:
                # 🔧 新增：步骤完成后检查是否有新的就绪步骤
                self._check_ready_steps()

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

    def _check_ready_steps(self):
        """检查并更新就绪步骤池（用于 DAG 更新后）"""
        for step_id, step in self.steps.items():
            if step_id in self.completed or step_id in self.running or step_id in self.ready_pool:
                continue
            deps = set(step.dependencies)
            if any(d not in self.steps for d in deps):
                continue
            if any(d in self.failed_steps for d in deps):
                continue
            if deps.issubset(self.completed):
                self.ready_pool.add(step_id)
                self.step_last_ready_time[step_id] = time.time()

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
        print(f"⏰ 时间：{start_str} → {end_str}")
        print(f"👷 执行者：{record.get('worker_id', 'Unknown')}")
        print(f"🆔 步骤 ID: {step_id[:8]}")
        if record.get("dependencies"):
            deps_str = ", ".join([d[:8] for d in record["dependencies"]])
            print(f"🔗 依赖步骤：{deps_str}")
        input_data = record.get("input", {})
        if input_data:
            print(f"\n📥 输入:")
            for _i, (key, value) in enumerate(list(input_data.items())[:3]):
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
        print(f"任务 ID: {self.task_id[:8]}")
        print(f"总耗时：{total_duration:.1f}s")
        print(f"总步骤数：{len(self.steps)}")
        print(f"成功完成：{len(self.completed)}")
        print(f"失败：{len(self.failed_steps)}")
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
# 事务性事件存储 (🔧 修复退出慢)
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
        self._stop_event = asyncio.Event()  # 🔧 新增：停止信号

    async def start_cleanup(self):
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        # 🔧 修复：可立即响应取消的清理循环
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._cleanup_interval)
                break
            except asyncio.TimeoutError:
                pass
            now = time.time()
            to_delete = []
            async with self._global_lock:
                for tid, proj in list(self._projections.items()):
                    if proj.is_terminal and proj.end_time and now - proj.end_time > 300:
                        to_delete.append(tid)
                for tid in to_delete:
                    del self._projections[tid]
                    self._logs.pop(tid, None)
                    self._task_locks.pop(tid, None)
                if to_delete:
                    logger.info(f"[EventStore] Cleaned up {len(to_delete)} tasks")

    def _get_task_lock(self, task_id: str) -> asyncio.Lock:
        if task_id not in self._task_locks:
            self._task_locks[task_id] = asyncio.Lock()
        return self._task_locks[task_id]

    async def append_and_publish(self, event: Event, max_retries: int = 3):
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

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                await self._bus.publish(event)
                return
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(f"[EventStore] Publish failed (attempt {attempt + 1}), retrying: {e}")
                    await asyncio.sleep(0.05 * (attempt + 1))  # 🔧 优化：减少重试间隔
                else:
                    logger.error(f"[EventStore] Publish failed after {max_retries + 1} attempts: {e}")
        if last_error:
            raise last_error

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
        # 🔧 修复：通知 cleanup 循环立即退出
        self._stop_event.set()
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        await asyncio.sleep(0)

# =========================
# 事件总线 (🔧 修复退出慢)
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
        # 🔧 优化：减少日志输出，只在调试模式下输出
        if logger.isEnabledFor(logging.DEBUG):
            logger.info(f"[EventBus] Publishing event {event.event_type.value} for task {event.task_id[:8]}")
        await self._queue.put(event)

    async def _process_queue(self):
        while not self._stop_event.is_set():
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                if event is None:  # 🔧 毒丸事件，退出循环
                    break
                # 🔧 优化：减少日志输出
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[EventBus] Processing event {event.event_type.value} for task {event.task_id[:8]}")
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
        # 🔧 修复：放入毒丸事件唤醒队列，带超时等待
        self._stop_event.set()
        await self._queue.put(None)
        try:
            await asyncio.wait_for(self._worker_task, timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("[EventBus] Shutdown timeout, forcing exit")
            self._worker_task.cancel()
        await asyncio.sleep(0)

# =========================
# 智能调度器
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
        # 🔧 优化：减少日志输出
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Scheduler] Received {event.event_type.value} for {event.task_id[:8]}")

        # === 终端事件处理（保持不变）===
        if event.event_type in (EventType.TASK_COMPLETED, EventType.TASK_FAILED):
            fut = self._completion_futures.get(event.task_id)
            if fut and not fut.done():
                fut.set_result(event.event_type)
                logger.info(f"[Scheduler] ✅ Completion future set: {event.event_type.value}")
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

        # === STEP_COMPLETED 事件处理：检查是否是终端步骤 ===
        if event.event_type == EventType.STEP_COMPLETED:
            proj = self.store.get_projection(event.task_id)
            if proj and not proj.is_terminal:
                # 检查是否是最后一个步骤（没有未完成的步骤）
                has_uncompleted = any(
                    sid not in proj.completed and sid not in proj.failed_steps
                    for sid in proj.steps.keys()
                )
                is_terminal_step = event.payload.get("is_terminal", False)
                # 如果标记为终端步骤且没有未完成的步骤，完成任务
                if is_terminal_step and not has_uncompleted:
                    logger.info(f"[Scheduler] ✅ All steps completed, publishing TASK_COMPLETED")
                    # 发布 TASK_COMPLETED 事件
                    await self.store.append_and_publish(Event(
                        task_id=event.task_id,
                        event_type=EventType.TASK_COMPLETED,
                        payload={"reason": "all_steps_completed"}
                    ))
                    proj.is_terminal = True
                    proj.end_time = event.timestamp
                    async with self._state_lock:
                        self._last_dispatch.pop(event.task_id, None)
                    return

        if event.event_type not in (
            EventType.TASK_SUBMITTED, EventType.STEP_FAILED,
            EventType.STEP_RETRY, EventType.STEP_DEAD, EventType.STEP_READY, EventType.DAG_UPDATED
        ):
            return

        # === 收集需要执行的操作（锁内只读）===
        task_lock = self.store._get_task_lock(event.task_id)
        events_to_publish = []
        steps_to_queue = []
        check_completion_needed = False  # 🔧 新增标记
        deadlock_event = None  # 🔧 新增标记

        async with task_lock:
            proj = self.store.get_projection(event.task_id)
            if not proj or proj.is_terminal:
                return

            # === DAG_UPDATED 事件处理 ===
            if event.event_type == EventType.DAG_UPDATED:
                # 检查是否有新的就绪步骤
                proj._check_ready_steps()
                # 将新就绪的步骤加入队列
                for step_id in list(proj.ready_pool):  # Use list to avoid modification during iteration
                    step = proj.steps.get(step_id)
                    if step and step_id not in proj.completed and step_id not in proj.running:
                        deps = set(step.dependencies)
                        if deps.issubset(proj.completed):
                            async with self._state_lock:
                                if step_id not in self._pending_steps:
                                    self._pending_steps.add(step_id)
                                    steps_to_queue.append((event.task_id, step_id))
                                    logger.info(f"[Scheduler] ✅ New step {step_id[:8]} queued after DAG update")
                # 🔧 修复：清空 ready_pool，避免重复添加相同步骤
                proj.ready_pool.clear()
                return

            if event.event_type == EventType.STEP_READY:
                step_id = event.step_id
                async with self._state_lock:
                    if step_id in self._pending_steps:
                        logger.debug(f"[Scheduler] Step {step_id[:8]} already pending, skipping")
                        return
                    self._pending_steps.add(step_id)
                # 🔧 修复：简化逻辑，直接入队，避免重复处理
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
                    events_to_publish.append(ready_event)
                    steps_to_queue.append((event.task_id, step.step_id))
                    logger.info(f"[Scheduler] ✅ Prepared dispatch step {step.step_id[:8]} ({step.step_type})")
            # 🔧 修复：移除过于激进的完成检查，改为合理的死锁检测
            else:
                proj = self.store.get_projection(event.task_id)
                if proj and not proj.is_terminal:
                    # 只有当有未完成步骤且没有活跃步骤时才认为是死锁
                    if proj.check_deadlock(min_age_seconds=2.0):
                        logger.error(f"[Scheduler] Deadlock detected for {event.task_id[:8]}")
                        deadlock_event = Event(
                            task_id=event.task_id,
                            event_type=EventType.TASK_FAILED,
                            payload={"reason": "deadlock"}
                        )
                        events_to_publish.append(deadlock_event)

        for evt in events_to_publish:
            try:
                await self.store.append_and_publish(evt)
            except Exception as e:
                logger.error(f"[Scheduler] Failed to publish event: {e}")
                if evt.step_id:
                    async with self._state_lock:
                        self._pending_steps.discard(evt.step_id)

        for item in steps_to_queue:
            await self.queue.put(item)
            logger.info(f"[Scheduler] ✅ Queued step {item[1][:8]}")

    def register_completion(self, task_id: str, future: asyncio.Future):
        self._completion_futures[task_id] = future

    async def shutdown(self):
        self._stop_event.set()
        if self._scan_task:
            self._scan_task.cancel()
            await asyncio.sleep(0)

# =========================
# Worker
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
            timeout=60.0  # 🔧 降低超时时间
        )
        # 🔧 新增：Tavily 搜索工具
        self.search_tool = TavilySearchTool()
        self._running = True
        self._stop_event = asyncio.Event()

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
                    # 🔧 修复：确保 task_done() 只在有效获取时调用
                    try:
                        self.queue.task_done()
                    except ValueError:
                        pass  # 队列已空，忽略
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
            # 🔧 修复：移除全局信号量限制，允许真正的并发执行
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

            if proj is not None and retries < max_retries:
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
        step_type = step.step_type

        # 🔧 如果是 search 步骤，直接调用 Tavily 搜索工具
        if step_type == "search":
            return await self._execute_search(step, proj)

        # 其他步骤使用 LLM 处理
        return await self._execute_llm_think(step, proj)

    async def _execute_search(self, step: StepDefine, proj: TaskProjection) -> Dict:
        """执行 Tavily 搜索"""
        task_text = proj.task_text
        current_date = time.strftime('%Y 年%m 月%d 日', time.localtime())

        logger.info(f"[{self.wid}] 🔍 执行搜索：{task_text[:50]}...")
        start_time = time.time()

        try:
            # 调用 Tavily 搜索
            results = await self.search_tool.search(
                query=task_text,
                max_results=5,
                search_depth="basic",  # 🔧 使用 basic 模式，更快
                include_answer=True
            )

            search_time = time.time() - start_time
            logger.info(f"[{self.wid}] ✅ 搜索完成，耗时{search_time:.1f}s，找到{len(results.get('results', []))}条结果")

            # 格式化搜索结果
            formatted_results = self.search_tool.format_results(results)

            # 🔧 检查是否还有未完成的后续步骤
            all_next_steps = [
                s.step_type for s in proj.steps.values()
                if s.step_id not in proj.completed and s.step_id not in proj.running and s.step_id != step.step_id
            ]
            has_next_steps = any(
                s_type in ["research", "think", "analyze", "answer", "summarize"]
                for s_type in all_next_steps
            )

            return {
                "output": f"【搜索时间】{current_date}\n\n{formatted_results}",
                "next_step": "analyze" if has_next_steps else "done",
                "is_terminal": not has_next_steps
            }
        except Exception as e:
            logger.error(f"[{self.wid}] ❌ 搜索失败：{e}")
            return {
                "output": f"搜索失败：{str(e)}",
                "next_step": "analyze",
                "is_terminal": False
            }

    async def _execute_llm_think(self, step: StepDefine, proj: TaskProjection) -> Dict:
        """使用 LLM 处理非搜索步骤"""
        dep_results = {}  # 🔧 修复：初始化 dep_results
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

        # 🔧 新增：获取步骤信息
        input_data = step.input_data
        step_type = step.step_type  # 🔧 修复：添加 step_type 定义
        step_index = input_data.get("step_index", 0)
        total_steps = input_data.get("total_steps", 1)
        is_last_step = input_data.get("is_last_step", False)

        # 🔧 新增：获取当前日期，作为重要信息添加到提示词中
        current_date = time.strftime('%Y年%m月%d日', time.localtime())

        # 根据步骤类型设置提示词
        if step_type == "search":
            stage_instruction = f"执行网络搜索，收集与任务相关的最新信息。【重要】当前日期是{current_date}，请结合此日期进行搜索，确保获取最新的市场数据和信息。输出搜索结果摘要，至少 150 字。"
        elif step_type == "research":
            stage_instruction = f"深入研究分析，结合已有信息提出见解。【重要】当前日期是{current_date}，请确保分析基于最新信息。输出不少于 200 字。"
        elif step_type == "think":
            stage_instruction = f"思考整合所有信息，形成最终观点。【重要】当前日期是{current_date}。输出不少于 200 字。"
        elif step_type == "analyze":
            stage_instruction = f"详细分析，提供专业见解。【重要】当前日期是{current_date}。输出不少于 200 字。"
        elif step_type == "answer":
            stage_instruction = f"直接给出最终答案，清晰明了。【重要】当前日期是{current_date}。"
        elif step_type == "summarize":
            stage_instruction = f"总结关键信息，形成简洁结论。【重要】当前日期是{current_date}。"
        else:
            stage_instruction = f"根据阶段类型 {step_type} 提供详细分析或计划，至少 150 字。【重要】当前日期是{current_date}。"

        # 🔧 关键：如果是最后一步，直接设置为终端步骤
        if is_last_step:
            stage_instruction += "\n\n【重要】这是最后一步，请给出最终结论，设置 is_terminal 为 true。"

        # 获取当前日期
        current_date = time.strftime('%Y年%m月%d日', time.localtime())

        prompt = f"""你是一个高级金融分析师。当前日期：{current_date}
当前任务：{proj.task_text}
当前步骤：{step.step_type}（第{step_index + 1}/{total_steps}步）
【上下文信息】
{full_context}
【执行指令】
{stage_instruction}
【响应格式】
必须返回如下 JSON 格式，不要包含任何开场白：
{{
    "output": "此处填写详细的分析内容或计划路线",
    "is_terminal": {str(is_last_step).lower()}
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

            if isinstance(data, list):
                data = {"output": json.dumps(data), "is_terminal": True}
            elif not isinstance(data, dict):
                data = {"output": str(data), "is_terminal": True}

            if not data.get("output") or str(data["output"]).strip() == "":
                data["output"] = content if content.strip() else "模型未生成有效内容"

            # 确保返回值包含必需字段
            data.setdefault("is_terminal", False)
            data.setdefault("next_step", "done")
            data.setdefault("next_steps", [])

            # 🔧 确保最后一步是终端步骤
            if is_last_step:
                data["is_terminal"] = True

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

        is_terminal = result.get("is_terminal", False)

        # 🔧 检查 DAG 中是否还有未完成的步骤
        has_pending_steps = any(
            sid not in proj.completed and sid not in proj.failed_steps
            for sid in proj.steps.keys()
        )

        if not next_steps or next_steps == ["done"] or is_terminal:
            # 🔧 修复：只有当没有未完成步骤时才设置 is_terminal
            should_complete = (not has_pending_steps) or is_terminal

            # 🔧 修复：检查是否有未完成的后续步骤（已存在但尚未执行）
            if not next_steps or next_steps == ["done"]:
                # 检查是否还有未完成的步骤
                for sid, step in proj.steps.items():
                    if sid not in proj.completed and sid not in proj.running and sid != step_id and step.step_type != "search":
                        should_complete = False
                        break

            # 🔧 修复：发布 STEP_COMPLETED 事件
            await self.store.append_and_publish(Event(
                task_id=task_id,
                step_id=step_id,
                event_type=EventType.STEP_COMPLETED,
                payload={
                    "output": result.get("output", ""),
                    "next_step": "done",
                    "is_terminal": should_complete,
                    "delta": result.get("delta", {})
                }
            ))
            # 🔧 修复：如果 is_terminal 为 True，发布 TASK_COMPLETED 事件
            if should_complete:
                logger.info(f"[{self.wid}] 🏁 Publishing TASK_COMPLETED for task {task_id[:8]}")
                await self.store.append_and_publish(Event(
                    task_id=task_id,
                    event_type=EventType.TASK_COMPLETED,
                    payload={"reason": "completed"}
                ))
            return

        # 🔧 修复：检查是否已有未完成的后续步骤，如果有，就不创建新步骤
        existing_unfinished_steps = []
        for next_step_type in next_steps:
            if next_step_type == "done":
                continue
        new_steps = []
        created_types = set()
        for i, next_step_type in enumerate(next_steps):
            if next_step_type == "done":
                continue
            # 🔧 跳过已有未完成步骤的类型
            if next_step_type in created_types:
                continue
            # 🔧 如果已有未完成的后续步骤，跳过创建
            if next_step_type in [s.step_type for s in proj.steps.values() if s.step_id not in proj.completed and s.step_id not in proj.running]:
                continue
            created_types.add(next_step_type)
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
            # 🔧 诊断日志
            step_details = [f"{s['step_id'][:8]}({s['step_type']})" for s in new_steps]
            logger.info(f"[{self.wid}] New steps details: {step_details}")
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
# 主控系统 (🔧 修复退出慢 + KeyError)
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

    async def _parse_user_intent(self, task: str) -> Dict:
        """
        🔧 新增：让 LLM 解析用户意图，自动判断工作流
        用户可以说："先搜索、再研究、然后思考出回答"
        LLM 会返回：{"workflow": ["search", "research", "think"], "reasoning": "..."}
        """
        # 🔧 先做关键词匹配，确保搜索请求被正确识别
        search_keywords = ["搜索", "查找", "查询", "最新", "实时", "网上", "网络", "browse", "search"]
        only_search_keywords = ["只搜索", "仅搜索", "只要搜索", "只查找", "仅查找"]

        # 🔧 调试日志
        logger.info(f"[Intent Parse] 收到任务：'{task}'")

        # 检查是否只搜索
        for kw in only_search_keywords:
            matched = kw in task
            if matched:
                logger.info(f"[Intent Parse] ✅ 关键词匹配：只搜索 ({kw})")
                return {
                    "has_explicit_workflow": True,
                    "workflow": ["search"],
                    "core_task": task,
                    "reasoning": f"关键词匹配：只搜索 ({kw})"
                }

        # 检查是否包含多个关键词，确定工作流
        found_keywords = []
        for kw in search_keywords:
            if kw in task:
                found_keywords.append("search")
                break

        # 检查分析相关关键词
        analyze_keywords = ["分析", "研究", "解读", "探讨", "讨论", "回答", "答案", "给出"]
        for kw in analyze_keywords:
            if kw in task:
                if "analyze" not in found_keywords:
                    found_keywords.append("analyze")
                break

        # 检查搜索+分析的组合
        if "search" in found_keywords and "analyze" in found_keywords:
            logger.info(f"[Intent Parse] ✅ 关键词匹配：搜索+分析组合")
            return {
                "has_explicit_workflow": True,
                "workflow": ["search", "analyze"],
                "core_task": task,
                "reasoning": "关键词匹配：搜索+分析组合"
            }

        # 只有搜索
        if "search" in found_keywords:
            logger.info(f"[Intent Parse] ✅ 关键词匹配：搜索")
            return {
                "has_explicit_workflow": True,
                "workflow": ["search"],
                "core_task": task,
                "reasoning": "关键词匹配：搜索"
            }

        # 只有分析
        if "analyze" in found_keywords:
            logger.info(f"[Intent Parse] ✅ 关键词匹配：分析")
            return {
                "has_explicit_workflow": True,
                "workflow": ["analyze"],
                "core_task": task,
                "reasoning": "关键词匹配：分析"
            }

        logger.info(f"[Intent Parse] ❌ 未匹配到搜索关键词，使用 LLM 解析")

        # 获取当前日期
        current_date = datetime.now().strftime("%Y年%m月%d日")

        # 否则用 LLM 解析
        prompt = f"""你是一个任务规划助手。请分析用户的任务描述，提取出用户想要的执行步骤顺序。

当前日期：{current_date}

用户任务：{task}

【重要判断规则】
1. 如果用户提到"搜索"、"查找"、"查询"、"最新"、"实时"等词 → 必须包含 "search" 步骤
2. 如果用户提到"分析"、"思考"、"解读"等词 → 使用 "think" 或 "analyze" 步骤
3. 如果用户要求"只搜索"或"仅搜索" → workflow 只包含 ["search"]

请分析用户是否明确指定了执行步骤（如"先搜索再回答"、"直接给出答案"等），然后返回 JSON 格式：
{{
    "has_explicit_workflow": true/false,  // 用户是否明确指定了工作流
    "workflow": ["step1", "step2", ...], // 推荐的步骤顺序（如果用户没指定，则由你推荐）
    "core_task": "提取出的核心任务描述",  // 去掉工作流指示后的核心任务
    "reasoning": "你的分析理由"
}}

可选的步骤类型：
- search: 网络搜索，获取最新信息
- research: 深入研究分析
- think: 思考整合
- analyze: 详细分析
- answer: 直接给出答案
- summarize: 总结

示例 1：用户说"先搜索一下，然后给出答案" → {{"has_explicit_workflow": true, "workflow": ["search", "answer"], ...}}
示例 2：用户说"只搜索 AH 股溢价" → {{"has_explicit_workflow": true, "workflow": ["search"], ...}}
示例 3：用户说"分析 AH 股溢价" → {{"has_explicit_workflow": false, "workflow": ["think"], ...}}
示例 4：用户说"先搜索、再研究、然后思考出回答" → {{"has_explicit_workflow": true, "workflow": ["search", "research", "think"], ...}}

注意：如果用户提到了"搜索"，workflow 中必须包含"search"。"""

        try:
            llm = AsyncOpenAI(base_url=f"{VLLM_URL}/v1", api_key="sk-ignore", timeout=30.0)
            resp = await llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            content = resp.choices[0].message.content

            # 解析 JSON
            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                return data
        except Exception as e:
            logger.warning(f"[Intent Parse] Failed: {e}, using default workflow")

        # 默认：只有 think 步骤
        return {
            "has_explicit_workflow": False,
            "workflow": ["think"],
            "core_task": task,
            "reasoning": "使用默认工作流"
        }

    def _create_steps_from_workflow(self, task: str, workflow: List[str], task_id: str) -> List[Dict]:
        """
        🔧 新增：根据工作流创建步骤链
        """
        steps = []
        prev_step_id = None

        for i, step_type in enumerate(workflow):
            step_id = f"step-{uuid.uuid4().hex[:8]}"
            dependencies = [prev_step_id] if prev_step_id else []

            input_data = {"task": task}
            input_data["step_index"] = i
            input_data["total_steps"] = len(workflow)
            input_data["is_last_step"] = (i == len(workflow) - 1)

            steps.append({
                "step_id": step_id,
                "step_type": step_type,
                "input_data": input_data,
                "dependencies": dependencies,
                "max_retries": 3,
                "version": 1
            })
            prev_step_id = step_id

        return steps

    async def run(self, task: str, timeout: float = 300.0) -> Dict:
        task_id = str(uuid.uuid4())

        # 🔧 新增：先让 LLM 解析用户意图
        print("🤖 正在解析用户意图...")
        intent = await self._parse_user_intent(task)
        workflow = intent.get("workflow", ["think"])
        core_task = intent.get("core_task", task)
        print(f"   工作流：{' → '.join(workflow)}")
        print(f"   核心任务：{core_task[:50]}...")

        print(f"\n{'='*70}")
        print(f"🚀 任务启动")
        print(f"{'='*70}")
        print(f"任务 ID: {task_id[:8]}")
        print(f"任务描述：{core_task[:60]}...")
        print(f"工作流：{' → '.join(workflow)}")
        print(f"开始时间：{time.strftime('%H:%M:%S')}")
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

        # 🔧 新增：根据解析的工作流创建步骤
        steps_data = self._create_steps_from_workflow(core_task, workflow, task_id)
        logger.info(f"[Workflow] Created {len(steps_data)} steps: {[s['step_type'] for s in steps_data]}")

        try:
            await self.store.append_and_publish(Event(
                task_id=task_id,
                event_type=EventType.TASK_SUBMITTED,
                payload={
                    "task": core_task,
                    "dag": steps_data
                }
            ))
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            for t in worker_tasks:
                t.cancel()
            if self._global_timeout_task:
                self._global_timeout_task.cancel()
            return {"task_id": task_id, "status": "submit_failed", "error": str(e), "completed_steps": 0, "total_steps": 0, "duration": 0}

        try:
            result_status = await asyncio.wait_for(completion_future, timeout=timeout + 5)
            proj = self.store.get_projection(task_id)
            result = {
                "task_id": task_id,
                "status": result_status.value if hasattr(result_status, 'value') else result_status,
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
            # 🔧 修复：超时返回也包含 completed_steps 和 total_steps
            return {
                "task_id": task_id,
                "status": "timeout",
                "completed_steps": len(proj.completed) if proj else 0,
                "total_steps": len(proj.steps) if proj else 0,
                "duration": timeout
            }
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

    # 🔧 示例 1：用户指定工作流
    task = "先搜索一下 AH 股溢价，然后分析现在的市场特征"

    # 🔧 示例 2：用户要求只搜索
    # task = "只搜索 AH 股溢价相关信息"

    # 🔧 示例 3：默认模式（让系统自动决定）
    # task = "分析 AH 股溢价对当前市场情绪的影响"

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
