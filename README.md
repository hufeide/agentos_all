```markdown
# Agent Run 流程说明

`agent.run("任务")` 的执行流程如下：

```
用户调用 agent.run("任务")
        │
        ▼
┌─────────────────┐
│ 1. 初始化组件    │
│ - LLM, Search   │
│ - Planner, Engine│
│ - Workers (N个) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Planner.plan()│
│ - 调用 LLM       │
│ - 生成 DAG 计划  │
│ - 检测循环依赖   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Engine.set_plan()│
│ - 注入计划       │
│ - 清空已发布集合 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Engine.start()│
│ - 调用 _publish_ready_steps()│
│ - 发布初始 STEP_READY 事件 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ 5. EventBus 广播 │────▶│ Worker.on_step_ready()│
│ STEP_READY      │     │ - claim_step()   │
└─────────────────┘     │ - execute()      │
                        │ - 发布完成/失败事件│
                        └────────┬────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌─────────────────┐
│ 6. Engine 监听   │◀────│ EventBus 广播    │
│ process_completed│    │ STEP_COMPLETED  │
│ - 更新状态       │    │ STEP_FAILED     │
│ - 调用 _publish_ready_steps()│
│ - 检查任务完成   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 7. 发布 TASK_COMPLETED│
│ - 触发 _completion_event│
│ - _wait_for_completion 返回│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 8. 清理资源      │
│ - close()       │
│ - bus.shutdown()│
└─────────────────┘
```

## 步骤说明

1. **初始化组件**：创建 LLM、Search、Planner、Engine 和 N 个 Worker 实例。
2. **生成计划**：Planner 调用 LLM 生成 DAG（有向无环图）计划，并检测循环依赖。
3. **注入计划**：Engine 加载该计划，并清空已发布步骤集合。
4. **启动引擎**：Engine 开始执行，发布初始的 `STEP_READY` 事件。
5. **事件分发与执行**：EventBus 将 `STEP_READY` 广播给 Worker，Worker 认领步骤并执行，完成后发布 `STEP_COMPLETED` 或 `STEP_FAILED` 事件。
6. **引擎处理完成**：Engine 监听步骤完成事件，更新状态，发布新的就绪步骤，并检查整个任务是否完成。
7. **任务完成**：Engine 发布 `TASK_COMPLETED` 事件，触发等待的完成事件，返回结果。
8. **清理资源**：调用 `close()` 和 `bus.shutdown()` 释放资源。

该流程基于事件驱动模型，实现了步骤的并行执行和依赖管理。
```
