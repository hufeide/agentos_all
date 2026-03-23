# Agent Run 执行流程

`agent.run("任务")` 的执行流程如下：

```mermaid
graph TD
    A[用户调用 agent.run] --> B[1. 初始化组件<br/>LLM, Search, Planner, Engine, Workers]
    B --> C[2. Planner.plan<br/>调用 LLM 生成 DAG 计划<br/>检测循环依赖]
    C --> D[3. Engine.set_plan<br/>注入计划，清空已发布集合]
    D --> E[4. Engine.start<br/>调用 _publish_ready_steps<br/>发布初始 STEP_READY 事件]
    E --> F[EventBus 广播 STEP_READY]
    F --> G[Worker.on_step_ready<br/>claim_step → execute<br/>发布完成/失败事件]
    G --> H[EventBus 广播 STEP_COMPLETED / STEP_FAILED]
    H --> I[Engine 监听 process_completed<br/>更新状态，_publish_ready_steps<br/>检查任务完成]
    I --> J[发布 TASK_COMPLETED<br/>触发 _completion_event<br/>_wait_for_completion 返回]
    J --> K[清理资源<br/>close, bus.shutdown]
