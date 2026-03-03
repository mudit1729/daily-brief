# Async Orchestration & Queue Management — Theory Preparation

> **Interview Focus**: Async orchestration and queue management in Python. Reasoning about how tasks and stages are executed, timed, and coordinated — not syntax trivia.

---

## 1. The Core Mental Model

The interview describes a system where:

- **Tasks** are units of work, each composed of multiple **stages**.
- Stages within a single task must run **sequentially** (stage 2 depends on stage 1's output).
- Multiple tasks should run in **parallel** (task A and task B are independent).
- A **queue-like structure** holds items representing task stages.
- Some interfaces and data structures are **fixed** — you must work within constraints.

This is fundamentally a **producer-consumer pipeline with per-task sequential ordering and cross-task parallelism**.

```
Task A: [Stage 1] → [Stage 2] → [Stage 3]    ← sequential
Task B: [Stage 1] → [Stage 2]                 ← sequential
Task C: [Stage 1] → [Stage 2] → [Stage 3]    ← sequential
         ↑            ↑            ↑
         └────────────┴────────────┘
              all tasks run in parallel
```

---

## 2. Python `asyncio` Foundations

### 2.1 The Event Loop

The event loop is the central execution engine. It runs coroutines, handles I/O callbacks, and manages the scheduling of tasks. In Python 3.10+, `asyncio.run()` creates and manages the loop for you.

```python
import asyncio

async def main():
    await do_work()

asyncio.run(main())
```

Key insight for interviews: the event loop is **single-threaded**. Concurrency comes from **cooperative multitasking** — coroutines yield control at `await` points, letting other coroutines run.

### 2.2 Coroutines, Tasks, and Futures

| Concept | What It Is | When to Use |
|---------|-----------|-------------|
| **Coroutine** | An `async def` function. Not running until awaited or wrapped in a Task. | Defining async work |
| **Task** | A scheduled coroutine (`asyncio.create_task()`). Starts running immediately. | When you want concurrency |
| **Future** | A low-level awaitable representing an eventual result. | Rarely used directly |

```python
async def fetch(url):
    await asyncio.sleep(1)  # simulate I/O
    return f"data from {url}"

async def main():
    # Sequential — total time ~3s
    r1 = await fetch("a")
    r2 = await fetch("b")
    r3 = await fetch("c")

    # Parallel — total time ~1s
    r1, r2, r3 = await asyncio.gather(
        fetch("a"), fetch("b"), fetch("c")
    )
```

### 2.3 `asyncio.gather` vs `asyncio.TaskGroup` vs `asyncio.wait`

#### `asyncio.gather(*coros, return_exceptions=False)`

- Runs all coroutines concurrently.
- Returns results **in the same order** as input.
- If `return_exceptions=False` (default): first exception propagates, **but other tasks keep running** (this is a gotcha).
- If `return_exceptions=True`: exceptions are returned as values in the result list.

```python
results = await asyncio.gather(
    task_a(), task_b(), task_c(),
    return_exceptions=True
)
# results might be: ["ok", ValueError("oops"), "ok"]
```

#### `asyncio.TaskGroup` (Python 3.11+)

- **Structured concurrency**: all tasks are guaranteed to finish before the `async with` block exits.
- If any task raises, **all remaining tasks are cancelled** automatically.
- Raises `ExceptionGroup` collecting all failures.
- **Preferred in modern Python** — safer, faster, and cleaner.

```python
async with asyncio.TaskGroup() as tg:
    t1 = tg.create_task(task_a())
    t2 = tg.create_task(task_b())
    t3 = tg.create_task(task_c())
# All done here. If any failed, ExceptionGroup is raised.
```

#### `asyncio.wait(tasks, return_when=...)`

- Returns `(done, pending)` sets.
- `return_when` options: `FIRST_COMPLETED`, `FIRST_EXCEPTION`, `ALL_COMPLETED`.
- Gives you the most control but requires more manual management.

```python
done, pending = await asyncio.wait(
    tasks, return_when=asyncio.FIRST_COMPLETED
)
```

**Interview tip**: Know when to use each. `TaskGroup` is the right default for this kind of problem because of its structured concurrency guarantees.

---

## 3. Queue-Based Patterns in `asyncio`

### 3.1 `asyncio.Queue`

This is the backbone of producer-consumer patterns in async Python.

```python
queue = asyncio.Queue(maxsize=0)  # 0 = unbounded

# Producer
await queue.put(item)

# Consumer
item = await queue.get()
# ... process item ...
queue.task_done()

# Wait for all items to be processed
await queue.join()
```

Key methods:

| Method | Behavior |
|--------|----------|
| `put(item)` | Add item. Blocks if queue is full (when maxsize > 0). |
| `get()` | Remove and return item. Blocks if queue is empty. |
| `task_done()` | Signal that a formerly enqueued item is fully processed. |
| `join()` | Block until all items have been processed (put count == task_done count). |
| `qsize()` | Current number of items in queue. |
| `empty()` | Returns `True` if queue is empty. |

### 3.2 Queue Variants

| Class | Behavior |
|-------|----------|
| `asyncio.Queue` | FIFO — first in, first out |
| `asyncio.PriorityQueue` | Lowest priority value comes out first |
| `asyncio.LifoQueue` | Last in, first out (stack) |

### 3.3 The Producer-Consumer Pattern

```python
async def producer(queue, items):
    for item in items:
        await queue.put(item)

async def consumer(queue, worker_id):
    while True:
        item = await queue.get()
        try:
            await process(item)
        finally:
            queue.task_done()

async def main():
    queue = asyncio.Queue()

    # Start consumers
    consumers = [
        asyncio.create_task(consumer(queue, i))
        for i in range(3)
    ]

    # Produce work
    await producer(queue, work_items)

    # Wait for all work to complete
    await queue.join()

    # Cancel consumers (they're in infinite loops)
    for c in consumers:
        c.cancel()
```

### 3.4 Multi-Stage Pipeline with Queues

This is the pattern most likely to appear in the interview:

```python
async def stage_worker(input_queue, output_queue, process_fn):
    while True:
        item = await input_queue.get()
        try:
            result = await process_fn(item)
            if output_queue:
                await output_queue.put(result)
        finally:
            input_queue.task_done()

async def pipeline():
    q1 = asyncio.Queue()
    q2 = asyncio.Queue()
    q3 = asyncio.Queue()

    # Wire up stages
    workers = [
        asyncio.create_task(stage_worker(q1, q2, stage_1_fn)),
        asyncio.create_task(stage_worker(q2, q3, stage_2_fn)),
        asyncio.create_task(stage_worker(q3, None, stage_3_fn)),
    ]

    # Feed work
    for item in work_items:
        await q1.put(item)

    await q1.join()
    await q2.join()
    await q3.join()
```

---

## 4. Orchestration: Parallel Tasks with Sequential Stages

This is the **core pattern** the interview is testing. Each task has ordered stages, but tasks themselves are independent.

### 4.1 The Naive (Wrong) Approach

Running everything sequentially:

```python
# BAD — no parallelism between tasks
for task in tasks:
    for stage in task.stages:
        await stage.run()
```

### 4.2 The Correct Pattern

Launch each task as a concurrent coroutine. Within each task, stages run sequentially.

```python
async def run_task(task):
    """Stages within a task run sequentially."""
    for stage in task.stages:
        result = await stage.run()
        # result may feed into next stage

async def orchestrate(tasks):
    """Tasks run in parallel."""
    await asyncio.gather(*[run_task(t) for t in tasks])
```

### 4.3 With a Queue-Based Interface

If stages are items in a queue (as described in the interview prompt), you need to maintain per-task ordering while processing the queue concurrently:

```python
async def orchestrate(queue_items):
    # Group by task
    tasks = defaultdict(list)
    for item in queue_items:
        tasks[item.task_id].append(item)

    async def run_task_stages(stages):
        for stage in sorted(stages, key=lambda s: s.order):
            await execute_stage(stage)

    # All tasks in parallel, stages within each task sequential
    await asyncio.gather(*[
        run_task_stages(stages)
        for stages in tasks.values()
    ])
```

---

## 5. Timing and Observability

The interview mentions measuring:

- **How long a stage takes to run** (execution duration).
- **How long it waited since the queue started** (queue wait time / latency).

### 5.1 Measuring Execution Time

Use `time.monotonic()` — not `time.time()`. Monotonic clocks aren't affected by system clock adjustments.

```python
import time

async def timed_execute(stage):
    start = time.monotonic()
    result = await stage.run()
    elapsed = time.monotonic() - start
    log(f"Stage {stage.id} took {elapsed:.4f}s")
    return result
```

### 5.2 Measuring Queue Wait Time

Record the time when an item enters the queue and when it starts processing:

```python
@dataclass
class QueueItem:
    stage: Stage
    enqueued_at: float  # time.monotonic()

# When enqueuing
item = QueueItem(stage=s, enqueued_at=time.monotonic())
await queue.put(item)

# When dequeuing
item = await queue.get()
wait_time = time.monotonic() - item.enqueued_at
log(f"Stage waited {wait_time:.4f}s in queue")
```

### 5.3 Why `time.monotonic()`?

| Clock | Monotonic? | Affected by NTP? | Use Case |
|-------|-----------|-------------------|----------|
| `time.time()` | No | Yes | Wall-clock timestamps |
| `time.monotonic()` | Yes | No | **Duration measurement** |
| `time.perf_counter()` | Yes | No | High-precision benchmarking |

For the interview, use `time.monotonic()` and mention why: precision and consistency across clock adjustments.

### 5.4 Structured Logging for Async Systems

```python
import logging

logger = logging.getLogger(__name__)

async def execute_with_logging(stage, queue_start_time):
    wait_time = time.monotonic() - stage.enqueued_at
    logger.info(
        "stage_start",
        extra={
            "task_id": stage.task_id,
            "stage_id": stage.stage_id,
            "wait_time_s": f"{wait_time:.4f}",
            "queue_elapsed_s": f"{time.monotonic() - queue_start_time:.4f}",
        }
    )

    start = time.monotonic()
    result = await stage.run()
    duration = time.monotonic() - start

    logger.info(
        "stage_complete",
        extra={
            "task_id": stage.task_id,
            "stage_id": stage.stage_id,
            "duration_s": f"{duration:.4f}",
        }
    )
    return result
```

---

## 6. Error Handling and Partial Retries

### 6.1 Error Handling Philosophy

The interview says: "Some stages may fail. Think about partial retries. You don't need a perfect production system, but you should show good judgment."

Key principles:

1. **Isolate failures**: One task's failure shouldn't crash others.
2. **Retry transient errors**: Network timeouts, rate limits — retry with backoff.
3. **Fail fast on permanent errors**: Bad input, missing dependencies — don't retry.
4. **Report partial results**: If 8 of 10 tasks succeed, return the 8 results plus error info.

### 6.2 Basic Retry with Exponential Backoff

```python
async def retry_stage(stage, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries + 1):
        try:
            return await stage.run()
        except TransientError as e:
            if attempt == max_retries:
                raise  # give up
            delay = base_delay * (2 ** attempt)  # 1s, 2s, 4s
            logger.warning(
                f"Stage {stage.id} failed (attempt {attempt + 1}), "
                f"retrying in {delay}s: {e}"
            )
            await asyncio.sleep(delay)
        except PermanentError:
            raise  # don't retry
```

### 6.3 Partial Failure Handling with `gather`

```python
async def run_all_tasks(tasks):
    results = await asyncio.gather(
        *[run_task(t) for t in tasks],
        return_exceptions=True
    )

    successes = []
    failures = []
    for task, result in zip(tasks, results):
        if isinstance(result, Exception):
            failures.append((task, result))
        else:
            successes.append((task, result))

    return successes, failures
```

### 6.4 Retry Within a Task's Stage Sequence

If stage 2 of task A fails, should you retry just stage 2, or re-run from stage 1?

**The right answer depends on whether stages are idempotent.**

```python
async def run_task_with_retry(task, max_retries=2):
    for stage in task.stages:
        retries = 0
        while True:
            try:
                await stage.run()
                break  # success, move to next stage
            except RetryableError:
                retries += 1
                if retries > max_retries:
                    raise TaskFailedError(
                        f"Task {task.id} failed at stage {stage.id} "
                        f"after {max_retries} retries"
                    )
                await asyncio.sleep(2 ** retries)
```

### 6.5 Error Classification

In the interview, talk about how you'd classify errors:

```python
class StageError(Exception):
    """Base error for stage execution."""
    pass

class TransientError(StageError):
    """Retryable: timeouts, rate limits, temporary failures."""
    pass

class PermanentError(StageError):
    """Non-retryable: bad input, missing resource, logic error."""
    pass
```

---

## 7. Concurrency Control

### 7.1 Semaphores

Limit how many stages run concurrently (e.g., don't overwhelm an external API):

```python
sem = asyncio.Semaphore(5)  # max 5 concurrent

async def rate_limited_stage(stage):
    async with sem:
        return await stage.run()
```

### 7.2 Bounded Queues

`asyncio.Queue(maxsize=N)` provides backpressure — producers block when the queue is full, preventing memory from growing unbounded.

```python
queue = asyncio.Queue(maxsize=100)
# put() will block when queue has 100 items
```

### 7.3 Avoiding Race Conditions

In async Python (single-threaded), race conditions are less common than in multithreaded code, but they **still happen** when you have shared mutable state and multiple `await` points between reading and writing:

```python
# RACE CONDITION — another coroutine can run between these lines
balance = await get_balance()
# <--- context switch possible here
await set_balance(balance - amount)

# SAFE — use a lock
lock = asyncio.Lock()
async with lock:
    balance = await get_balance()
    await set_balance(balance - amount)
```

---

## 8. Patterns You Should Know Cold

### 8.1 Fan-Out / Fan-In

Distribute work across workers, collect results:

```python
async def fan_out_fan_in(items, worker_fn, max_workers=5):
    queue = asyncio.Queue()
    results = []

    for item in items:
        await queue.put(item)

    async def worker():
        while not queue.empty():
            item = await queue.get()
            result = await worker_fn(item)
            results.append(result)
            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(max_workers)]
    await queue.join()
    return results
```

### 8.2 Timeout Wrapper

```python
async def with_timeout(coro, timeout_s):
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except asyncio.TimeoutError:
        raise StageTimeoutError(f"Stage timed out after {timeout_s}s")
```

### 8.3 Graceful Shutdown

```python
async def shutdown(tasks):
    for task in tasks:
        task.cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
            logger.error(f"Error during shutdown: {result}")
```

---

## 9. What Scale AI Cares About (Interview Intelligence)

Based on research into Scale AI's interview process:

**System Design at Scale AI** focuses on data pipelines, annotation workflows, task routing infrastructure, and human-in-the-loop operations. The interview is about data quality, task correctness, workflow orchestration, and operational efficiency.

**Common patterns in Scale AI interviews:**

- Task assignment using priority scheduling (min-heap to least-loaded worker).
- Building systems around LLM black-box services where requests are split into segments processed asynchronously.
- Worker state management and task queue with priority scheduling.
- Handling continuous streams of work with quality control checkpoints.

**What they evaluate:**

- Your mental model of async systems — can you draw the execution timeline?
- Whether you reason about concurrency without race conditions.
- How cleanly you structure logic under constraints (fixed interfaces, etc.).
- Your ability to explain decisions as you go — talk through tradeoffs.

---

## 10. Key Vocabulary

| Term | Meaning |
|------|---------|
| **Coroutine** | An `async def` function; cooperative, yields at `await` |
| **Task** | A scheduled coroutine running on the event loop |
| **Structured concurrency** | All child tasks complete before parent exits (`TaskGroup`) |
| **Backpressure** | Slowing producers when consumers can't keep up |
| **Idempotent** | Running the same operation multiple times gives the same result |
| **Fan-out / Fan-in** | Distribute work, then collect results |
| **Exponential backoff** | Doubling wait time between retries: 1s, 2s, 4s, 8s... |
| **Monotonic clock** | A clock that only moves forward (for duration measurement) |
| **Semaphore** | Limits concurrent access to a resource |
| **Deadlock** | Two or more tasks waiting for each other forever |
| **Starvation** | A task never gets to run because others keep taking priority |

---

## Sources

- [Scale AI System Design Interview Guide](https://www.systemdesignhandbook.com/guides/scale-ai-system-design-interview/)
- [Scale AI Coding Interview Questions](https://www.codinginterview.com/guide/scale-ai-coding-interview-questions/)
- [Scale AI Software Engineer Interview Guide — InterviewQuery](https://www.interviewquery.com/interview-guides/scale-software-engineer)
- [Scale AI Interview Process — Exponent](https://www.tryexponent.com/blog/scale-ai-interview-process)
- [Python asyncio.Queue Documentation](https://docs.python.org/3/library/asyncio-queue.html)
- [Python Coroutines and Tasks Documentation](https://docs.python.org/3/library/asyncio-task.html)
- [Real Python: asyncio Walkthrough](https://realpython.com/async-io-python/)
- [asyncio.gather vs TaskGroup — Codemia](https://codemia.io/knowledge-hub/path/asynciogather_vs_asynciowait_vs_asynciotaskgroup)
