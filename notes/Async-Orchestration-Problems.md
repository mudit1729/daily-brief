# Async Orchestration & Queue Management — Practice Problems

> All solutions are in Python using `asyncio`. Problems are ordered from foundational to interview-level difficulty.

---

## Problem 1: Basic — Parallel Tasks, Sequential Stages

**Prompt**: You have a list of tasks. Each task has a name and a list of stages (async functions). Stages within a task must run sequentially. All tasks should run in parallel. Implement the orchestrator.

### Setup

```python
import asyncio
import time

async def simulate_stage(task_name: str, stage_num: int, duration: float):
    """Simulates a stage that takes `duration` seconds."""
    print(f"[{task_name}] Stage {stage_num} started")
    await asyncio.sleep(duration)
    print(f"[{task_name}] Stage {stage_num} completed ({duration}s)")
    return f"{task_name}-stage{stage_num}-result"
```

### Solution

```python
from dataclasses import dataclass, field
from typing import List, Callable, Awaitable, Any

@dataclass
class Stage:
    name: str
    execute: Callable[[], Awaitable[Any]]

@dataclass
class Task:
    name: str
    stages: List[Stage] = field(default_factory=list)

async def run_task(task: Task) -> list:
    """Run all stages of a task sequentially. Return list of results."""
    results = []
    for stage in task.stages:
        result = await stage.execute()
        results.append(result)
    return results

async def orchestrate(tasks: List[Task]) -> dict:
    """Run all tasks in parallel. Return dict mapping task name -> results."""
    async def run_and_label(task):
        return task.name, await run_task(task)

    coros = [run_and_label(t) for t in tasks]
    pairs = await asyncio.gather(*coros)
    return dict(pairs)

# --- Demo ---
async def main():
    tasks = [
        Task("TaskA", [
            Stage("A1", lambda: simulate_stage("TaskA", 1, 1.0)),
            Stage("A2", lambda: simulate_stage("TaskA", 2, 0.5)),
            Stage("A3", lambda: simulate_stage("TaskA", 3, 0.8)),
        ]),
        Task("TaskB", [
            Stage("B1", lambda: simulate_stage("TaskB", 1, 0.7)),
            Stage("B2", lambda: simulate_stage("TaskB", 2, 1.2)),
        ]),
        Task("TaskC", [
            Stage("C1", lambda: simulate_stage("TaskC", 1, 0.3)),
            Stage("C2", lambda: simulate_stage("TaskC", 2, 0.4)),
            Stage("C3", lambda: simulate_stage("TaskC", 3, 0.6)),
        ]),
    ]

    start = time.monotonic()
    results = await orchestrate(tasks)
    elapsed = time.monotonic() - start

    print(f"\nAll tasks completed in {elapsed:.2f}s")
    # Expected: ~2.3s (TaskA is the longest: 1.0 + 0.5 + 0.8)
    # NOT ~4.5s (which would mean sequential execution)
    for name, res in results.items():
        print(f"  {name}: {res}")

asyncio.run(main())
```

**What to talk about**:
- Why `asyncio.gather` and not sequential `await`?
- Total time = max of per-task times, not sum of all stage times.
- Stages within a task ARE sequential (we iterate and `await` each).

---

## Problem 2: Queue-Based Stage Executor with Timing

**Prompt**: Implement a system where stages are enqueued into an `asyncio.Queue`. A pool of workers dequeues stages and executes them. But stages **within the same task must still run in order** — stage 2 can't start before stage 1 finishes. Track and report execution time and queue wait time for each stage.

### Solution

```python
import asyncio
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Callable, Awaitable

@dataclass
class StageItem:
    task_id: str
    stage_index: int
    execute: Callable[[], Awaitable[Any]]
    enqueued_at: float = 0.0

@dataclass
class StageResult:
    task_id: str
    stage_index: int
    result: Any
    wait_time: float
    exec_time: float

class OrderedTaskExecutor:
    """
    Executes stages from a queue. Ensures per-task ordering:
    stage N of a task only runs after stage N-1 completes.
    """

    def __init__(self, num_workers: int = 3):
        self.queue: asyncio.Queue[StageItem] = asyncio.Queue()
        self.num_workers = num_workers
        # Tracks the next stage index each task expects to run
        self._task_next_stage: dict[str, int] = defaultdict(int)
        # Condition variable to signal when a stage completes
        self._stage_done = asyncio.Condition()
        self.results: list[StageResult] = []

    async def enqueue(self, item: StageItem):
        item.enqueued_at = time.monotonic()
        await self.queue.put(item)

    async def _worker(self, worker_id: int):
        while True:
            item = await self.queue.get()
            try:
                # Wait until it's this stage's turn for its task
                async with self._stage_done:
                    while self._task_next_stage[item.task_id] != item.stage_index:
                        # Not our turn yet — put back and wait
                        await self._stage_done.wait()

                # Now it's our turn
                wait_time = time.monotonic() - item.enqueued_at
                exec_start = time.monotonic()

                result = await item.execute()

                exec_time = time.monotonic() - exec_start

                self.results.append(StageResult(
                    task_id=item.task_id,
                    stage_index=item.stage_index,
                    result=result,
                    wait_time=wait_time,
                    exec_time=exec_time,
                ))

                print(
                    f"  [Worker {worker_id}] {item.task_id}/stage{item.stage_index} "
                    f"| wait={wait_time:.3f}s | exec={exec_time:.3f}s"
                )

                # Advance the task's expected stage
                async with self._stage_done:
                    self._task_next_stage[item.task_id] += 1
                    self._stage_done.notify_all()

            finally:
                self.queue.task_done()

    async def run(self):
        workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.num_workers)
        ]
        await self.queue.join()
        for w in workers:
            w.cancel()
        return self.results


# --- Demo ---
async def make_stage(task_id, stage_idx, duration):
    await asyncio.sleep(duration)
    return f"{task_id}-s{stage_idx}-done"

async def main():
    executor = OrderedTaskExecutor(num_workers=4)

    # Define tasks and stages
    stages = [
        StageItem("A", 0, lambda: make_stage("A", 0, 0.5)),
        StageItem("A", 1, lambda: make_stage("A", 1, 0.3)),
        StageItem("A", 2, lambda: make_stage("A", 2, 0.4)),
        StageItem("B", 0, lambda: make_stage("B", 0, 0.6)),
        StageItem("B", 1, lambda: make_stage("B", 1, 0.2)),
        StageItem("C", 0, lambda: make_stage("C", 0, 0.3)),
        StageItem("C", 1, lambda: make_stage("C", 1, 0.5)),
        StageItem("C", 2, lambda: make_stage("C", 2, 0.1)),
    ]

    # Enqueue all stages
    for s in stages:
        await executor.enqueue(s)

    start = time.monotonic()
    results = await executor.run()
    total = time.monotonic() - start

    print(f"\nTotal orchestration time: {total:.3f}s")
    print("\n--- Timing Report ---")
    for r in sorted(results, key=lambda x: (x.task_id, x.stage_index)):
        print(
            f"  {r.task_id}/stage{r.stage_index}: "
            f"wait={r.wait_time:.3f}s, exec={r.exec_time:.3f}s"
        )

asyncio.run(main())
```

**What to talk about**:
- The `Condition` variable prevents out-of-order execution within a task.
- Workers are generic — they don't know about task structure, only the queue.
- Problem: the current approach has an issue — if a worker picks up stage 2 before stage 1 finishes, it holds the worker hostage waiting. Discuss this tradeoff (see Problem 3 for a better design).

---

## Problem 3: Improved Design — Per-Task Coroutines with Shared Worker Pool

**Prompt**: The previous design has a flaw: workers can get blocked waiting for ordering. Design a better system where per-task ordering is handled by dedicated per-task coroutines, while a semaphore-based worker pool controls concurrency.

### Solution

```python
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

@dataclass
class Stage:
    task_id: str
    index: int
    execute: Callable[[], Awaitable[Any]]

@dataclass
class Task:
    task_id: str
    stages: list[Stage] = field(default_factory=list)

@dataclass
class TimingRecord:
    task_id: str
    stage_index: int
    queued_at: float
    started_at: float
    finished_at: float

    @property
    def wait_time(self) -> float:
        return self.started_at - self.queued_at

    @property
    def exec_time(self) -> float:
        return self.finished_at - self.started_at

class PipelineOrchestrator:
    def __init__(self, max_concurrency: int = 4):
        self._sem = asyncio.Semaphore(max_concurrency)
        self._timings: list[TimingRecord] = []
        self._start_time: float = 0.0

    async def _run_stage(self, stage: Stage, queued_at: float) -> Any:
        """Acquire a concurrency slot, then execute the stage."""
        async with self._sem:
            started_at = time.monotonic()
            result = await stage.execute()
            finished_at = time.monotonic()

            record = TimingRecord(
                task_id=stage.task_id,
                stage_index=stage.index,
                queued_at=queued_at,
                started_at=started_at,
                finished_at=finished_at,
            )
            self._timings.append(record)

            print(
                f"  [{stage.task_id}] stage {stage.index}: "
                f"wait={record.wait_time:.3f}s, exec={record.exec_time:.3f}s"
            )
            return result

    async def _run_task(self, task: Task) -> list:
        """Run all stages of one task sequentially."""
        results = []
        for stage in task.stages:
            queued_at = time.monotonic()
            result = await self._run_stage(stage, queued_at)
            results.append(result)
        return results

    async def run(self, tasks: list[Task]) -> dict[str, list]:
        """Run all tasks in parallel, respecting max concurrency."""
        self._start_time = time.monotonic()

        async def labeled(task):
            return task.task_id, await self._run_task(task)

        pairs = await asyncio.gather(*[labeled(t) for t in tasks])
        return dict(pairs)

    def print_report(self):
        total = time.monotonic() - self._start_time
        print(f"\n{'='*50}")
        print(f"ORCHESTRATION REPORT  (total: {total:.3f}s)")
        print(f"{'='*50}")
        for r in sorted(self._timings, key=lambda x: x.started_at):
            offset = r.started_at - self._start_time
            print(
                f"  [{r.task_id}] stage {r.stage_index:>2} | "
                f"@{offset:.3f}s | "
                f"wait={r.wait_time:.3f}s | "
                f"exec={r.exec_time:.3f}s"
            )

# --- Demo ---
async def sim(task_id, idx, dur):
    await asyncio.sleep(dur)
    return f"{task_id}-{idx}"

async def main():
    tasks = [
        Task("A", [
            Stage("A", 0, lambda: sim("A", 0, 0.5)),
            Stage("A", 1, lambda: sim("A", 1, 0.3)),
            Stage("A", 2, lambda: sim("A", 2, 0.4)),
        ]),
        Task("B", [
            Stage("B", 0, lambda: sim("B", 0, 0.6)),
            Stage("B", 1, lambda: sim("B", 1, 0.2)),
        ]),
        Task("C", [
            Stage("C", 0, lambda: sim("C", 0, 0.3)),
            Stage("C", 1, lambda: sim("C", 1, 0.5)),
            Stage("C", 2, lambda: sim("C", 2, 0.1)),
        ]),
    ]

    orch = PipelineOrchestrator(max_concurrency=3)
    results = await orch.run(tasks)
    orch.print_report()

    print("\nResults:")
    for tid, res in results.items():
        print(f"  {tid}: {res}")

asyncio.run(main())
```

**Why this design is better** (talk about this!):
- No worker gets blocked waiting for ordering — per-task coroutines naturally enforce order.
- The semaphore controls concurrency across ALL tasks globally.
- Clean separation: ordering logic vs concurrency control.
- Easy to add timeouts, retries, logging without touching orchestration logic.

---

## Problem 4: Error Handling and Partial Retries

**Prompt**: Extend Problem 3 so that:
1. If a stage fails with a transient error, retry it up to 3 times with exponential backoff.
2. If a stage fails permanently, mark the task as failed but let other tasks continue.
3. Report which tasks succeeded and which failed (and at which stage).

### Solution

```python
import asyncio
import time
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
from enum import Enum

# --- Error types ---
class TransientError(Exception):
    pass

class PermanentError(Exception):
    pass

class TaskStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"

@dataclass
class Stage:
    task_id: str
    index: int
    execute: Callable[[], Awaitable[Any]]

@dataclass
class Task:
    task_id: str
    stages: list[Stage] = field(default_factory=list)

@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    results: list[Any] = field(default_factory=list)
    error: Exception | None = None
    failed_stage: int | None = None

class ResilientOrchestrator:
    def __init__(
        self,
        max_concurrency: int = 4,
        max_retries: int = 3,
        base_delay: float = 0.5,
    ):
        self._sem = asyncio.Semaphore(max_concurrency)
        self._max_retries = max_retries
        self._base_delay = base_delay

    async def _run_stage_with_retry(self, stage: Stage) -> Any:
        """Execute a stage with retry logic for transient errors."""
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                async with self._sem:
                    return await stage.execute()
            except TransientError as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = self._base_delay * (2 ** attempt)
                    print(
                        f"  [{stage.task_id}] stage {stage.index} "
                        f"transient failure (attempt {attempt + 1}), "
                        f"retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    print(
                        f"  [{stage.task_id}] stage {stage.index} "
                        f"exhausted retries after {self._max_retries + 1} attempts"
                    )
                    raise
            except PermanentError:
                print(
                    f"  [{stage.task_id}] stage {stage.index} "
                    f"permanent failure — not retrying"
                )
                raise

    async def _run_task(self, task: Task) -> TaskResult:
        """Run a task's stages sequentially. Handle failures gracefully."""
        results = []
        try:
            for stage in task.stages:
                result = await self._run_stage_with_retry(stage)
                results.append(result)
                print(f"  [{task.task_id}] stage {stage.index} OK")
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.SUCCESS,
                results=results,
            )
        except (TransientError, PermanentError) as e:
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                results=results,  # partial results from completed stages
                error=e,
                failed_stage=len(results),  # index of the stage that failed
            )

    async def run(self, tasks: list[Task]) -> list[TaskResult]:
        """Run all tasks in parallel. Never lets one failure crash others."""
        start = time.monotonic()
        results = await asyncio.gather(*[self._run_task(t) for t in tasks])
        elapsed = time.monotonic() - start

        # Report
        print(f"\n{'='*55}")
        print(f"RESULTS  (total time: {elapsed:.3f}s)")
        print(f"{'='*55}")
        successes = [r for r in results if r.status == TaskStatus.SUCCESS]
        failures = [r for r in results if r.status == TaskStatus.FAILED]

        print(f"  Succeeded: {len(successes)}/{len(results)}")
        for r in successes:
            print(f"    {r.task_id}: {r.results}")

        if failures:
            print(f"  Failed: {len(failures)}/{len(results)}")
            for r in failures:
                print(
                    f"    {r.task_id}: failed at stage {r.failed_stage} "
                    f"({type(r.error).__name__}: {r.error})"
                )
                if r.results:
                    print(f"      partial results: {r.results}")

        return results


# --- Demo with flaky stages ---
async def flaky_stage(task_id, idx, dur, fail_type=None, fail_prob=0.0):
    await asyncio.sleep(dur)
    if fail_type and random.random() < fail_prob:
        raise fail_type(f"{task_id}-stage{idx} failed")
    return f"{task_id}-{idx}-ok"

async def always_fail(task_id, idx):
    await asyncio.sleep(0.1)
    raise PermanentError(f"bad input for {task_id}-stage{idx}")

async def main():
    tasks = [
        Task("A", [
            Stage("A", 0, lambda: flaky_stage("A", 0, 0.3)),
            Stage("A", 1, lambda: flaky_stage("A", 1, 0.2)),
        ]),
        Task("B", [
            Stage("B", 0, lambda: flaky_stage("B", 0, 0.2, TransientError, 0.7)),
            Stage("B", 1, lambda: flaky_stage("B", 1, 0.3)),
        ]),
        Task("C", [
            Stage("C", 0, lambda: flaky_stage("C", 0, 0.1)),
            Stage("C", 1, lambda: always_fail("C", 1)),  # will always fail
            Stage("C", 2, lambda: flaky_stage("C", 2, 0.2)),  # never reached
        ]),
    ]

    orch = ResilientOrchestrator(max_concurrency=3, max_retries=3)
    await orch.run(tasks)

asyncio.run(main())
```

**What to talk about**:
- `return_exceptions` is NOT used here — instead each task catches its own errors and returns a structured result. This is cleaner.
- Partial results are preserved — if stage 0 and 1 succeed but stage 2 fails, you still have stage 0 and 1 results.
- Transient vs permanent error distinction: talk about real-world examples (network timeout vs invalid input).
- Exponential backoff prevents thundering herd on retries.

---

## Problem 5: Full Interview Simulation — Queue with Fixed Interface

**Prompt**: You are given the following fixed interfaces (you **cannot** change them). Implement `Orchestrator.process()`.

```python
from dataclasses import dataclass
from typing import Awaitable, Callable
import asyncio

@dataclass
class QueueItem:
    """Represents one stage of a task in the queue. DO NOT MODIFY."""
    task_id: str
    stage_index: int       # 0-based, must execute in order per task
    total_stages: int      # total number of stages in this task
    execute: Callable[[], Awaitable[str]]  # the async work to perform

@dataclass
class ProcessingResult:
    """Result of processing the entire queue. DO NOT MODIFY."""
    completed_tasks: dict[str, list[str]]   # task_id -> [stage results in order]
    failed_tasks: dict[str, str]            # task_id -> error message
    timing: dict[str, dict[str, float]]     # task_id -> {"total": ..., "stages": {...}}
```

Requirements:
1. All tasks run in parallel. Stages within a task run sequentially.
2. If a stage fails, the task is marked failed but other tasks continue.
3. Track total time per task and per-stage execution time.
4. Handle the case where queue items arrive in any order (stage 2 might be enqueued before stage 0).

### Solution

```python
import asyncio
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Awaitable, Callable

# --- FIXED INTERFACES (do not modify) ---

@dataclass
class QueueItem:
    task_id: str
    stage_index: int
    total_stages: int
    execute: Callable[[], Awaitable[str]]

@dataclass
class ProcessingResult:
    completed_tasks: dict[str, list[str]]
    failed_tasks: dict[str, str]
    timing: dict[str, dict]

# --- YOUR IMPLEMENTATION ---

class Orchestrator:
    def __init__(self):
        self._task_stages: dict[str, dict[int, QueueItem]] = defaultdict(dict)
        self._task_expected_total: dict[str, int] = {}

    async def process(self, items: list[QueueItem]) -> ProcessingResult:
        """
        Process all queue items. Items may arrive in any order.
        Tasks run in parallel. Stages within a task run sequentially.
        """
        # Step 1: Group items by task_id and sort by stage_index
        tasks: dict[str, list[QueueItem]] = defaultdict(list)
        for item in items:
            tasks[item.task_id].append(item)

        # Sort each task's stages by index
        for task_id in tasks:
            tasks[task_id].sort(key=lambda x: x.stage_index)

        # Validate: check for gaps
        for task_id, stages in tasks.items():
            indices = [s.stage_index for s in stages]
            expected = list(range(stages[0].total_stages))
            if indices != expected:
                # Handle missing stages gracefully
                pass  # Could log a warning here

        # Step 2: Run all tasks in parallel
        completed_tasks: dict[str, list[str]] = {}
        failed_tasks: dict[str, str] = {}
        timing: dict[str, dict] = {}

        async def run_task(task_id: str, stages: list[QueueItem]):
            task_start = time.monotonic()
            stage_timings: dict[str, float] = {}
            results: list[str] = []

            try:
                for stage in stages:
                    stage_start = time.monotonic()
                    result = await stage.execute()
                    stage_elapsed = time.monotonic() - stage_start

                    stage_timings[f"stage_{stage.stage_index}"] = round(
                        stage_elapsed, 4
                    )
                    results.append(result)

                task_total = time.monotonic() - task_start
                completed_tasks[task_id] = results
                timing[task_id] = {
                    "total": round(task_total, 4),
                    "stages": stage_timings,
                }

            except Exception as e:
                task_total = time.monotonic() - task_start
                failed_tasks[task_id] = (
                    f"Failed at stage {len(results)}: {type(e).__name__}: {e}"
                )
                timing[task_id] = {
                    "total": round(task_total, 4),
                    "stages": stage_timings,
                }

        # Launch all tasks concurrently
        await asyncio.gather(*[
            run_task(tid, stages) for tid, stages in tasks.items()
        ])

        return ProcessingResult(
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            timing=timing,
        )


# --- TEST HARNESS ---

async def make_executor(name, duration, should_fail=False):
    async def execute():
        await asyncio.sleep(duration)
        if should_fail:
            raise RuntimeError(f"{name} encountered an error")
        return f"{name}-result"
    return execute

async def main():
    # Simulate queue items arriving in scrambled order
    items = [
        QueueItem("task-1", 2, 3, await make_executor("t1-s2", 0.2)),
        QueueItem("task-2", 0, 2, await make_executor("t2-s0", 0.3)),
        QueueItem("task-1", 0, 3, await make_executor("t1-s0", 0.5)),
        QueueItem("task-3", 0, 2, await make_executor("t3-s0", 0.1)),
        QueueItem("task-1", 1, 3, await make_executor("t1-s1", 0.3)),
        QueueItem("task-2", 1, 2, await make_executor("t2-s1", 0.2)),
        QueueItem("task-3", 1, 2, await make_executor("t3-s1", 0.1, should_fail=True)),
    ]

    orch = Orchestrator()
    start = time.monotonic()
    result = await orch.process(items)
    total = time.monotonic() - start

    print(f"Total time: {total:.3f}s\n")

    print("Completed tasks:")
    for tid, res in result.completed_tasks.items():
        print(f"  {tid}: {res}")

    print("\nFailed tasks:")
    for tid, err in result.failed_tasks.items():
        print(f"  {tid}: {err}")

    print("\nTiming:")
    for tid, t in result.timing.items():
        print(f"  {tid}: total={t['total']}s, stages={t['stages']}")

asyncio.run(main())
```

**Expected output** (approximately):

```
Total time: ~1.0s  (task-1 is longest: 0.5 + 0.3 + 0.2 = 1.0s)

Completed tasks:
  task-1: ['t1-s0-result', 't1-s1-result', 't1-s2-result']
  task-2: ['t2-s0-result', 't2-s1-result']

Failed tasks:
  task-3: Failed at stage 1: RuntimeError: t3-s1 encountered an error

Timing:
  task-1: total=1.0012s, stages={'stage_0': 0.5003, 'stage_1': 0.3005, 'stage_2': 0.2004}
  task-2: total=0.5009s, stages={'stage_0': 0.3004, 'stage_1': 0.2005}
  task-3: total=0.2006s, stages={'stage_0': 0.1003}
```

---

## Problem 6: Dynamic Stage Enqueueing (Advanced)

**Prompt**: Stages arrive dynamically over time (like a live system). Implement an orchestrator that can accept new stages while already processing others. It should shut down gracefully when a sentinel value is received.

### Solution

```python
import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable
from collections import defaultdict

SENTINEL = None  # signals "no more items coming"

@dataclass
class StageItem:
    task_id: str
    stage_index: int
    total_stages: int
    execute: Callable[[], Awaitable[str]]

class LiveOrchestrator:
    def __init__(self):
        self._inbox: asyncio.Queue[StageItem | None] = asyncio.Queue()
        self._task_queues: dict[str, asyncio.Queue] = {}
        self._task_runners: dict[str, asyncio.Task] = {}
        self._results: dict[str, list] = defaultdict(list)
        self._errors: dict[str, str] = {}

    async def submit(self, item: StageItem | None):
        """Submit a stage item (or None to signal completion)."""
        await self._inbox.put(item)

    async def _task_runner(self, task_id: str, total_stages: int):
        """Process stages for a single task in order."""
        task_queue = self._task_queues[task_id]
        pending: dict[int, StageItem] = {}
        next_index = 0

        while next_index < total_stages:
            # Get next item from this task's queue
            item = await task_queue.get()
            pending[item.stage_index] = item

            # Execute all stages that are ready in order
            while next_index in pending:
                stage = pending.pop(next_index)
                try:
                    result = await stage.execute()
                    self._results[task_id].append(result)
                    print(f"  [{task_id}] stage {next_index} -> {result}")
                except Exception as e:
                    self._errors[task_id] = (
                        f"Failed at stage {next_index}: {e}"
                    )
                    return  # stop this task
                next_index += 1

    async def _dispatcher(self):
        """Read from inbox, route items to per-task queues."""
        while True:
            item = await self._inbox.get()
            if item is SENTINEL:
                break

            task_id = item.task_id
            if task_id not in self._task_queues:
                self._task_queues[task_id] = asyncio.Queue()
                self._task_runners[task_id] = asyncio.create_task(
                    self._task_runner(task_id, item.total_stages)
                )

            await self._task_queues[task_id].put(item)

    async def run(self) -> tuple[dict, dict]:
        """Start processing. Returns (results, errors) when done."""
        dispatcher = asyncio.create_task(self._dispatcher())

        # Wait for dispatcher to finish (sentinel received)
        await dispatcher

        # Wait for all task runners to complete
        if self._task_runners:
            await asyncio.gather(*self._task_runners.values())

        return dict(self._results), dict(self._errors)


# --- Demo: simulate stages arriving over time ---
async def sim(name, dur):
    await asyncio.sleep(dur)
    return f"{name}-done"

async def main():
    orch = LiveOrchestrator()

    async def simulate_incoming():
        """Simulates stages arriving from an external source."""
        # Batch 1 — arrives at t=0
        await orch.submit(StageItem("X", 0, 3, lambda: sim("X-0", 0.2)))
        await orch.submit(StageItem("Y", 0, 2, lambda: sim("Y-0", 0.3)))

        await asyncio.sleep(0.1)  # small delay

        # Batch 2 — arrives at t=0.1
        await orch.submit(StageItem("X", 1, 3, lambda: sim("X-1", 0.2)))
        await orch.submit(StageItem("Y", 1, 2, lambda: sim("Y-1", 0.1)))

        await asyncio.sleep(0.2)

        # Batch 3 — arrives at t=0.3 (out of order for X!)
        await orch.submit(StageItem("X", 2, 3, lambda: sim("X-2", 0.1)))

        # Signal done
        await orch.submit(SENTINEL)

    # Run the producer and orchestrator concurrently
    start = time.monotonic()
    _, (results, errors) = await asyncio.gather(
        simulate_incoming(),
        orch.run(),
    )
    elapsed = time.monotonic() - start

    print(f"\nTotal time: {elapsed:.3f}s")
    print(f"Results: {results}")
    print(f"Errors: {errors}")

asyncio.run(main())
```

**What to talk about**:
- This is a real-world pattern: stages arrive from an external source (network, message broker).
- The dispatcher creates per-task queues and runners on demand.
- Out-of-order arrival is handled by buffering in the per-task runner.
- Sentinel value provides a clean shutdown mechanism.
- This separates concerns: dispatching, ordering, execution, and result collection are all independent.

---

## Problem 7: Concurrency-Limited Queue with Priority and Timeout

**Prompt**: Build an orchestrator where:
1. Max N stages execute concurrently across all tasks.
2. Tasks have priority (lower number = higher priority).
3. Each stage has a timeout. If exceeded, the stage fails.

### Solution

```python
import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable
from heapq import heappush, heappop

@dataclass(order=True)
class PrioritizedTask:
    priority: int
    task_id: str = field(compare=False)
    stages: list = field(compare=False, default_factory=list)

@dataclass
class Stage:
    task_id: str
    index: int
    execute: Callable[[], Awaitable[str]]
    timeout: float = 5.0  # seconds

class PriorityOrchestrator:
    def __init__(self, max_concurrency: int = 3):
        self._sem = asyncio.Semaphore(max_concurrency)

    async def _run_stage(self, stage: Stage) -> str:
        async with self._sem:
            try:
                return await asyncio.wait_for(
                    stage.execute(), timeout=stage.timeout
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"{stage.task_id}/stage{stage.index} "
                    f"timed out after {stage.timeout}s"
                )

    async def _run_task(self, task_id: str, stages: list[Stage]):
        results = []
        for stage in stages:
            result = await self._run_stage(stage)
            results.append(result)
        return results

    async def run(self, prioritized_tasks: list[PrioritizedTask]):
        # Sort by priority — lowest first
        sorted_tasks = sorted(prioritized_tasks)

        completed = {}
        failed = {}

        async def safe_run(pt: PrioritizedTask):
            try:
                results = await self._run_task(pt.task_id, pt.stages)
                completed[pt.task_id] = results
            except Exception as e:
                failed[pt.task_id] = str(e)

        # Launch all (sorted by priority, but all start "at once")
        # Priority affects who gets the semaphore first in practice
        await asyncio.gather(*[safe_run(pt) for pt in sorted_tasks])

        return completed, failed


# --- Demo ---
async def sim(name, dur):
    await asyncio.sleep(dur)
    return f"{name}-ok"

async def slow_stage(name):
    await asyncio.sleep(10)  # will timeout
    return f"{name}-ok"

async def main():
    tasks = [
        PrioritizedTask(
            priority=2, task_id="low-pri",
            stages=[
                Stage("low-pri", 0, lambda: sim("lp-0", 0.5)),
                Stage("low-pri", 1, lambda: sim("lp-1", 0.3)),
            ]
        ),
        PrioritizedTask(
            priority=0, task_id="high-pri",
            stages=[
                Stage("high-pri", 0, lambda: sim("hp-0", 0.2)),
                Stage("high-pri", 1, lambda: sim("hp-1", 0.2)),
            ]
        ),
        PrioritizedTask(
            priority=1, task_id="med-pri",
            stages=[
                Stage("med-pri", 0, lambda: sim("mp-0", 0.3)),
                Stage("med-pri", 1, lambda: slow_stage("mp-1"), 0.5),  # will timeout
            ]
        ),
    ]

    orch = PriorityOrchestrator(max_concurrency=2)
    start = time.monotonic()
    completed, failed = await orch.run(tasks)
    elapsed = time.monotonic() - start

    print(f"Time: {elapsed:.3f}s")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")

asyncio.run(main())
```

---

## Key Things to Remember During the Interview

### Before You Code

1. **Draw the execution model first.** Sketch the timeline: which tasks run in parallel, which stages are sequential. Confirm your understanding with the interviewer.

2. **State your assumptions aloud.** "I'm assuming stages are idempotent for retry purposes." "I'm assuming the queue is bounded." This shows maturity.

3. **Identify the fixed interfaces.** What can you change vs what's given? Design around constraints, not against them.

### While Coding

4. **Use `time.monotonic()`, never `time.time()`.** Mention why — clock adjustments. This is a signal that you know what you're doing.

5. **Prefer `asyncio.TaskGroup` (3.11+) over `gather`.** If the interviewer allows it. It's safer (auto-cancellation) and shows you know modern Python. But know `gather` cold too.

6. **Separate ordering from concurrency.** Don't mix "which order should stages run" with "how many things run at once." Use per-task coroutines for ordering and semaphores for concurrency limits.

7. **Always call `queue.task_done()` in a `finally` block.** Otherwise `queue.join()` hangs forever if a stage throws.

8. **Classify errors before retrying.** Transient vs permanent. Don't retry bad input. Don't give up on a network timeout after one try.

9. **Use `asyncio.wait_for()` for timeouts.** Wrap the stage execution, not the entire task.

10. **Handle partial results.** If 3 of 5 stages complete before a failure, preserve those results. Don't throw everything away.

### Communication Tips

11. **Talk through tradeoffs out loud.** "I could use a condition variable here, but per-task coroutines are simpler and avoid worker starvation."

12. **Mention what you'd add in production.** Circuit breakers, dead letter queues, metrics, structured logging — even if you don't implement them.

13. **If stuck, simplify first.** Get the basic parallel-tasks-sequential-stages pattern working, then add error handling, then timing, then retries.

14. **Name your variables well.** `sem` for semaphore, `stage_start` for timing, `task_result` for outcomes. Clean code communicates intent.

15. **Don't over-engineer.** The interview wants good judgment, not a production framework. A clean 50-line solution beats a messy 200-line one.

---

## Sources

- [Python asyncio.Queue Documentation](https://docs.python.org/3/library/asyncio-queue.html)
- [Python Coroutines and Tasks Documentation](https://docs.python.org/3/library/asyncio-task.html)
- [Scale AI System Design Interview Guide](https://www.systemdesignhandbook.com/guides/scale-ai-system-design-interview/)
- [Async Queue Interview Problem — David Gomes](https://davidgomes.com/async-queue-interview-ai/)
- [Real Python asyncio Walkthrough](https://realpython.com/async-io-python/)
