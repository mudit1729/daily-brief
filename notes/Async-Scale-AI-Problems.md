# Scale AI — Domain-Specific Async Problems

> These problems simulate the kind of systems Scale AI actually builds: data annotation pipelines, LLM orchestration, worker routing, and quality control. Each problem has a full Python solution and interview talking points.

---

## Context: What Scale AI Actually Does

Scale AI's core product is a **human-in-the-loop data labeling platform**. At a systems level, this means:

- **Tasks** (label this image, rate this LLM response, transcribe this audio) arrive as a continuous stream.
- **Workers** (human annotators) have different skill levels, availability, and speed.
- Tasks flow through **multi-stage pipelines**: initial annotation → review → quality audit → consensus resolution.
- **LLM services** are integrated as black boxes — you send a prompt, wait for a response, and route it through human review.
- **Quality control** is not optional — it's the product. Consensus, spot-checks, and golden-set evaluation run at every stage.

The interview tests whether you can build orchestration systems that handle this kind of workload cleanly.

---

## Problem 1: Priority Task Assignment with Min-Heap Worker Pool

**Scenario**: You have a pool of workers. Each worker has a current load (number of active tasks). When a new task arrives, assign it to the **least-loaded worker**. If workers have equal load, pick the one with the **highest skill score** (tiebreaker). Workers process tasks asynchronously — when a task finishes, the worker's load decreases.

### Solution

```python
import asyncio
import heapq
import time
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class Worker:
    """
    Min-heap ordering: (current_load, -skill_score, worker_id)
    Lower load = higher priority.
    Higher skill = higher priority (negated for min-heap).
    """
    current_load: int
    neg_skill: float = field(compare=True)    # negated so higher skill wins
    worker_id: str = field(compare=True)
    skill_score: float = field(compare=False, repr=True)

    @classmethod
    def create(cls, worker_id: str, skill_score: float, current_load: int = 0):
        return cls(
            current_load=current_load,
            neg_skill=-skill_score,
            worker_id=worker_id,
            skill_score=skill_score,
        )

@dataclass
class Task:
    task_id: str
    difficulty: float  # 0.0 to 1.0
    duration: float    # simulated processing time

@dataclass
class Assignment:
    task_id: str
    worker_id: str
    assigned_at: float
    completed_at: float | None = None
    result: str | None = None


class MinHeapWorkerPool:
    """
    Assigns tasks to the least-loaded worker using a min-heap.
    Thread-safe via asyncio (single-threaded event loop).
    """

    def __init__(self, workers: list[Worker]):
        self._heap: list[Worker] = list(workers)
        heapq.heapify(self._heap)
        self._lock = asyncio.Lock()
        self._assignments: list[Assignment] = []

    async def assign_task(self, task: Task) -> Assignment:
        """Pop least-loaded worker, increment load, push back, execute task."""
        async with self._lock:
            # Pop the least-loaded worker
            worker = heapq.heappop(self._heap)

            # Create assignment
            assignment = Assignment(
                task_id=task.task_id,
                worker_id=worker.worker_id,
                assigned_at=time.monotonic(),
            )

            # Increment load and push back
            updated = Worker.create(
                worker.worker_id,
                worker.skill_score,
                worker.current_load + 1,
            )
            heapq.heappush(self._heap, updated)

        # Execute task (outside lock — allows other assignments concurrently)
        print(
            f"  [{worker.worker_id}] (load={worker.current_load}, "
            f"skill={worker.skill_score}) assigned {task.task_id}"
        )
        await asyncio.sleep(task.duration)  # simulate work

        # Decrement load
        async with self._lock:
            # Find and update this worker in the heap
            for i, w in enumerate(self._heap):
                if w.worker_id == worker.worker_id:
                    self._heap[i] = Worker.create(
                        w.worker_id, w.skill_score, w.current_load - 1
                    )
                    heapq.heapify(self._heap)  # re-heapify after mutation
                    break

        assignment.completed_at = time.monotonic()
        assignment.result = f"{task.task_id} done by {worker.worker_id}"
        self._assignments.append(assignment)
        return assignment

    async def assign_batch(self, tasks: list[Task]) -> list[Assignment]:
        """Assign all tasks concurrently. Heap auto-balances load."""
        return await asyncio.gather(*[self.assign_task(t) for t in tasks])

    def print_report(self):
        print("\n--- Assignment Report ---")
        for a in sorted(self._assignments, key=lambda x: x.assigned_at):
            duration = (a.completed_at or 0) - a.assigned_at
            print(
                f"  {a.task_id} → {a.worker_id} "
                f"(duration: {duration:.3f}s)"
            )

        # Worker load distribution
        print("\n--- Final Worker State ---")
        for w in sorted(self._heap, key=lambda x: x.worker_id):
            print(
                f"  {w.worker_id}: load={w.current_load}, "
                f"skill={w.skill_score}"
            )


# --- Demo ---
async def main():
    workers = [
        Worker.create("alice", skill_score=0.95),
        Worker.create("bob", skill_score=0.82),
        Worker.create("carol", skill_score=0.88),
        Worker.create("dave", skill_score=0.75),
    ]

    pool = MinHeapWorkerPool(workers)

    tasks = [
        Task(f"task-{i}", difficulty=0.5, duration=0.2 + (i % 3) * 0.1)
        for i in range(10)
    ]

    start = time.monotonic()
    await pool.assign_batch(tasks)
    elapsed = time.monotonic() - start

    pool.print_report()
    print(f"\nTotal time: {elapsed:.3f}s")

asyncio.run(main())
```

### What to Talk About

- **Why min-heap?** O(log n) insert and pop — efficient even with hundreds of workers. `heapq` is Python's built-in, no dependencies needed.
- **Tiebreaking**: The `Worker` dataclass uses `(load, -skill, id)` ordering. Higher skill workers get priority when loads are equal.
- **Lock scope**: The lock only covers heap mutations (fast), not task execution (slow). This is critical — locking during execution would serialize everything.
- **Heap re-heapify**: After modifying a worker's load in-place, we call `heapify()`. In production you'd use a more efficient indexed heap, but for an interview this is fine and correct.
- **Race condition awareness**: Between popping a worker and the task finishing, the worker appears in the heap with load+1. This is correct — it prevents over-assignment.

### Scale AI Connection

This is exactly how Scale AI routes annotation tasks. Workers have quality scores, speed metrics, and domain expertise. The system must assign tasks to the best available worker in real-time, balancing load across the pool while maximizing output quality.

---

## Problem 2: LLM Black-Box Pipeline with Segmented Requests

**Scenario**: You're building a system that processes documents through an LLM service. Each document is split into segments (chunks). Each segment is sent to the LLM independently (in parallel), but results must be assembled in the original order. The LLM is a black box — it may be slow, it may fail, and you have rate limits.

### Solution

```python
import asyncio
import time
import random
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Segment:
    doc_id: str
    segment_index: int
    text: str

@dataclass
class SegmentResult:
    doc_id: str
    segment_index: int
    output: str
    latency: float
    attempts: int

class LLMClient:
    """Simulates a flaky, rate-limited LLM API."""

    def __init__(self, rate_limit: int = 5, failure_rate: float = 0.2):
        self._sem = asyncio.Semaphore(rate_limit)
        self._failure_rate = failure_rate
        self._call_count = 0

    async def complete(self, text: str) -> str:
        """Simulate LLM API call with latency and random failures."""
        async with self._sem:
            self._call_count += 1
            # Simulate variable latency (100-500ms)
            latency = 0.1 + random.random() * 0.4
            await asyncio.sleep(latency)

            # Simulate random failures
            if random.random() < self._failure_rate:
                raise ConnectionError("LLM service temporarily unavailable")

            return f"[LLM output for: {text[:30]}...]"

    @property
    def total_calls(self) -> int:
        return self._call_count


class SegmentedLLMPipeline:
    """
    Processes documents by splitting into segments, sending each to LLM
    in parallel, and reassembling results in order.
    """

    def __init__(
        self,
        llm: LLMClient,
        max_retries: int = 3,
        chunk_size: int = 200,
    ):
        self._llm = llm
        self._max_retries = max_retries
        self._chunk_size = chunk_size

    def _split_document(self, doc_id: str, text: str) -> list[Segment]:
        """Split document into fixed-size chunks."""
        segments = []
        for i in range(0, len(text), self._chunk_size):
            segments.append(Segment(
                doc_id=doc_id,
                segment_index=len(segments),
                text=text[i:i + self._chunk_size],
            ))
        return segments

    async def _process_segment(self, segment: Segment) -> SegmentResult:
        """Process one segment with retry logic."""
        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                start = time.monotonic()
                output = await self._llm.complete(segment.text)
                latency = time.monotonic() - start

                return SegmentResult(
                    doc_id=segment.doc_id,
                    segment_index=segment.segment_index,
                    output=output,
                    latency=latency,
                    attempts=attempt,
                )
            except ConnectionError as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = 0.5 * (2 ** (attempt - 1))
                    print(
                        f"  [{segment.doc_id}] segment {segment.segment_index} "
                        f"failed (attempt {attempt}), retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"Segment {segment.doc_id}/{segment.segment_index} failed "
            f"after {self._max_retries} attempts: {last_error}"
        )

    async def process_document(self, doc_id: str, text: str) -> dict:
        """
        Split document → process segments in parallel → reassemble in order.
        """
        segments = self._split_document(doc_id, text)
        print(f"[{doc_id}] Split into {len(segments)} segments")

        start = time.monotonic()

        # Process all segments concurrently
        results = await asyncio.gather(
            *[self._process_segment(seg) for seg in segments],
            return_exceptions=True,
        )

        elapsed = time.monotonic() - start

        # Check for failures
        failures = [
            (i, r) for i, r in enumerate(results)
            if isinstance(r, Exception)
        ]
        successes = [
            r for r in results if isinstance(r, SegmentResult)
        ]

        if failures:
            failed_indices = [i for i, _ in failures]
            return {
                "doc_id": doc_id,
                "status": "partial_failure",
                "assembled_text": None,
                "failed_segments": failed_indices,
                "successful_segments": len(successes),
                "total_segments": len(segments),
                "elapsed": elapsed,
            }

        # Reassemble in original order
        ordered = sorted(successes, key=lambda r: r.segment_index)
        assembled = "\n".join(r.output for r in ordered)

        total_attempts = sum(r.attempts for r in ordered)
        avg_latency = sum(r.latency for r in ordered) / len(ordered)

        return {
            "doc_id": doc_id,
            "status": "success",
            "assembled_text": assembled,
            "total_segments": len(segments),
            "total_llm_calls": total_attempts,
            "avg_segment_latency": round(avg_latency, 4),
            "wall_clock_time": round(elapsed, 4),
            "speedup_vs_sequential": round(
                sum(r.latency for r in ordered) / elapsed, 2
            ),
        }

    async def process_batch(self, documents: dict[str, str]) -> list[dict]:
        """Process multiple documents concurrently."""
        return await asyncio.gather(*[
            self.process_document(doc_id, text)
            for doc_id, text in documents.items()
        ])


# --- Demo ---
async def main():
    llm = LLMClient(rate_limit=5, failure_rate=0.15)
    pipeline = SegmentedLLMPipeline(llm, max_retries=3, chunk_size=100)

    # Simulate documents of varying length
    documents = {
        "doc-1": "A" * 350,   # 4 segments
        "doc-2": "B" * 200,   # 2 segments
        "doc-3": "C" * 500,   # 5 segments
    }

    start = time.monotonic()
    results = await pipeline.process_batch(documents)
    total = time.monotonic() - start

    print(f"\n{'='*60}")
    print(f"PIPELINE RESULTS (total: {total:.3f}s, LLM calls: {llm.total_calls})")
    print(f"{'='*60}")
    for r in results:
        print(f"\n  {r['doc_id']}: {r['status']}")
        print(f"    segments: {r['total_segments']}")
        if r['status'] == 'success':
            print(f"    LLM calls: {r['total_llm_calls']} (incl. retries)")
            print(f"    avg latency/segment: {r['avg_segment_latency']}s")
            print(f"    wall clock: {r['wall_clock_time']}s")
            print(f"    speedup: {r['speedup_vs_sequential']}x")
        else:
            print(f"    failed segments: {r['failed_segments']}")

asyncio.run(main())
```

### What to Talk About

- **Segment ordering**: `gather` preserves order, but we sort by `segment_index` explicitly for clarity and safety. This is defensive coding.
- **Rate limiting**: The `Semaphore(5)` inside `LLMClient` means at most 5 concurrent LLM calls. This simulates real API rate limits.
- **Retry isolation**: Each segment retries independently. If segment 3 fails and retries, segments 1, 2, 4, 5 don't wait.
- **Speedup metric**: We track `sum(individual_latencies) / wall_clock_time`. With 5 segments and rate limit 5, ideal speedup is ~5x.
- **Partial failure handling**: If some segments fail permanently, we report which ones failed. In production you'd offer reprocessing just the failed segments.

### Scale AI Connection

Scale AI uses LLMs for pre-labeling, quality estimation, and response evaluation. Documents (prompts, conversations, articles) are split into segments for parallel processing. The system must handle LLM API flakiness, rate limits, and reassemble results correctly. This pattern is fundamental to their RLHF (Reinforcement Learning from Human Feedback) pipeline.

---

## Problem 3: Worker State Management with Quality-Based Routing

**Scenario**: Workers have state: active task count, accuracy score (rolling average), and specialization tags. Tasks have required tags. The router must:
1. Only assign tasks to workers with matching tags.
2. Among eligible workers, prefer workers with higher accuracy.
3. Workers have a max concurrent task limit.
4. Update accuracy scores as tasks complete (with quality feedback).

### Solution

```python
import asyncio
import time
import random
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

@dataclass
class WorkerState:
    worker_id: str
    tags: set[str]
    max_concurrent: int = 3
    active_tasks: int = 0
    accuracy_scores: list[float] = field(default_factory=list)
    total_completed: int = 0

    @property
    def accuracy(self) -> float:
        if not self.accuracy_scores:
            return 0.5  # default for new workers
        # Rolling average of last 20 scores
        recent = self.accuracy_scores[-20:]
        return sum(recent) / len(recent)

    @property
    def available(self) -> bool:
        return self.active_tasks < self.max_concurrent

    @property
    def load_fraction(self) -> float:
        return self.active_tasks / self.max_concurrent

@dataclass
class AnnotationTask:
    task_id: str
    required_tags: set[str]
    duration: float
    difficulty: float  # affects quality outcome

@dataclass
class TaskCompletion:
    task_id: str
    worker_id: str
    quality_score: float  # 0.0 to 1.0, determined by QA
    duration: float


class QualityAwareRouter:
    """
    Routes tasks to workers based on:
    1. Tag matching (hard filter)
    2. Availability (must have capacity)
    3. Accuracy score (soft preference — higher is better)
    """

    def __init__(self, workers: list[WorkerState]):
        self._workers = {w.worker_id: w for w in workers}
        self._lock = asyncio.Lock()
        self._completions: list[TaskCompletion] = []

    def _find_best_worker(self, task: AnnotationTask) -> Optional[WorkerState]:
        """Find the best available worker for this task."""
        eligible = [
            w for w in self._workers.values()
            if w.available
            and task.required_tags.issubset(w.tags)  # worker has all required tags
        ]

        if not eligible:
            return None

        # Sort by accuracy (descending), then by load (ascending) as tiebreaker
        eligible.sort(key=lambda w: (-w.accuracy, w.load_fraction))
        return eligible[0]

    async def _execute_task(
        self, task: AnnotationTask, worker: WorkerState
    ) -> TaskCompletion:
        """Simulate task execution and generate quality score."""
        start = time.monotonic()
        await asyncio.sleep(task.duration)
        elapsed = time.monotonic() - start

        # Simulate quality: based on worker accuracy + task difficulty + noise
        base_quality = worker.accuracy * (1 - task.difficulty * 0.3)
        noise = random.gauss(0, 0.05)
        quality = max(0.0, min(1.0, base_quality + noise))

        return TaskCompletion(
            task_id=task.task_id,
            worker_id=worker.worker_id,
            quality_score=round(quality, 3),
            duration=round(elapsed, 4),
        )

    async def route_and_execute(self, task: AnnotationTask) -> TaskCompletion | str:
        """Route a task to the best worker and execute it."""
        async with self._lock:
            worker = self._find_best_worker(task)
            if worker is None:
                return f"No eligible worker for {task.task_id} (tags: {task.required_tags})"

            # Reserve the worker
            worker.active_tasks += 1
            print(
                f"  {task.task_id} → {worker.worker_id} "
                f"(accuracy={worker.accuracy:.2f}, load={worker.active_tasks}/{worker.max_concurrent})"
            )

        try:
            # Execute outside lock
            completion = await self._execute_task(task, worker)

            # Update worker state
            async with self._lock:
                worker.active_tasks -= 1
                worker.accuracy_scores.append(completion.quality_score)
                worker.total_completed += 1

            self._completions.append(completion)
            return completion

        except Exception as e:
            async with self._lock:
                worker.active_tasks -= 1
            raise

    async def process_batch(self, tasks: list[AnnotationTask]):
        """Process all tasks concurrently with quality-aware routing."""
        results = await asyncio.gather(
            *[self.route_and_execute(t) for t in tasks],
            return_exceptions=True,
        )
        return results

    def print_report(self):
        print(f"\n{'='*60}")
        print("ROUTING REPORT")
        print(f"{'='*60}")

        # Per-worker stats
        print("\nWorker Performance:")
        for w in sorted(self._workers.values(), key=lambda x: -x.accuracy):
            print(
                f"  {w.worker_id}: accuracy={w.accuracy:.3f}, "
                f"completed={w.total_completed}, tags={w.tags}"
            )

        # Quality distribution
        if self._completions:
            scores = [c.quality_score for c in self._completions]
            print(f"\nQuality Distribution:")
            print(f"  avg={sum(scores)/len(scores):.3f}")
            print(f"  min={min(scores):.3f}, max={max(scores):.3f}")

            # Per-worker quality
            worker_scores = defaultdict(list)
            for c in self._completions:
                worker_scores[c.worker_id].append(c.quality_score)

            print(f"\nPer-Worker Quality:")
            for wid, scores in sorted(worker_scores.items()):
                avg = sum(scores) / len(scores)
                print(f"  {wid}: avg={avg:.3f} ({len(scores)} tasks)")


# --- Demo ---
async def main():
    workers = [
        WorkerState("alice", tags={"nlp", "sentiment", "rlhf"}, max_concurrent=3,
                     accuracy_scores=[0.92, 0.88, 0.95, 0.91]),
        WorkerState("bob", tags={"nlp", "classification"}, max_concurrent=2,
                     accuracy_scores=[0.78, 0.82, 0.80]),
        WorkerState("carol", tags={"vision", "segmentation"}, max_concurrent=3,
                     accuracy_scores=[0.90, 0.93, 0.89]),
        WorkerState("dave", tags={"nlp", "rlhf", "classification"}, max_concurrent=2,
                     accuracy_scores=[0.85, 0.87, 0.84]),
    ]

    tasks = [
        AnnotationTask(f"nlp-{i}", {"nlp"}, duration=0.2, difficulty=0.3)
        for i in range(5)
    ] + [
        AnnotationTask(f"rlhf-{i}", {"nlp", "rlhf"}, duration=0.3, difficulty=0.6)
        for i in range(4)
    ] + [
        AnnotationTask(f"vision-{i}", {"vision"}, duration=0.25, difficulty=0.4)
        for i in range(3)
    ]

    random.shuffle(tasks)

    router = QualityAwareRouter(workers)
    start = time.monotonic()
    results = await router.process_batch(tasks)
    elapsed = time.monotonic() - start

    router.print_report()
    print(f"\nTotal time: {elapsed:.3f}s")

asyncio.run(main())
```

### What to Talk About

- **Tag matching is a hard filter** — you can't assign a vision task to an NLP-only worker. This mirrors Scale AI's domain specialization.
- **Accuracy is a soft preference** — among eligible workers, higher accuracy wins. But if the best worker is at capacity, we fall back to the next best.
- **Rolling average for accuracy** — last 20 scores. This means a worker who improves over time isn't penalized by early mistakes, and a worker who degrades gets fewer tasks quickly.
- **Lock granularity** — we lock only during worker selection (fast), not during task execution (slow). This prevents a single slow task from blocking all routing.
- **What's missing for production**: worker cooldown periods, task priority, deadline awareness, and geographic affinity. Mention these to show systems thinking.

---

## Problem 4: Continuous Stream with Quality Control Checkpoints

**Scenario**: Tasks arrive as a continuous stream. Each task goes through three stages:
1. **Annotate**: A worker labels the data.
2. **Review**: A different worker reviews the annotation (cannot be the same worker).
3. **Audit**: If the review score is below a threshold, send to a senior reviewer. Otherwise, auto-approve.

The system must track throughput, quality metrics, and handle backpressure when workers are overloaded.

### Solution

```python
import asyncio
import time
import random
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

class QualityGate(Enum):
    PASS = "pass"
    FAIL = "fail"
    NEEDS_AUDIT = "needs_audit"

@dataclass
class StreamItem:
    item_id: str
    data: str
    created_at: float = 0.0

@dataclass
class AnnotationResult:
    item_id: str
    annotator: str
    label: str
    confidence: float

@dataclass
class ReviewResult:
    item_id: str
    reviewer: str
    score: float        # 0.0 to 1.0
    gate: QualityGate

@dataclass
class AuditResult:
    item_id: str
    auditor: str
    final_score: float
    approved: bool

@dataclass
class PipelineMetrics:
    total_items: int = 0
    completed: int = 0
    failed: int = 0
    audited: int = 0
    auto_approved: int = 0
    total_latency: float = 0.0
    stage_times: dict = field(default_factory=lambda: defaultdict(list))

    @property
    def avg_latency(self) -> float:
        return self.total_latency / max(self.completed, 1)

    @property
    def throughput(self) -> float:
        """Items per second."""
        if not self.stage_times["total"]:
            return 0
        total_time = max(self.stage_times["total"])
        return self.completed / max(total_time, 0.001)


class AnnotationPipeline:
    """
    3-stage pipeline: Annotate → Review → (optional) Audit
    With quality gates, worker exclusion, and backpressure.
    """

    REVIEW_THRESHOLD = 0.7   # below this → needs audit
    AUDIT_THRESHOLD = 0.6    # below this → rejected

    def __init__(
        self,
        annotators: list[str],
        reviewers: list[str],
        auditors: list[str],
        max_in_flight: int = 10,
    ):
        self._annotators = annotators
        self._reviewers = reviewers
        self._auditors = auditors

        # Backpressure: limit items in the pipeline at once
        self._sem = asyncio.Semaphore(max_in_flight)

        # Per-worker semaphores (max 2 concurrent tasks per worker)
        self._worker_sems = {
            w: asyncio.Semaphore(2)
            for w in annotators + reviewers + auditors
        }

        self.metrics = PipelineMetrics()
        self._pipeline_start = 0.0

    async def _annotate(self, item: StreamItem) -> AnnotationResult:
        """Stage 1: Initial annotation."""
        annotator = random.choice(self._annotators)
        async with self._worker_sems[annotator]:
            start = time.monotonic()
            await asyncio.sleep(random.uniform(0.1, 0.3))  # simulate work
            elapsed = time.monotonic() - start
            self.metrics.stage_times["annotate"].append(elapsed)

            return AnnotationResult(
                item_id=item.item_id,
                annotator=annotator,
                label=f"label-for-{item.item_id}",
                confidence=random.uniform(0.5, 1.0),
            )

    async def _review(
        self, annotation: AnnotationResult
    ) -> ReviewResult:
        """Stage 2: Review by a DIFFERENT worker."""
        # Exclude the original annotator
        eligible = [r for r in self._reviewers if r != annotation.annotator]
        if not eligible:
            eligible = self._reviewers  # fallback if pool is too small

        reviewer = random.choice(eligible)
        async with self._worker_sems[reviewer]:
            start = time.monotonic()
            await asyncio.sleep(random.uniform(0.05, 0.2))
            elapsed = time.monotonic() - start
            self.metrics.stage_times["review"].append(elapsed)

            score = random.uniform(0.3, 1.0)
            gate = (
                QualityGate.PASS if score >= self.REVIEW_THRESHOLD
                else QualityGate.NEEDS_AUDIT
            )

            return ReviewResult(
                item_id=annotation.item_id,
                reviewer=reviewer,
                score=round(score, 3),
                gate=gate,
            )

    async def _audit(self, review: ReviewResult) -> AuditResult:
        """Stage 3 (conditional): Senior audit for low-quality items."""
        auditor = random.choice(self._auditors)
        async with self._worker_sems[auditor]:
            start = time.monotonic()
            await asyncio.sleep(random.uniform(0.1, 0.25))
            elapsed = time.monotonic() - start
            self.metrics.stage_times["audit"].append(elapsed)

            final_score = random.uniform(0.4, 1.0)
            approved = final_score >= self.AUDIT_THRESHOLD

            return AuditResult(
                item_id=review.item_id,
                auditor=auditor,
                final_score=round(final_score, 3),
                approved=approved,
            )

    async def _process_item(self, item: StreamItem) -> dict:
        """Full pipeline for one item: annotate → review → maybe audit."""
        item_start = time.monotonic()

        async with self._sem:  # backpressure: limit items in-flight
            # Stage 1: Annotate
            annotation = await self._annotate(item)

            # Stage 2: Review
            review = await self._review(annotation)

            # Stage 3: Conditional Audit
            audit = None
            if review.gate == QualityGate.NEEDS_AUDIT:
                audit = await self._audit(review)
                self.metrics.audited += 1
                if not audit.approved:
                    self.metrics.failed += 1
                    return {
                        "item_id": item.item_id,
                        "status": "rejected",
                        "reason": f"Audit score {audit.final_score} < {self.AUDIT_THRESHOLD}",
                        "annotator": annotation.annotator,
                        "reviewer": review.reviewer,
                        "auditor": audit.auditor,
                    }
            else:
                self.metrics.auto_approved += 1

            total_time = time.monotonic() - item_start
            self.metrics.completed += 1
            self.metrics.total_latency += total_time
            self.metrics.stage_times["total"].append(total_time)

            return {
                "item_id": item.item_id,
                "status": "approved",
                "review_score": review.score,
                "audited": audit is not None,
                "annotator": annotation.annotator,
                "reviewer": review.reviewer,
            }

    async def process_stream(self, items: list[StreamItem]) -> list[dict]:
        """Process a stream of items through the pipeline."""
        self.metrics.total_items = len(items)
        self._pipeline_start = time.monotonic()

        # All items processed concurrently (semaphore limits in-flight)
        results = await asyncio.gather(
            *[self._process_item(item) for item in items],
            return_exceptions=True,
        )

        # Handle any unexpected errors
        final = []
        for item, result in zip(items, results):
            if isinstance(result, Exception):
                self.metrics.failed += 1
                final.append({
                    "item_id": item.item_id,
                    "status": "error",
                    "error": str(result),
                })
            else:
                final.append(result)
        return final

    def print_report(self):
        m = self.metrics
        pipeline_wall = time.monotonic() - self._pipeline_start

        print(f"\n{'='*60}")
        print(f"PIPELINE REPORT")
        print(f"{'='*60}")
        print(f"  Total items:    {m.total_items}")
        print(f"  Completed:      {m.completed}")
        print(f"  Failed/Rejected:{m.failed}")
        print(f"  Auto-approved:  {m.auto_approved}")
        print(f"  Sent to audit:  {m.audited}")
        print(f"  Avg latency:    {m.avg_latency:.3f}s per item")
        print(f"  Wall clock:     {pipeline_wall:.3f}s")
        print(f"  Throughput:     {m.completed / max(pipeline_wall, 0.001):.1f} items/sec")

        print(f"\n  Stage Breakdown (avg):")
        for stage in ["annotate", "review", "audit"]:
            times = m.stage_times.get(stage, [])
            if times:
                avg = sum(times) / len(times)
                print(f"    {stage:>10}: {avg:.3f}s ({len(times)} executions)")


# --- Demo ---
async def main():
    pipeline = AnnotationPipeline(
        annotators=["ann-1", "ann-2", "ann-3"],
        reviewers=["rev-1", "rev-2"],
        auditors=["audit-1"],
        max_in_flight=8,
    )

    # Simulate stream of 20 items
    items = [
        StreamItem(
            item_id=f"item-{i:03d}",
            data=f"data-{i}",
            created_at=time.monotonic(),
        )
        for i in range(20)
    ]

    results = await pipeline.process_stream(items)
    pipeline.print_report()

    # Show a few results
    print(f"\nSample Results:")
    for r in results[:5]:
        print(f"  {r['item_id']}: {r['status']}")

asyncio.run(main())
```

### What to Talk About

- **Worker exclusion**: The reviewer must be different from the annotator. This prevents self-review bias. Mention that Scale AI implements this at every level.
- **Conditional stages**: Not every item goes to audit. This is a branching pipeline — talk about how real pipelines have multiple paths based on quality signals.
- **Backpressure via semaphore**: `max_in_flight=10` limits how many items are in the pipeline at once. Without this, 1000 items would all start immediately and overwhelm workers.
- **Per-worker semaphores**: Each worker can handle at most 2 tasks at once. This prevents a single worker from being overloaded even if the global limit is higher.
- **Metrics are first-class**: Throughput, latency, audit rate, auto-approval rate. Scale AI's platform monitors these in real-time to detect quality degradation.
- **What's missing for production**: dynamic threshold adjustment, worker fatigue modeling, golden-set evaluation (inject known-answer tasks to measure annotator accuracy), and escalation paths.

---

## Problem 5: Consensus-Based Annotation with Majority Voting

**Scenario**: For high-stakes tasks, multiple annotators label the same item independently. The system waits for N annotations, then resolves the label by majority vote. If there's no consensus, escalate to a tie-breaking expert.

### Solution

```python
import asyncio
import time
import random
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional

@dataclass
class ConsensusTask:
    task_id: str
    data: str
    required_annotations: int = 3
    consensus_threshold: float = 0.67  # 2/3 majority

@dataclass
class Annotation:
    task_id: str
    worker_id: str
    label: str
    confidence: float
    duration: float

@dataclass
class ConsensusResult:
    task_id: str
    final_label: str
    method: str   # "majority_vote", "tiebreaker", "unanimous"
    annotations: list[Annotation]
    agreement_score: float  # fraction that agreed with final label
    total_time: float

POSSIBLE_LABELS = ["positive", "negative", "neutral"]

class ConsensusOrchestrator:
    def __init__(
        self,
        annotators: list[str],
        tiebreaker: str,
        annotations_per_task: int = 3,
    ):
        self._annotators = annotators
        self._tiebreaker = tiebreaker
        self._n = annotations_per_task
        self._results: list[ConsensusResult] = []

    async def _get_annotation(
        self, task: ConsensusTask, worker_id: str
    ) -> Annotation:
        """Simulate one worker annotating."""
        start = time.monotonic()
        await asyncio.sleep(random.uniform(0.1, 0.4))
        duration = time.monotonic() - start

        # Simulate: workers agree ~70% of the time on the "true" label
        true_label = POSSIBLE_LABELS[hash(task.data) % len(POSSIBLE_LABELS)]
        if random.random() < 0.7:
            label = true_label
        else:
            label = random.choice(POSSIBLE_LABELS)

        return Annotation(
            task_id=task.task_id,
            worker_id=worker_id,
            label=label,
            confidence=random.uniform(0.5, 1.0),
            duration=round(duration, 4),
        )

    async def _resolve_consensus(
        self, task: ConsensusTask, annotations: list[Annotation]
    ) -> ConsensusResult:
        """Resolve label from annotations. Escalate if no consensus."""
        start = time.monotonic()

        label_counts = Counter(a.label for a in annotations)
        most_common_label, most_common_count = label_counts.most_common(1)[0]
        agreement = most_common_count / len(annotations)

        # Check for unanimous agreement
        if agreement == 1.0:
            return ConsensusResult(
                task_id=task.task_id,
                final_label=most_common_label,
                method="unanimous",
                annotations=annotations,
                agreement_score=1.0,
                total_time=time.monotonic() - start,
            )

        # Check for majority
        if agreement >= task.consensus_threshold:
            return ConsensusResult(
                task_id=task.task_id,
                final_label=most_common_label,
                method="majority_vote",
                annotations=annotations,
                agreement_score=round(agreement, 3),
                total_time=time.monotonic() - start,
            )

        # No consensus — escalate to tiebreaker
        print(f"  [{task.task_id}] No consensus ({label_counts}), escalating...")
        tiebreaker_ann = await self._get_annotation(task, self._tiebreaker)
        annotations.append(tiebreaker_ann)

        # Use tiebreaker's label
        return ConsensusResult(
            task_id=task.task_id,
            final_label=tiebreaker_ann.label,
            method="tiebreaker",
            annotations=annotations,
            agreement_score=round(agreement, 3),
            total_time=time.monotonic() - start,
        )

    async def process_task(self, task: ConsensusTask) -> ConsensusResult:
        """Get N annotations in parallel, then resolve."""
        # Select N random annotators
        selected = random.sample(
            self._annotators,
            min(self._n, len(self._annotators)),
        )

        # All annotators work in parallel
        annotations = await asyncio.gather(*[
            self._get_annotation(task, worker) for worker in selected
        ])

        result = await self._resolve_consensus(task, list(annotations))
        self._results.append(result)
        return result

    async def process_batch(
        self, tasks: list[ConsensusTask]
    ) -> list[ConsensusResult]:
        """Process all tasks with consensus resolution."""
        return await asyncio.gather(*[
            self.process_task(t) for t in tasks
        ])

    def print_report(self):
        print(f"\n{'='*60}")
        print("CONSENSUS REPORT")
        print(f"{'='*60}")

        method_counts = Counter(r.method for r in self._results)
        print(f"\n  Resolution methods:")
        for method, count in method_counts.most_common():
            pct = count / len(self._results) * 100
            print(f"    {method}: {count} ({pct:.0f}%)")

        avg_agreement = sum(r.agreement_score for r in self._results) / len(self._results)
        print(f"\n  Average agreement score: {avg_agreement:.3f}")

        escalated = [r for r in self._results if r.method == "tiebreaker"]
        if escalated:
            print(f"  Escalation rate: {len(escalated)/len(self._results)*100:.1f}%")

        # Label distribution
        labels = Counter(r.final_label for r in self._results)
        print(f"\n  Label distribution:")
        for label, count in labels.most_common():
            print(f"    {label}: {count}")


# --- Demo ---
async def main():
    orch = ConsensusOrchestrator(
        annotators=["w1", "w2", "w3", "w4", "w5"],
        tiebreaker="senior-1",
        annotations_per_task=3,
    )

    tasks = [
        ConsensusTask(f"task-{i:03d}", data=f"sample-data-{i}")
        for i in range(15)
    ]

    start = time.monotonic()
    results = await orch.process_batch(tasks)
    elapsed = time.monotonic() - start

    orch.print_report()
    print(f"\n  Total time: {elapsed:.3f}s")
    print(f"  Throughput: {len(tasks)/elapsed:.1f} tasks/sec")

asyncio.run(main())
```

### What to Talk About

- **Parallel annotation**: All N annotators work simultaneously on the same item. This is faster than sequential and prevents bias (no annotator sees others' labels).
- **Consensus threshold**: 67% means 2 out of 3 must agree. This is a configurable quality lever. Higher thresholds = higher quality but more escalations.
- **Tiebreaker escalation**: When there's no majority, a senior annotator makes the final call. This is the human-in-the-loop fallback that Scale AI uses.
- **Agreement score as a quality signal**: Low agreement across many tasks suggests the labeling guidelines are ambiguous or the task is inherently subjective. This metric feeds back into project design.
- **Cost awareness**: Each escalation costs more (senior annotators are expensive). Mention that production systems track escalation rate as a cost metric.

---

## Key Interview Patterns Cheat Sheet

| Pattern | When to Use | Python Tool |
|---------|------------|-------------|
| Min-heap worker pool | Assign to least-loaded worker | `heapq` + `asyncio.Lock` |
| Semaphore rate limiting | Cap concurrent LLM/API calls | `asyncio.Semaphore(N)` |
| Segmented parallel processing | Split big input, process chunks, reassemble | `asyncio.gather` + sort by index |
| Quality gate branching | Conditional pipeline stages | `if/else` after review stage |
| Consensus with escalation | Multiple annotators + majority vote | `Counter` + tiebreaker fallback |
| Backpressure | Limit in-flight items | `asyncio.Queue(maxsize=N)` or `Semaphore` |
| Worker exclusion | Reviewer ≠ annotator | Filter eligible set before routing |
| Rolling accuracy | Track worker quality over time | List of last N scores, sliding window |

---

## What to Say If Asked "How Would You Scale This?"

1. **Horizontal scaling**: Replace in-process queues with a message broker (Redis Streams, Kafka, SQS). Each worker pool becomes a service.
2. **Persistence**: Add a task state store (Postgres, DynamoDB) so the system can recover from crashes. Currently everything is in-memory.
3. **Monitoring**: Emit structured metrics (Prometheus, Datadog) for throughput, latency percentiles (p50/p95/p99), error rates, and queue depths.
4. **Dynamic worker pools**: Auto-scale worker pools based on queue depth. If annotation queue grows, spin up more annotator processes.
5. **Circuit breakers**: If the LLM API starts failing > 50% of requests, stop sending for 30 seconds and retry. Prevents wasting money on doomed requests.
6. **Golden-set evaluation**: Periodically inject tasks with known-correct answers to measure annotator accuracy in real-time without waiting for review.

---

## Sources

- [Scale AI System Design Interview Guide](https://www.systemdesignhandbook.com/guides/scale-ai-system-design-interview/)
- [Intelligent Task Routing — Keylabs](https://keylabs.ai/blog/intelligent-task-routing-assigning-the-right-labelers-to-the-right-jobs/)
- [Scale AI Data Annotation Guide](https://scale.com/blog/data-annotation-how-to)
- [Async LLM Pipelines in Python](https://dasroot.net/posts/2026/02/async-llm-pipelines-python-bottlenecks/)
- [Python heapq Documentation](https://docs.python.org/3/library/heapq.html)
- [Python asyncio.Queue Documentation](https://docs.python.org/3/library/asyncio-queue.html)
- [How to Build Scalable Data Labeling Workflows](https://www.sapien.io/blog/scale-ai-labeling)
