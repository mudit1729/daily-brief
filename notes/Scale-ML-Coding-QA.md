# ML Coding — Scale AI Interview Prep

**Focus**: Practical implementation, clean code, ML concepts, LLM generation/debugging, data handling, debugging under time pressure

---

## Data Parsing and Transformation

### 1. Read a text, JSONL, or CSV file and produce summary statistics. What corner cases do you guard against before you even touch modeling?

Before computing any statistics, you need to handle the reality that real-world data files are messy. For CSV, use `csv.DictReader` or `pandas.read_csv` with explicit `dtype` mappings, `na_values`, and `encoding='utf-8-sig'` (to strip BOM). For JSONL, wrap each `json.loads(line)` in a try/except so one malformed line does not kill the entire pipeline. For plain text, decide upfront whether you are splitting on newlines, tabs, or some delimiter, and whether blank lines are data or separators. Always check the file size before loading into memory — if it exceeds a threshold, switch to streaming.

Corner cases to guard against: (a) mixed types in a column (strings where you expect floats), (b) duplicate column names in CSV headers, (c) NaN vs. the literal string "nan" vs. empty string vs. `None`, (d) inconsistent line endings (`\r\n` vs `\n`), (e) numeric columns stored with commas or currency symbols, (f) dates in multiple formats within the same column, (g) unicode normalization issues (e.g., accented characters in two different byte representations). Before modeling, compute: row count, column count, null fraction per column, dtype distribution, cardinality of categorical columns, min/max/mean/std of numeric columns, and a sample of unique values per column.

```python
import json, csv, io
from collections import defaultdict

def summarize_jsonl(path: str) -> dict:
    stats = defaultdict(lambda: {"count": 0, "nulls": 0, "types": defaultdict(int)})
    n_rows, n_bad = 0, 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            n_rows += 1
            for key, val in row.items():
                stats[key]["count"] += 1
                stats[key]["types"][type(val).__name__] += 1
                if val is None or val == "":
                    stats[key]["nulls"] += 1
    return {"total_rows": n_rows, "bad_lines": n_bad, "columns": dict(stats)}
```

A good habit is to print a summary report before any downstream processing and fail loudly if null fractions exceed a threshold or if expected columns are missing. This prevents silent data corruption from propagating into model training, where it becomes much harder to debug.

### 2. Given logs from an annotation pipeline, write code to identify malformed rows, missing labels, duplicate IDs, and inconsistent timestamps.

Annotation pipelines produce logs that are often append-only and written by multiple workers, so they accumulate subtle corruption over time. The approach is to make a single streaming pass through the log file, maintaining a set of seen IDs, tracking the last timestamp per annotator, and validating each row against a schema. Malformed rows are those that fail JSON parsing or are missing required keys. Missing labels are rows where the label field is present but null, empty, or not in the allowed label set.

```python
from datetime import datetime
from collections import defaultdict

def audit_annotation_logs(path: str, required_keys: set, valid_labels: set):
    seen_ids = set()
    last_ts = {}  # annotator_id -> last timestamp
    issues = defaultdict(list)

    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                issues["malformed"].append(line_no)
                continue

            # Missing required keys
            missing = required_keys - set(row.keys())
            if missing:
                issues["missing_keys"].append((line_no, missing))
                continue

            # Duplicate IDs
            row_id = row.get("id")
            if row_id in seen_ids:
                issues["duplicate_ids"].append((line_no, row_id))
            seen_ids.add(row_id)

            # Missing or invalid labels
            label = row.get("label")
            if label is None or label == "":
                issues["missing_labels"].append(line_no)
            elif valid_labels and label not in valid_labels:
                issues["invalid_labels"].append((line_no, label))

            # Inconsistent timestamps (non-monotonic per annotator)
            annotator = row.get("annotator_id")
            ts_str = row.get("timestamp")
            if ts_str and annotator:
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if annotator in last_ts and ts < last_ts[annotator]:
                        issues["timestamp_regression"].append(
                            (line_no, annotator, str(last_ts[annotator]), str(ts))
                        )
                    last_ts[annotator] = ts
                except ValueError:
                    issues["bad_timestamp"].append((line_no, ts_str))

    return dict(issues)
```

Timestamp inconsistency is particularly insidious: if annotations are supposed to be sequential but timestamps go backward, it may indicate data was re-uploaded out of order, clocks were unsynchronized, or rows were manually edited. You should distinguish between per-annotator monotonicity (which is expected) and global monotonicity (which may not hold with concurrent annotators). Report the fraction of affected rows for each issue category so you can prioritize which problems to fix first.

### 3. You inherit a Python script that loads a large dataset into memory and crashes. How would you rewrite it to stream, batch, and validate incrementally?

The first step is to identify the memory bottleneck. Usually it is a pattern like `data = json.load(open(path))` or `df = pd.read_csv(path)` that pulls everything into RAM at once. Replace this with a generator-based approach: read line by line (for JSONL) or use `pd.read_csv(path, chunksize=10000)` for CSV. Each chunk is validated, transformed, and either written to an output file or yielded to the next stage of the pipeline. This changes the memory profile from O(N) to O(batch_size).

```python
def stream_and_validate(path: str, batch_size: int = 1024):
    batch = []
    with open(path, "r") as f:
        for line in f:
            try:
                row = json.loads(line.strip())
            except json.JSONDecodeError:
                continue  # or log
            if not is_valid(row):
                continue  # or log
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch

def is_valid(row: dict) -> bool:
    return (
        "id" in row
        and "text" in row
        and isinstance(row["text"], str)
        and len(row["text"]) > 0
    )
```

Beyond streaming, consider: (a) using memory-mapped files (`mmap`) for random access without full load, (b) converting to a columnar format like Parquet or Arrow for downstream processing, (c) using `resource.setrlimit` or monitoring `psutil.virtual_memory()` to fail gracefully before the OOM killer strikes, (d) for PyTorch datasets, implementing an `IterableDataset` rather than a map-style dataset. If the data must be shuffled, use a shuffle buffer (hold N items in memory, randomly replace one when a new item arrives) rather than loading everything to shuffle. The key principle is: never trust that data fits in memory; always design for the streaming case and add random access as an optimization.

### 4. Write a function to group examples by class, compute imbalance ratios, and propose a sampling plan.

Class imbalance is one of the most common issues in real annotation data. Grouping by class and computing ratios gives you the information needed to decide between oversampling minority classes, undersampling majority classes, or using class-weighted losses. The imbalance ratio is typically defined as count(majority) / count(minority), and anything above 10:1 deserves explicit handling.

```python
from collections import Counter
import math

def analyze_imbalance(labels: list) -> dict:
    counts = Counter(labels)
    total = sum(counts.values())
    sorted_classes = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    max_count = sorted_classes[0][1]

    report = {
        "class_counts": dict(sorted_classes),
        "class_fractions": {c: n / total for c, n in sorted_classes},
        "imbalance_ratio": max_count / sorted_classes[-1][1] if sorted_classes[-1][1] > 0 else float("inf"),
    }

    # Propose sampling plan
    # Target: square root of frequency (a common compromise)
    sqrt_counts = {c: math.sqrt(n) for c, n in counts.items()}
    sqrt_total = sum(sqrt_counts.values())
    target_fractions = {c: v / sqrt_total for c, v in sqrt_counts.items()}

    # Compute per-class sampling weights for weighted random sampling
    sampling_weights = {c: target_fractions[c] / (counts[c] / total) for c in counts}

    report["proposed_sampling_weights"] = sampling_weights
    report["effective_samples_per_class"] = {
        c: int(target_fractions[c] * total) for c in counts
    }
    return report


def build_weighted_sampler(labels: list):
    """Returns per-sample weights for torch.utils.data.WeightedRandomSampler."""
    counts = Counter(labels)
    class_weight = {c: 1.0 / n for c, n in counts.items()}
    sample_weights = [class_weight[label] for label in labels]
    return sample_weights
```

The square-root rebalancing strategy is a practical middle ground between uniform sampling (which over-represents rare classes) and natural distribution (which ignores them). For very small minority classes, you may also want to augment the data or synthetically generate examples. Always verify that your sampling plan does not cause the model to never see certain classes during a single epoch. A good practice is to log the actual class distribution seen during training at the end of each epoch, comparing it to the theoretical target.

### 5. Given a list of model predictions and labels, compute precision, recall, F1, confusion matrix, and per-class metrics without using sklearn.

This is a bread-and-butter coding question. The key insight is that everything derives from the confusion matrix, which is just a count of (true_class, predicted_class) pairs. From there, precision for class c is TP_c / (TP_c + FP_c), recall is TP_c / (TP_c + FN_c), and F1 is their harmonic mean. The macro average treats all classes equally; the weighted average weights by support.

```python
from collections import defaultdict

def confusion_matrix(y_true: list, y_pred: list) -> dict:
    """Returns {(true_label, pred_label): count}."""
    cm = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        cm[(t, p)] += 1
    return dict(cm)


def compute_metrics(y_true: list, y_pred: list) -> dict:
    classes = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred)

    # Build matrix as 2D list for display
    n = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    matrix = [[0] * n for _ in range(n)]
    for (t, p), count in cm.items():
        matrix[class_to_idx[t]][class_to_idx[p]] = count

    per_class = {}
    for c in classes:
        tp = cm.get((c, c), 0)
        fp = sum(cm.get((other, c), 0) for other in classes if other != c)
        fn = sum(cm.get((c, other), 0) for other in classes if other != c)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)

        per_class[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    # Macro and weighted averages
    macro_p = sum(m["precision"] for m in per_class.values()) / len(classes)
    macro_r = sum(m["recall"] for m in per_class.values()) / len(classes)
    macro_f1 = sum(m["f1"] for m in per_class.values()) / len(classes)

    total_support = sum(m["support"] for m in per_class.values())
    weighted_p = sum(m["precision"] * m["support"] for m in per_class.values()) / total_support
    weighted_r = sum(m["recall"] * m["support"] for m in per_class.values()) / total_support
    weighted_f1 = sum(m["f1"] * m["support"] for m in per_class.values()) / total_support

    return {
        "confusion_matrix": matrix,
        "class_labels": classes,
        "per_class": per_class,
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "weighted": {"precision": weighted_p, "recall": weighted_r, "f1": weighted_f1},
        "accuracy": sum(cm.get((c, c), 0) for c in classes) / len(y_true),
    }
```

Edge cases to watch: (a) a class appears in predictions but never in ground truth (or vice versa), (b) division by zero when a class has zero TP+FP or zero TP+FN, (c) empty input lists, (d) multiclass vs. multilabel (this code handles multiclass; multilabel requires binarizing per label). In an interview, mention that micro-average precision/recall/F1 are all equal to accuracy in the multiclass single-label setting — this shows you understand the math rather than just the API.

### 6. How would you implement top-k accuracy, calibration bins, or an expected calibration error function from scratch?

Top-k accuracy checks whether the true label is among the k highest-probability predictions. It is a relaxation of top-1 accuracy useful when classes are semantically close (e.g., fine-grained recognition). Calibration measures whether predicted probabilities match empirical frequencies — a model that says "80% confidence" should be correct 80% of the time.

```python
import numpy as np

def top_k_accuracy(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    probs: (N, C) array of predicted probabilities
    labels: (N,) array of integer true class indices
    """
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]  # (N, k)
    correct = np.any(top_k_preds == labels[:, None], axis=1)
    return correct.mean()


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> dict:
    """
    probs: (N, C) array of predicted probabilities
    labels: (N,) array of integer true class indices
    Returns ECE scalar and per-bin details.
    """
    confidences = np.max(probs, axis=1)          # (N,)
    predictions = np.argmax(probs, axis=1)        # (N,)
    accuracies = (predictions == labels)           # (N,) bool

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_info = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            bins_info.append({"range": (lo, hi), "count": 0, "avg_conf": 0, "avg_acc": 0})
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        ece += (n_in_bin / len(labels)) * abs(avg_acc - avg_conf)
        bins_info.append({
            "range": (lo, hi),
            "count": int(n_in_bin),
            "avg_conf": float(avg_conf),
            "avg_acc": float(avg_acc),
        })

    return {"ece": float(ece), "bins": bins_info}
```

Common pitfalls: (a) using `>=` for the lower bin boundary on all bins causes the first bin to include confidence=0 items or miss them entirely — be consistent and handle the edge; (b) ECE is sensitive to the number of bins, so always report which bin count you used; (c) for top-k, using `np.argpartition` instead of `np.argsort` is O(N*C) instead of O(N*C*log C), which matters at scale; (d) calibration of a model can be good on average but terrible per class — always check per-class calibration if any class has low support. Temperature scaling is the simplest post-hoc fix for miscalibration: fit a single scalar T on a held-out set to minimize NLL, then divide logits by T at inference.

---

## Model Debugging and ML Hygiene

### 1. Your training loss goes down but validation mAP is flat. Walk through the first ten things you would check.

This is a classic overfitting signal, but the real answer requires systematically ruling out non-obvious causes. Here are the ten things in order of likelihood and ease of checking: (1) **Learning rate too high or no schedule** — the model memorizes training data but does not generalize; plot the learning rate and try reducing it. (2) **Insufficient regularization** — add or increase dropout, weight decay, or data augmentation. (3) **Train/val distribution mismatch** — compare class distributions, image resolutions, source domains between train and val sets. (4) **Data leakage in training set** — check if near-duplicate images exist across splits, especially if mAP was initially high. (5) **Evaluation code bug** — verify you are running the exact same preprocessing (resize, normalize, padding) at eval time as at train time; a common bug is forgetting to turn off augmentation during eval.

(6) **mAP implementation or IoU threshold mismatch** — confirm the IoU threshold used in mAP computation matches expectations (0.5 vs 0.5:0.95). (7) **Model.eval() not called** — BatchNorm and Dropout behave differently in train vs eval mode; forgetting `model.eval()` can silently destroy validation performance. (8) **Gradient accumulation or batch size issue** — effective batch size might differ between train and val, affecting BatchNorm statistics. (9) **Label noise in training set** — if annotations are noisy, the model fits noise; compare loss distribution per sample to find outliers. (10) **Capacity mismatch** — the model may be too large for the dataset size, or the task may require features the architecture cannot learn (e.g., small objects with too much downsampling). After checking these ten, plot per-class AP to see if the problem is concentrated in specific classes, which narrows the diagnosis further.

### 2. A model predicts one dominant class for almost everything. How do you tell whether the root cause is label skew, loss weighting, bugged targets, or a preprocessing bug?

Start with data-level diagnostics. Compute the class distribution in training labels: if one class is 90%+ of the data, the model has found a trivially good strategy by always predicting the majority class. But label skew alone does not necessarily cause this — a well-configured loss function can handle 10:1 imbalance. So check the loss function: is it unweighted cross-entropy? If so, the gradient signal from the minority class is overwhelmed. Try class-weighted loss or focal loss and see if behavior changes. If the dominant prediction class does not even match the most frequent training class, that rules out label skew as the primary cause.

Next, check for bugged targets. Sample 100 random training examples and manually verify that the label matches the input. Common bugs: (a) labels are off-by-one due to indexing differences (label file is 1-indexed, model expects 0-indexed), (b) a default/fallback label is assigned when parsing fails, inflating one class, (c) for detection, bounding boxes have been incorrectly mapped to class indices. Also check preprocessing: if your normalization, resizing, or augmentation is destroying the signal (e.g., cropping out the object, normalizing with wrong mean/std, converting grayscale images with a color-expecting pipeline), the model sees effectively random inputs and falls back to the prior. A definitive test: overfit on 10 examples. If the model cannot achieve near-zero loss on 10 examples, the bug is in the training loop, loss function, or preprocessing — not in the data distribution.

### 3. You suspect train/val leakage. What concrete tests would you run?

Leakage means information from the validation (or test) set has bled into training, producing artificially high metrics that do not generalize. The first test is an **ID overlap check**: extract unique identifiers (file names, sample IDs, patient IDs, session IDs) from both sets and compute the intersection. Even if raw IDs differ, near-duplicates can leak — for image data, compute perceptual hashes (pHash, dHash) and flag pairs with similarity above a threshold. For tabular data, check if any row in val is an exact or near-exact match to a training row.

The second test is a **temporal leakage check**: if data has a time dimension, verify that all training timestamps precede all validation timestamps. Using random splits on time-series data is a common source of leakage because the model learns temporal patterns that span the split boundary. The third test is a **feature leakage check**: look for features that are proxies for the label (e.g., a feature that is only populated when the label is positive, or a feature derived from future information). Train a simple model (logistic regression) on each feature individually; if any single feature achieves suspiciously high accuracy, investigate its provenance. The fourth test is **performance gap analysis**: if your model achieves 99% on val but 70% on a truly held-out test set, leakage is almost certain. A model that is "too good" on val relative to the task difficulty is a red flag.

### 4. A model works on offline evaluation but fails badly in production. How do you localize whether the issue is feature skew, serving skew, threshold mismatch, or broken post-processing?

This is one of the most common and painful failure modes in ML systems. The systematic approach is to progressively eliminate each hypothesis by inserting logging at every stage of the production pipeline and comparing it against the offline pipeline.

**Feature skew**: Log the raw input features in production and compare their distributions to the offline eval set. Common causes: missing features get filled with different defaults (0 vs NaN vs mean), tokenization or normalization is implemented differently in the serving library vs training code, or image decoding produces different pixel values (libjpeg vs Pillow vs OpenCV handle color space and subsampling differently). Run the production preprocessing on a batch of offline eval inputs and compare outputs byte-for-byte.

**Serving skew**: Ensure the exported model (ONNX, TorchScript, TF SavedModel) produces identical logits to the training-time model on the same inputs. Float precision differences (FP32 vs FP16), operator implementation differences, and dynamic batching/padding can all cause divergence. Run a numerical diff on 100 examples.

**Threshold mismatch**: The confidence threshold chosen during offline eval may have been optimized on the eval set distribution. If production data has a different difficulty distribution, the same threshold yields different precision/recall. Log raw scores in production and plot the score distribution; compare it to the offline distribution.

**Broken post-processing**: NMS parameters, score filtering, label mapping, coordinate denormalization — any of these can differ between offline eval and production. The fastest diagnostic is to dump the raw model output in production and run the offline post-processing on it. If offline post-processing recovers good metrics, the bug is in the production post-processing code.

### 5. How do you distinguish optimization failure from representation failure?

Optimization failure means the model architecture is capable of representing a good solution but the training procedure fails to find it. Representation failure means the architecture fundamentally cannot express the mapping from inputs to outputs. The practical test is the **overfit test**: take a tiny subset (10-100 examples) and train until convergence. If the model cannot drive training loss to near zero on this tiny set, it is an optimization failure — the gradients are not flowing, the learning rate is wrong, there is a bug in the loss computation, or the initialization is pathological. If it can overfit the tiny set but fails to generalize, the issue is likely either regularization, data quality, or insufficient data — but not representation.

To further distinguish: try increasing model capacity (more layers, wider layers). If a larger model suddenly starts learning while the smaller one was flat, the smaller model had a representation failure — it could not express the needed function. If the larger model also fails, the problem is likely in the optimization landscape (bad loss function, poor conditioning, vanishing gradients). Other signals: if training loss oscillates wildly, the learning rate is too high. If training loss decreases very slowly and plateaus early, try a different optimizer (switch from SGD to Adam) or add learning rate warmup. If loss goes to NaN, check for numerical instability (log of zero, division by zero, exploding gradients). Always plot the full training curve, not just the final number — the shape of the curve is diagnostic.

### 6. When does adding more data fail to help, and what would you inspect next?

Adding more data fails to help in several scenarios. First, if the model is **underfitting** — it cannot fit even the current data well, so more data just adds more examples it cannot learn from. Check whether training loss has plateaued at a high value; if so, the model needs more capacity or a better architecture, not more data. Second, if the new data is **redundant** — it comes from the same narrow distribution as existing data. A model with 1M images of cats from the same angle gains little from another 1M similar images. What it needs is data diversity: different angles, lighting, occlusions, domains.

Third, if the **label quality is poor** — adding more noisily-labeled data can actually hurt performance once the noise rate exceeds the model's ability to regularize against it. Inspect annotation agreement rates and per-annotator accuracy. Fourth, if there is a **systematic bias** in the data collection process — e.g., all positive examples share a spurious correlate (all dogs are photographed outdoors, all cats indoors), more data from the same collection process just reinforces the bias. What to inspect next: (a) learning curves — plot performance vs. dataset size; if the curve has flattened, more data of the same type will not help; (b) error analysis — categorize failures into types and see if any type dominates; (c) data diversity metrics — measure coverage of the input space; (d) active learning — use model uncertainty to select which new examples would be most informative. Sometimes the answer is not more data but better data: cleaning labels, balancing classes, or collecting data specifically in the failure modes.

---

## LLM Generation and Evaluation

### 1. Implement a minimal text generation loop with temperature, top-k, top-p, and stop tokens. What bugs show up most often?

The generation loop is autoregressive: at each step, run the model to get logits for the next token, apply sampling controls, sample a token, append it, and repeat until a stop condition is met. The core logic is deceptively simple but full of subtle bugs.

```python
import torch
import torch.nn.functional as F

def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    stop_tokens: list = None,
) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    stop_token_ids = set()
    if stop_tokens:
        for st in stop_tokens:
            stop_token_ids.update(tokenizer.encode(st, add_special_tokens=False))

    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated).logits[:, -1, :]  # (1, vocab_size)

            # Temperature scaling — must come before sampling
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, top_k, dim=-1)
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Remove tokens with cumulative probability above threshold
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                # Scatter back to original indices
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() in stop_token_ids:
                break

            generated = torch.cat([generated, next_token], dim=-1)

    # Decode only the newly generated tokens
    output_ids = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)
```

The most common bugs: (1) **Applying temperature after softmax instead of before** — temperature must scale logits, not probabilities. (2) **Off-by-one in KV cache indexing** — when using a KV cache for efficiency, passing the wrong position IDs or attention mask causes the model to attend to the wrong context. (3) **Not handling the EOS token** — the model may generate EOS but the loop keeps going, producing garbage. (4) **Top-p scatter bug** — the mask must be applied to sorted logits and then scattered back; getting the scatter wrong silently corrupts the distribution. (5) **Temperature of 0** — this should be handled as argmax (greedy), but dividing by 0 crashes. (6) **Stop token tokenization mismatch** — a stop string like "\n\n" may tokenize differently depending on context, so checking for exact token ID matches can miss it. Better to decode the full output at each step and check for the stop string in the decoded text.

### 2. How would you debug repetition, premature truncation, or mode collapse in generation?

**Repetition** (the model outputs the same phrase over and over) is the most common generation pathology. Causes: (a) temperature too low or top-k/top-p too restrictive, pushing the model into a deterministic loop; (b) the model has learned degenerate patterns from training data with repeated boilerplate; (c) attention is collapsing to attend only to recent tokens (a context window issue). Fixes: increase temperature, apply a repetition penalty (scale down logits of tokens that have already appeared), or use frequency/presence penalties. To debug, log the token-level probabilities at each step — if the repeated token has >0.99 probability, the issue is in the model's distribution; if it has 0.3 probability but keeps getting sampled, the issue is sampling luck (increase temperature or switch to beam search with diversity).

**Premature truncation** (output ends too early) is usually caused by: (a) the EOS token appearing in the logits with high probability due to training on short examples, (b) a stop token check that triggers on a false positive (e.g., a period is treated as a stop token), (c) max_tokens set too low. Debug by logging why generation stopped at each call — was it EOS, stop token, or length limit?

**Mode collapse** (all prompts produce nearly identical outputs) indicates the model has lost diversity, often due to fine-tuning on homogeneous data or overfit RLHF. Check the entropy of the output distribution at each step; if entropy is consistently near zero, the model is deterministic regardless of sampling parameters. The fix is usually at the training level (more diverse data, KL penalty against the base model during RLHF, lower learning rate during fine-tuning), not at inference time.

### 3. What is the difference between teacher-forced training metrics and rollout-time behavior?

During teacher-forced training, the model receives the ground-truth previous token as input at each step, regardless of what it would have predicted. This means training loss (perplexity, cross-entropy) is computed under ideal conditions where the model never sees its own mistakes. At rollout time (autoregressive generation), the model feeds its own predictions back as input. If the model makes an error early in the sequence, all subsequent predictions are conditioned on that error, causing **exposure bias** — a compounding divergence from the training distribution.

This discrepancy has several practical implications. First, teacher-forced perplexity can be excellent while generation quality is poor, because the model has never learned to recover from its own errors. Second, metrics like BLEU or ROUGE computed on teacher-forced outputs (by taking argmax at each position independently) will overestimate quality compared to metrics on actual rollouts. Third, sequence-level properties — coherence, factual consistency, length distribution — cannot be assessed from teacher-forced metrics at all, because they depend on the full generated sequence.

Mitigation strategies include: (a) **scheduled sampling** — during training, randomly replace some teacher-forced inputs with the model's own predictions, annealing the probability over time; (b) **sequence-level training** (REINFORCE, PPO) — optimize directly for rollout-time metrics; (c) **evaluation on rollouts** — always report metrics on actual generated sequences, not teacher-forced ones. In practice, for LLMs, the gap is smaller because the models are large enough to be robust to small perturbations, but it still matters for tasks requiring long, structured outputs (code generation, multi-step reasoning).

### 4. Design a lightweight rubric-based evaluator for model outputs. What failure modes would you expect from an LLM-as-judge pipeline?

A rubric-based evaluator scores model outputs along predefined dimensions (e.g., relevance, accuracy, fluency, completeness) using explicit criteria for each score level. The design has three components: (1) a rubric definition with clear criteria per dimension per score level, (2) a prompt template that presents the rubric, the input, and the output to a judge model, and (3) aggregation logic that collects scores and computes summary statistics.

```python
RUBRIC = {
    "accuracy": {
        1: "Contains factual errors or contradicts the source.",
        2: "Partially correct but has significant omissions or minor errors.",
        3: "Mostly accurate with minor issues.",
        4: "Fully accurate and consistent with the source.",
    },
    "relevance": {
        1: "Does not address the question.",
        2: "Partially addresses the question.",
        3: "Addresses the question with some tangential content.",
        4: "Directly and completely addresses the question.",
    },
}

def build_judge_prompt(question: str, answer: str, rubric: dict) -> str:
    rubric_text = ""
    for dim, levels in rubric.items():
        rubric_text += f"\n### {dim}\n"
        for score, desc in levels.items():
            rubric_text += f"  {score}: {desc}\n"

    return f"""Evaluate the following answer based on the rubric below.
Return a JSON object with scores for each dimension.

## Rubric
{rubric_text}

## Question
{question}

## Answer
{answer}

## Your Evaluation (JSON only, no explanation):
"""
```

Failure modes of LLM-as-judge: (1) **Position bias** — the judge favors the first or last option in comparisons; mitigate by randomizing order and averaging. (2) **Verbosity bias** — longer answers score higher regardless of quality; mitigate by adding "penalize unnecessary verbosity" to the rubric. (3) **Self-preference** — an LLM judge favors outputs from the same model family; mitigate by using a different judge model or human calibration. (4) **Score compression** — the judge assigns 3 or 4 to almost everything, making it hard to distinguish quality levels; mitigate by requiring the judge to quote evidence for each score and using binary rubrics (pass/fail) instead of Likert scales. (5) **Inconsistency** — the same input scored twice gets different scores; mitigate by running multiple evaluations and flagging high-variance items for human review. (6) **Rubric gaming** — if the model being evaluated is also an LLM, it can produce outputs that "sound right" to the judge without being correct; mitigate by including verifiable facts in the rubric criteria.

### 5. How would you detect hallucination or factual inconsistency when the task has no single exact ground truth?

When there is no single ground truth, hallucination detection must rely on indirect signals rather than exact string matching. The most practical approaches are:

**Consistency-based detection**: Generate multiple responses to the same prompt (with different random seeds or temperatures) and check agreement. If the model gives contradictory answers across samples, the claims it disagrees on are likely hallucinated — the model is sampling from a high-entropy region of its distribution rather than recalling a learned fact. You can operationalize this by extracting atomic claims from each response and checking the fraction of claims that appear across a majority of samples.

**Entailment-based detection**: Use a natural language inference (NLI) model to check whether claims in the output are entailed by a reference source (retrieved documents, knowledge base entries). Break the output into sentences or claims, retrieve relevant sources for each, and classify each claim as entailed, neutral, or contradicted. Claims that are contradicted or have no supporting source are flagged as potential hallucinations. This does not require a single ground truth — it requires a corpus of facts to check against.

**Self-consistency with retrieval**: Augment the generation pipeline with retrieved evidence and compare the output with and without retrieval. If the model makes a claim without retrieval that contradicts retrieved evidence, it is likely hallucinating. You can also ask the model itself to verify its claims by prompting it to identify which parts of its answer are unsupported by the provided context — this is imperfect but provides a useful first pass.

The fundamental challenge is that hallucination detection without ground truth is inherently probabilistic — you can estimate a confidence that something is hallucinated but cannot be certain. Log all flagged items for human review, and track the false positive rate of your detector to calibrate trust in its outputs.

### 6. What logging would you add to a generation pipeline so you can debug weird outputs after the fact?

Comprehensive logging is essential because generation bugs are often non-reproducible without exact state. At minimum, log the following for every generation call:

**Input-level**: the raw prompt string, the tokenized input IDs, the prompt template name/version, any retrieved context or few-shot examples injected into the prompt, and the generation hyperparameters (temperature, top_k, top_p, max_tokens, stop tokens, random seed).

**Step-level** (optional, expensive but invaluable for debugging): at each autoregressive step, log the top-5 token IDs and their probabilities, the entropy of the distribution, and whether any filtering (top-k, top-p) was applied. This lets you reconstruct exactly why the model chose each token and identify the step where generation went wrong. In production, you might enable this only on a sample or when triggered by anomaly detection.

**Output-level**: the full generated text, the token IDs, the total number of tokens generated, the reason for stopping (max_tokens, EOS, stop token, timeout), wall-clock latency, and any post-processing applied (regex filtering, safety classifiers, format parsing). Also log the model version/checkpoint and the inference server instance ID so you can reproduce the exact computation path.

**Anomaly flags**: automatically flag outputs that are (a) suspiciously short or empty, (b) repetitive (e.g., n-gram repetition ratio above a threshold), (c) contain patterns known to be failure modes (e.g., the model outputting its system prompt), or (d) fail downstream parsing (e.g., expected JSON but got prose). These flags allow you to set up alerting without human review of every output. Store all logs in a structured format (JSONL) with a unique request ID that threads through the entire pipeline so you can correlate generation logs with upstream requests and downstream user feedback.

---

## CV and Tensor Debugging

### 1. Implement IoU / vectorized IoU from scratch and explain the tensor shapes line by line.

Intersection over Union (IoU) measures the overlap between two bounding boxes. For a single pair, it is straightforward: compute the intersection area and divide by the union area. The vectorized version computes IoU between all pairs from two sets of boxes, which is needed for matching predictions to ground truth in detection evaluation.

```python
import numpy as np

def iou_single(box_a, box_b):
    """
    box_a, box_b: [x1, y1, x2, y2] — top-left and bottom-right corners.
    """
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def iou_vectorized(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    boxes_a: (N, 4) — N boxes, each [x1, y1, x2, y2]
    boxes_b: (M, 4) — M boxes, each [x1, y1, x2, y2]
    Returns: (N, M) IoU matrix
    """
    N = boxes_a.shape[0]
    M = boxes_b.shape[0]

    # Expand dims for broadcasting: (N, 1, 4) vs (1, M, 4)
    a = boxes_a[:, None, :]   # (N, 1, 4)
    b = boxes_b[None, :, :]   # (1, M, 4)

    # Intersection coordinates: element-wise max/min broadcast to (N, M)
    inter_x1 = np.maximum(a[..., 0], b[..., 0])  # (N, M)
    inter_y1 = np.maximum(a[..., 1], b[..., 1])  # (N, M)
    inter_x2 = np.minimum(a[..., 2], b[..., 2])  # (N, M)
    inter_y2 = np.minimum(a[..., 3], b[..., 3])  # (N, M)

    # Intersection area: clamp to 0 for non-overlapping boxes
    inter_w = np.maximum(0, inter_x2 - inter_x1)  # (N, M)
    inter_h = np.maximum(0, inter_y2 - inter_y1)  # (N, M)
    inter_area = inter_w * inter_h                  # (N, M)

    # Individual areas: (N,) and (M,)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])  # (N,)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])  # (M,)

    # Union: broadcast (N, 1) + (1, M) - (N, M) -> (N, M)
    union_area = area_a[:, None] + area_b[None, :] - inter_area  # (N, M)

    return inter_area / np.maximum(union_area, 1e-6)  # (N, M)
```

The key broadcasting trick is expanding `boxes_a` to shape (N, 1, 4) and `boxes_b` to (1, M, 4), so all pairwise operations broadcast to (N, M, 4) and then reduce the last dimension via coordinate-wise operations. The `np.maximum(union_area, 1e-6)` prevents division by zero for degenerate boxes with zero area. A common bug is confusing the coordinate convention: some systems use (x_center, y_center, w, h) instead of (x1, y1, x2, y2), and forgetting to convert will produce wrong IoU values without any error.

### 2. Write non-maximum suppression and explain the complexity trade-off.

NMS removes redundant detections by iterating through boxes sorted by confidence: the highest-confidence box is kept, and all boxes with IoU above a threshold are suppressed. This is repeated until no boxes remain.

```python
def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> list:
    """
    boxes: (N, 4) in [x1, y1, x2, y2] format
    scores: (N,) confidence scores
    Returns: list of kept indices
    """
    if len(boxes) == 0:
        return []

    # Sort by score descending
    order = np.argsort(scores)[::-1]

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []

    while len(order) > 0:
        # Pick the box with highest score
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        # Compute IoU of this box with all remaining boxes
        remaining = order[1:]

        inter_x1 = np.maximum(boxes[i, 0], boxes[remaining, 0])
        inter_y1 = np.maximum(boxes[i, 1], boxes[remaining, 1])
        inter_x2 = np.minimum(boxes[i, 2], boxes[remaining, 2])
        inter_y2 = np.minimum(boxes[i, 3], boxes[remaining, 3])

        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        union_area = areas[i] + areas[remaining] - inter_area
        ious = inter_area / np.maximum(union_area, 1e-6)

        # Keep only boxes with IoU below threshold
        mask = ious <= iou_threshold
        order = remaining[mask]

    return keep
```

**Complexity**: The naive implementation is O(N^2) in the worst case — each iteration removes at least one box, and each iteration computes IoU against all remaining boxes. In practice, NMS is fast because (a) many boxes are suppressed in early iterations (high-confidence detections suppress clusters of nearby boxes), and (b) the vectorized IoU computation against remaining boxes is efficient with NumPy. For real-time applications with thousands of boxes, you can use spatial indexing (R-tree, grid-based bucketing) to avoid computing IoU between far-apart boxes, bringing the average case closer to O(N log N). GPU-accelerated NMS (e.g., `torchvision.ops.nms`) pushes the inner loop to CUDA for even better throughput.

**Alternatives**: Soft-NMS decays scores instead of hard suppression, preserving more boxes in crowded scenes. Matrix NMS (used in SOLO) computes the full N x N IoU matrix in one vectorized operation and applies suppression via matrix operations, which is more GPU-friendly than the sequential loop. The trade-off is that Matrix NMS uses O(N^2) memory for the IoU matrix but runs faster on GPU due to better parallelism.

### 3. Given boxes in one coordinate system and images resized with padding, how do you map predictions back correctly?

This is a common source of silent bugs in detection pipelines. When an image is resized with padding (letterboxing), the coordinate transform involves three steps: (1) the original image is scaled to fit within the target size while preserving aspect ratio, (2) padding is added to fill the remaining space, and (3) model predictions are in the padded coordinate system and must be mapped back to the original image.

```python
def map_boxes_to_original(
    boxes: np.ndarray,       # (N, 4) in [x1, y1, x2, y2], in padded image coords
    original_size: tuple,    # (orig_h, orig_w)
    target_size: tuple,      # (target_h, target_w) — the model input size
) -> np.ndarray:
    orig_h, orig_w = original_size
    target_h, target_w = target_size

    # Compute scale factor (same for both dimensions to preserve aspect ratio)
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Compute padding offsets
    pad_x = (target_w - new_w) / 2.0
    pad_y = (target_h - new_h) / 2.0

    # Remove padding offset, then undo scaling
    boxes_orig = boxes.copy()
    boxes_orig[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes_orig[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

    # Clip to image boundaries
    boxes_orig[:, [0, 2]] = np.clip(boxes_orig[:, [0, 2]], 0, orig_w)
    boxes_orig[:, [1, 3]] = np.clip(boxes_orig[:, [1, 3]], 0, orig_h)

    return boxes_orig
```

The most common bugs: (a) applying the inverse transform in the wrong order (scale then un-pad vs. un-pad then scale), (b) using different padding strategies at train and test time (center padding vs. bottom-right padding), (c) integer vs. float rounding in the padding offset — if `pad_x` is computed as an integer during preprocessing but as a float during postprocessing, boxes shift by up to 0.5 pixels, (d) forgetting that some frameworks pad to a multiple of 32 (for stride alignment), adding extra padding beyond what the aspect ratio requires. Always write a round-trip test: take known box coordinates in the original image, apply the forward transform (resize + pad), then apply the inverse transform, and verify you recover the original coordinates to within floating-point tolerance.

### 4. You are told a detector is missing small objects. Which code paths and data transforms do you inspect first?

Small object detection failures are systematic and usually traceable to specific code paths. Inspect in this order:

**(1) Input resolution and downsampling**: If the model's input resolution is 640x640 and the original images are 4000x3000, small objects that occupy 20x20 pixels in the original become 3x3 pixels after resizing — below the detection threshold for most architectures. Check whether multi-scale training/testing is enabled. FPN (Feature Pyramid Network) is designed to handle this, but only if the smallest feature map has enough resolution.

**(2) Anchor or query configuration**: For anchor-based detectors, check whether anchors cover the small object size range. If the smallest anchor is 32x32 but the objects are 16x16, they have no matching anchor during training and are effectively invisible. For anchor-free detectors (FCOS, CenterNet), check the feature level assignment — small objects should be assigned to the highest-resolution feature map (P3, stride 8), not P5 (stride 32).

**(3) Data augmentation**: Random cropping can cut small objects entirely, and if the pipeline does not filter out crops with no remaining objects, the model trains on confusing negative examples. Mosaic augmentation can shrink objects further. Check augmentation parameters and whether minimum object size filters are applied after augmentation. Also verify that bounding boxes are correctly updated after geometric augmentations (flip, rotate, crop).

**(4) Loss function and matching**: During training, if the IoU matching threshold is too high (e.g., 0.7), small objects with slight localization error might not match any prediction, producing no gradient signal. Lowering the threshold or using a distance-based matching criterion (as in FCOS) helps. Also check if a minimum box area filter in the data loader is accidentally removing small ground truth boxes. Finally, **(5) NMS threshold and score threshold**: small objects often have lower confidence scores; if the score threshold is too high, they are filtered before NMS even runs.

### 5. A segmentation mask is shifted relative to the image. How do you isolate whether the bug is in augmentation, resizing, interpolation mode, or coordinate convention?

A shifted mask is a spatial misalignment bug, and the debugging strategy is to isolate each transform stage and verify alignment independently.

**Step 1: Verify raw data**. Before any augmentation, load the raw image and raw mask, overlay them, and visually confirm they are aligned. If they are misaligned in the raw data, the bug is upstream of your code (annotation tool, export format). Check the coordinate convention: some annotation tools use (row, col) while code expects (x, y), effectively transposing the mask.

**Step 2: Disable all augmentation** and check alignment after only loading and resizing. If the mask is now aligned, the bug is in augmentation; if still shifted, it is in resizing. For resizing, the most common bug is using different interpolation modes for image and mask: the image uses bilinear interpolation (which is correct) but the mask also uses bilinear, which smears label boundaries and can shift the effective mask by a pixel. Masks must always use nearest-neighbor interpolation. Also check whether `cv2.resize` vs `PIL.Image.resize` vs `F.interpolate` have different conventions for pixel center alignment (half-pixel offset).

**Step 3: Add augmentations back one at a time**. Spatial augmentations (rotation, affine, elastic) must apply the identical transform to both image and mask. A common bug is applying a random transform to the image, then generating a different random transform for the mask because the random seed was not shared. Check that both transformations use the same random parameters. For horizontal flip, verify the flip axis is correct (flipping along axis=1 for width, not axis=0). For padding, verify that the same padding is applied to both image and mask, and that the mask padding value is the ignore index (typically 255), not 0 (which may be a valid class).

**Step 4: Check coordinate convention at the model output**. If the model outputs at a different resolution than the input (e.g., stride-8 output), the upsampling back to input resolution can introduce a shift if `align_corners` is set incorrectly in `F.interpolate`. With `align_corners=True`, pixels are aligned at corners; with `align_corners=False`, they are aligned at centers. Mixing conventions between training and inference causes a half-pixel shift that becomes visible on small objects. The definitive test: create a synthetic image with a perfectly aligned mask (e.g., a centered square), run it through the full pipeline, and verify the output mask is still centered.

### 6. What are the most common silent bugs in PyTorch training loops?

Silent bugs are errors that do not crash the program or produce NaN but cause the model to train incorrectly, producing subtly wrong results that are hard to trace.

**(1) Forgetting `model.eval()` and `model.train()`**: BatchNorm uses running statistics in eval mode and batch statistics in train mode. Dropout is active in train mode and inactive in eval mode. If you forget `model.eval()` before validation, your metrics include dropout noise and use batch statistics (which are noisy on small val batches). If you forget to switch back to `model.train()` after validation, the model trains without dropout for the rest of the epoch.

**(2) Not zeroing gradients**: `optimizer.zero_grad()` must be called before the backward pass (or at the start of each iteration). If omitted, gradients accumulate across iterations, which is equivalent to training with an increasingly large effective batch size and nonsensical gradients. This is distinct from intentional gradient accumulation, where you call `zero_grad()` every K steps.

**(3) In-place operations breaking autograd**: Operations like `x += 1` or `tensor[idx] = value` can break the computational graph silently or produce wrong gradients. PyTorch sometimes detects this and raises an error, but not always — especially when the in-place operation is on a non-leaf tensor that is not directly needed for the backward pass.

**(4) Data loader workers and random state**: With `num_workers > 0`, each worker is a separate process with its own random state. If your dataset uses `random.random()` or `np.random.random()` for augmentation, all workers may produce identical augmentations because they inherited the same seed at fork time. Fix with a `worker_init_fn` that seeds each worker differently. **(5) Mixed precision pitfalls**: using `torch.cuda.amp.autocast()` without a `GradScaler` can cause gradients to underflow to zero in FP16. Also, loss computation should happen inside the autocast context, but some custom losses with log/exp can lose precision. **(6) Detaching tensors accidentally**: calling `.item()`, `.numpy()`, or `.detach()` on a tensor that needs to be part of the gradient graph silently stops gradient flow. This is correct for logging but wrong if used inside the loss computation. **(7) Learning rate scheduler step timing**: some schedulers expect `scheduler.step()` after each epoch, others after each batch. Calling it at the wrong frequency silently warps the learning rate schedule.

---

## Reasoning Under Constraints

### 1. You cannot change the given interface. How do you still make the code testable, debuggable, and robust?

When the interface is fixed (a common interview constraint), your flexibility lies in the internal structure. Write small, pure helper functions that the main function calls — these helpers are individually testable even if the outer function's signature is fixed. For example, if the interface requires `def process(data: list) -> list`, you can internally call `validate(data)`, `transform(data)`, and `aggregate(data)` as separate functions, each of which can be unit tested independently.

For debuggability, add internal assertions and logging that do not change the interface. Use `assert` for invariants during development (e.g., `assert len(output) == len(input)`) — these serve as executable documentation and catch bugs early. Add optional verbose logging controlled by a module-level flag or environment variable, not by a function parameter (since you cannot change the signature). For robustness, handle edge cases at the boundary: check for empty inputs, None values, and type mismatches at the top of the function, and return sensible defaults or raise clear exceptions. Wrap risky operations in try/except blocks with specific exception types — never use bare `except:` because it swallows bugs.

A good pattern is the "functional core, imperative shell" approach: the fixed interface function is the imperative shell that handles I/O, validation, and error handling, while the core logic is in pure functions that take and return values with no side effects. This separation makes the code testable and debuggable despite the constraint.

### 2. How do you explain assumptions out loud without sounding lost?

The key is to frame assumptions as deliberate engineering decisions rather than uncertainty. Instead of saying "I'm not sure if the input can be empty," say "I'm going to handle the empty input case explicitly — if the list is empty, we return an empty result immediately. Let me know if the expected behavior differs." This shows you are thinking about edge cases, not floundering.

Use a consistent verbal pattern: **state the assumption, justify it briefly, and note what would change if the assumption is wrong**. For example: "I'll assume the input boxes are in [x1, y1, x2, y2] format with x2 > x1. If they're in [cx, cy, w, h] format, I'd add a conversion step here." This demonstrates both awareness and adaptability. Similarly, when you are not sure about an optimal approach, say "There are two ways to handle this — sorting first for O(N log N) or using a hash map for O(N) with more memory. I'll go with the hash map since we're not memory-constrained here. Does that match your preference?"

Avoid long pauses followed by "hmm, I think..." Instead, narrate your thought process in real time: "I need to compute pairwise distances, so the naive approach is O(N^2). Let me think about whether we need all pairs or just nearest neighbors..." This fills silence productively and lets the interviewer redirect you if you are heading down a suboptimal path.

### 3. What unit tests would you write first if the interviewer lets you?

Prioritize tests that catch the bugs most likely to occur in interview-style code and that run instantly. The three categories, in order:

**Edge case tests**: Empty input, single element, all identical elements, input with None/NaN values. These take 30 seconds to write and catch a disproportionate number of bugs. For example, `assert compute_metrics([], []) == expected_empty_result` immediately validates your zero-division handling.

**Known-answer tests**: Construct a small input where you can compute the expected output by hand. For IoU: two identical boxes should give IoU=1.0, two non-overlapping boxes should give 0.0, two boxes with exactly 50% overlap should give a value you can verify manually. For metrics: use a 2x2 confusion matrix where precision and recall are easy to compute mentally (e.g., 3 TP, 1 FP, 1 FN gives precision=0.75, recall=0.75).

**Invariant tests / property tests**: These check structural properties without computing exact answers. For NMS: the output should be a subset of the input indices. The number of output boxes should be <= the number of input boxes. All output boxes should have pairwise IoU <= the threshold. For sorting-based algorithms: the output should be a permutation of the input. These tests are powerful because they catch a wide class of bugs without requiring you to pre-compute exact answers.

```python
def test_iou_identical():
    box = [0, 0, 10, 10]
    assert abs(iou_single(box, box) - 1.0) < 1e-6

def test_iou_no_overlap():
    assert iou_single([0, 0, 5, 5], [10, 10, 20, 20]) == 0.0

def test_nms_preserves_best():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]])
    scores = np.array([0.9, 0.8])
    kept = nms(boxes, scores, iou_threshold=0.5)
    assert kept[0] == 0  # highest score is always kept
    assert len(kept) <= len(scores)

def test_metrics_empty():
    result = compute_metrics([], [])
    # Should not crash; should return something sensible
```

### 4. What would you log in a live coding round to make debugging faster?

In an interview setting, you do not have a debugger, so print statements are your primary tool. The goal is to make each print statement maximally informative so you need fewer debug cycles.

**Log intermediate shapes**: For any tensor or array computation, print `.shape` at each step. Most bugs in numerical code are shape mismatches, and seeing the shapes immediately localizes the error. `print(f"boxes: {boxes.shape}, scores: {scores.shape}")` takes 5 seconds and saves minutes.

**Log boundary values**: Print the first and last element, the min, the max, and whether any values are NaN or infinite. `print(f"scores: min={scores.min():.4f}, max={scores.max():.4f}, nans={np.isnan(scores).sum()}")`. This catches normalization bugs, off-by-one errors, and numerical instability.

**Log at decision points**: Before and after filtering, sorting, or conditional logic, print the count of items. "Had 100 boxes, after score filter: 23, after NMS: 7." If one of these numbers is unexpected (e.g., 0 after filtering), you immediately know where the bug is.

**Use labeled prints**: Always include a label so you know which print produced which output. `print(f"[IoU] inter_area: {inter_area}")` is much faster to scan than a bare number. If running inside a loop, include the iteration index: `print(f"[NMS iter {i}] remaining: {len(order)}")`. In an interview, resist the urge to add many print statements at once — add one, run, interpret, then add the next. This disciplined approach is faster than scattershot logging and shows the interviewer a systematic debugging methodology.

### 5. When do you choose clarity over abstraction in an interview solution?

Almost always. Interview code is read once (by the interviewer, in real time) and never maintained, so the priorities are reversed compared to production code. Clarity means: flat control flow (avoid deep nesting), descriptive variable names (even if longer), explicit operations (avoid clever one-liners that require mental unpacking), and comments at non-obvious steps. A five-line explicit loop is better than a two-line nested list comprehension if the interviewer can follow the loop without pausing.

Choose abstraction only when it **reduces** the amount of code the interviewer needs to read, not when it demonstrates your OOP knowledge. Extract a helper function if: (a) you need the same logic in two places (DRY), (b) the helper has a clear name that communicates intent (e.g., `compute_iou` called from `nms`), or (c) the main function is getting so long that the interviewer loses the thread. Do not create classes, inheritance hierarchies, or design patterns in a 45-minute coding interview unless the problem specifically calls for it. Similarly, avoid premature generalization — do not write `def process(data, mode="train", loss_fn=None, augment=True, ...)` when the problem only asks for the training case.

The exception is when the interviewer explicitly asks for extensible or production-quality code. In that case, demonstrate that you know how to abstract, but keep it to one level: a function that calls helpers, not a framework. The test is whether a reader can understand the code top-to-bottom in one pass without jumping between definitions. If they can, your abstraction level is right. If they need to hold three class definitions in their head simultaneously, you have over-abstracted for the interview context.
