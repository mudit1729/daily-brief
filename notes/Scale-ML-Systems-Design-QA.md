# ML Systems Design — Scale AI Interview Prep

**Focus**: Design a continually improving, embedding-based near-duplicate detection service

**This is the most important prep area. Over-prepare here.**

---

## The Best Opening Move — Requirements to Ask First

### 1. What exactly counts as a near-duplicate here?

This is the single most important clarifying question because it determines every downstream design choice. Near-duplicate can mean pixel-level near-identity (e.g., re-encoded JPEGs, crops, watermark additions), semantic near-identity (paraphrased text, same scene from slightly different angles), or functional equivalence (two images that teach a model the same thing). Each definition implies a different similarity function and threshold regime.

At Scale, the most likely answer is functional equivalence for training data quality: two items are near-duplicates if including both in a training set provides negligible marginal information gain. This is a harder, fuzzier definition than bitwise similarity, and it means our system must support tunable, context-dependent thresholds rather than a single bright line.

Push the interviewer to commit to a definition because it shapes whether you need a simple perceptual hash, a learned embedding, or a full retrieval-plus-classifier pipeline. If they say "you decide," propose a tiered system: exact duplicates caught by hash, near-duplicates caught by embedding similarity, and borderline cases routed to human review. This shows you understand the spectrum and aren't over-engineering or under-engineering.

Also clarify whether "near-duplicate" is symmetric and transitive. Symmetry is usually assumed, but transitivity is dangerous — A~B and B~C does not necessarily mean A~C when similarity is threshold-based. This matters for clustering later.

### 2. Is the service for images, text, multimodal items, or all?

Scale handles images (bounding boxes, segmentation), text (RLHF, instruction tuning), video, audio, and 3D point clouds. Ask which modalities are in scope because it determines the embedding model, dimensionality, distance metric, and index structure.

For images, models like CLIP ViT-L/14, DINOv2, or SigLIP produce 768-1024d embeddings that capture semantic content well. For text, sentence-transformers (e.g., E5-large, GTE, BGE) in 384-1024d are standard. For multimodal, CLIP-family models give a shared embedding space but may sacrifice within-modality discrimination.

If the answer is "all modalities," propose starting with the highest-volume modality (likely images for AV/robotics or text for RLHF) and designing the architecture to be modality-agnostic — the embedding is just a float vector to the index layer. Modality-specific logic lives in the encoder and threshold calibration, not in the retrieval infrastructure.

Also ask about item granularity: for video, is a "document" a frame, a clip, or an entire video? For text, is it a sentence, a paragraph, or a full conversation? This determines corpus size and embedding generation cost.

### 3. Do we need online blocking at ingest, offline cleanup, or both?

Online blocking means every new item is checked against the corpus at write time and flagged/rejected before entering the pipeline. This requires low latency (~10-50ms per query) and high availability. Offline cleanup means periodic batch jobs scan the corpus and cluster/remove duplicates. The operational requirements are radically different.

Online blocking is critical if duplicates cause immediate downstream harm — e.g., if a labeling queue sends the same image to multiple annotators, wasting budget. Offline cleanup is appropriate if duplicates are tolerable short-term but degrade model training quality over time.

The best answer for Scale is both: a fast online gate that catches obvious duplicates at ingest (high-confidence, low-latency) plus a deeper offline pipeline that runs more expensive comparisons, builds full duplicate clusters, and feeds hard cases to human review. The online path uses a pre-built ANN index with conservative thresholds (high precision, moderate recall). The offline path can afford exhaustive search, transitive closure, and classifier-based re-ranking.

This dual-mode design also provides a natural fallback: if the online service is degraded, items flow through unblocked and the offline pipeline catches them later. Design for graceful degradation from the start.

### 4. Scale assumptions: QPS, corpus size, embedding dim, latency budget?

Propose concrete numbers and let the interviewer adjust. For a Scale-like platform: corpus size 100M-1B items, growing at 1-10M items/day. Embedding dimensionality 256-1024 (propose 768 as a default for CLIP-class models). Online query latency budget: p50 < 10ms, p99 < 50ms for the similarity search component (excluding embedding generation). Offline throughput: process the full corpus in < 24 hours for a complete re-index.

Memory math: 1B items x 768 dims x 4 bytes/float = ~3TB raw embeddings. With PQ compression (e.g., 64 bytes/vector), that drops to ~64GB, which fits in RAM on a single large instance or a small cluster. This is important because it determines whether you need a distributed index (Milvus, Pinecone, Weaviate) or can run a single-node FAISS/ScaNN instance.

QPS depends on the online/offline split. Online ingest blocking: 100-10K QPS depending on pipeline throughput. Offline batch: throughput-optimized, not latency-optimized. The QPS number matters because HNSW at 1B scale with 768d vectors can serve ~1-5K QPS per node with p99 < 10ms, so you can estimate cluster size.

### 5. False-positive cost vs false-negative cost?

A false positive means declaring two items as duplicates when they are meaningfully different — this removes useful training data and reduces dataset diversity. A false negative means missing a true duplicate — this wastes labeling budget and may degrade model quality through redundancy. The relative cost determines your operating point on the precision-recall curve.

For Scale's data quality use case, false positives are generally more costly than false negatives. Removing a genuinely unique edge case from a training set (especially for safety-critical AV perception) can cause silent model regression. Missing a duplicate just means slightly redundant labeling — wasteful but recoverable.

This asymmetry suggests operating at high precision (>0.95) even at the cost of moderate recall (0.7-0.85) for the online blocking path. The offline cleanup path can afford lower precision because it routes uncertain pairs to human review rather than auto-deleting. Communicate this trade-off explicitly — it shows you think about the business impact of ML system errors, not just metric optimization.

In some domains the cost structure flips: for legal/compliance dedup (e.g., removing PII-containing duplicates), false negatives are catastrophic. Ask the interviewer which regime applies.

### 6. Cluster duplicates, return nearest neighbors, or binary yes/no?

The output format shapes the API contract and downstream consumption. A binary yes/no per item is simplest but loses information. Nearest-neighbor retrieval (return top-k most similar items with scores) is more flexible — the consumer can threshold, rank, or present for review. Full clustering (group all near-duplicates into equivalence classes) is most powerful but computationally expensive and raises transitivity issues.

Propose a layered approach: the online API returns top-k neighbors with similarity scores and a recommended binary decision (duplicate/not). The offline pipeline produces full clusters using connected-components or hierarchical agglomerative clustering on the similarity graph. Downstream consumers choose the output format they need.

For the clustering output, be prepared to discuss cluster quality metrics: cluster purity, coverage, and the problem of "mega-clusters" where transitive chaining connects unrelated items through a chain of borderline similarities. You'll need cluster size caps or single-linkage avoidance to prevent this.

### 7. What human feedback is available?

This question unlocks the "continual improvement" part of the prompt. Ask what labeling budget exists for duplicate review, whether annotators are already flagging duplicates organically (e.g., marking "I've seen this before" in the annotation tool), and whether there's a feedback loop from downstream model training (e.g., data influence analysis showing which training examples are redundant).

At Scale, the most natural feedback source is the annotation workforce itself. If annotators encounter items they've seen before, that signal can be captured passively. More actively, a dedicated review queue can present candidate pairs for binary judgment (duplicate/not-duplicate). The key constraint is that human review is expensive, so the system must be smart about which pairs to surface — prioritize pairs near the decision boundary, pairs from underrepresented slices, and pairs where the model disagrees with rule-based heuristics.

Also ask about feedback latency: can human labels flow back into the system within hours (enabling daily threshold recalibration) or is the turnaround weeks (limiting improvement to periodic retraining)? This determines whether you design for streaming feedback or batch retraining.

### 8. How often does the embedding model change?

Model changes are the most disruptive event in an embedding-based system. A new model means all existing embeddings are in an incompatible vector space, so the entire corpus must be re-embedded and the index rebuilt. At 1B items, re-embedding with a ViT-L model at ~100 items/sec/GPU takes ~100K GPU-hours — nontrivial cost and time.

Ask how frequently the team expects to ship new embedding models. Monthly? Quarterly? Annually? If monthly, you need a parallel embedding pipeline that can re-embed the corpus in the background while the old index continues serving. If annually, a maintenance window approach may suffice.

Also clarify whether "model change" means a full architecture change or fine-tuning on the same architecture. Fine-tuning may produce embeddings that are somewhat compatible with the old space (especially with techniques like alignment loss), reducing the urgency of full re-indexing. But in general, treat any model change as requiring full re-embedding — partial compatibility is fragile.

Design the system with explicit embedding version IDs. Every vector in the index is tagged with its model version. Queries specify which version to search against. During transitions, you run dual indexes and migrate gradually. This is a key architectural decision that shows maturity.

### 9. Privacy/tenant isolation constraints?

Scale serves multiple customers (tenants) whose data must never cross-contaminate. A near-duplicate in Customer A's data is completely irrelevant to Customer B, and cross-tenant similarity search would be a privacy violation.

Ask whether the service is single-tenant (each customer gets their own deployment) or multi-tenant (shared infrastructure with logical isolation). Multi-tenant is more cost-efficient but requires strict partition keys in the index, query-time filtering, and audit logging.

For a multi-tenant design, every embedding is tagged with a tenant ID, and every query is scoped to a single tenant. The ANN index can be partitioned per-tenant (separate HNSW graphs) or shared with mandatory metadata filtering. Per-tenant partitions are safer but waste memory for small tenants. Shared indexes with filtering are more efficient but require careful engineering to prevent information leakage (e.g., ANN traversal must not reveal the existence of vectors from other tenants).

Also consider data residency: some tenants may require their data to stay in specific regions. This constrains index placement and replication topology.

---

## Core Design Questions

### 1. Sketch end-to-end architecture from ingest to decision to feedback loop

**Ingest path (online):** New item arrives at the ingest API. First, compute a perceptual hash (pHash for images, MinHash for text) for exact/near-exact dedup — this is cheap and catches the easy cases in <1ms. If the hash matches, short-circuit with a duplicate decision. Otherwise, generate the embedding using the current model version (either inline if latency allows, or via a pre-computed embedding service). Query the ANN index with the embedding, retrieving top-k neighbors (k=10-20) with similarity scores. Apply the calibrated threshold: if max similarity > threshold, flag as duplicate. If similarity is in the "gray zone" (between auto-reject and auto-accept thresholds), route to human review queue. Write the embedding to the index regardless (with a "pending review" flag if in the gray zone).

**Offline pipeline:** A daily/hourly batch job processes recently ingested items and runs deeper analysis. This includes: (a) exhaustive k-NN search with larger k, (b) transitive closure to build duplicate clusters, (c) cluster quality checks (size caps, diameter checks), (d) cross-referencing with human feedback labels to recalibrate thresholds. Output: updated duplicate clusters, updated threshold parameters, flagged items for human review.

**Feedback loop:** Human review labels (duplicate/not-duplicate on presented pairs) flow into a labeled dataset. This dataset is used for: (a) threshold recalibration (adjust per-slice decision boundaries), (b) hard-negative mining for embedding model fine-tuning, (c) evaluation set maintenance. The feedback loop runs on a weekly cadence for threshold updates and monthly/quarterly for model retraining.

**Component breakdown:** Embedding service (GPU fleet, model versioning) -> ANN index service (FAISS/ScaNN, sharded by tenant) -> Decision service (threshold lookup, business rules) -> Review queue service (sampling, prioritization) -> Feedback ingestion service (label processing, dataset management) -> Monitoring/observability stack.

### 2. Which embedding model to start with for text/image/multimodal and why

**Images:** Start with DINOv2 ViT-L/14 (768d) or SigLIP ViT-SO400M (1152d, can PCA to 768). DINOv2 is preferred for near-duplicate detection because it was trained with self-supervised objectives that emphasize visual similarity rather than text-image alignment. CLIP-family models are biased toward semantic similarity (a photo of a dog and a cartoon of a dog are "similar"), which may not match the near-duplicate definition. DINOv2 better captures visual/structural similarity.

**Text:** Start with E5-large-v2 or GTE-large (1024d). These instruction-tuned sentence embeddings outperform older sentence-transformers on retrieval benchmarks. For near-duplicate detection specifically, you want embeddings that distinguish paraphrases from genuinely different content, so models trained with hard-negative mining (like E5) are preferred. Alternatively, if compute budget allows, use the embedding from a fine-tuned encoder (e.g., a BERT-large model fine-tuned on Scale's own duplicate/not-duplicate pairs).

**Multimodal:** If cross-modal dedup is needed (e.g., detecting that an image and its textual description are duplicates of another image-text pair), use SigLIP or CLIP to project both modalities into a shared space. But be cautious: shared-space embeddings sacrifice within-modality discrimination. A better approach is modality-specific embeddings with a late fusion layer that combines similarity scores from each modality.

**Why not train from scratch?** The initial version should use off-the-shelf models to ship quickly and establish baselines. Fine-tuning comes in iteration 2, using the human feedback labels collected from V1. The embedding model is the most impactful component to improve, but it's also the most expensive to change, so get the infrastructure right first with a good-enough pretrained model.

### 3. One universal embedding space vs modality-specific towers + late fusion

**Universal space (e.g., CLIP):** Pros — simpler infrastructure (one index, one distance computation), enables cross-modal dedup. Cons — lower within-modality discrimination, forced to use the same dimensionality and distance metric for all modalities, harder to fine-tune for modality-specific notions of similarity.

**Modality-specific towers + late fusion:** Pros — best-in-class models per modality, can use different dimensionalities and metrics, can fine-tune independently. Cons — more complex infrastructure (separate indexes per modality, fusion logic), cannot do cross-modal dedup without additional alignment.

**Recommendation:** Start with modality-specific towers. Near-duplicate detection is fundamentally a within-modality problem (image vs image, text vs text). Cross-modal dedup is a different problem (and much rarer). Use DINOv2 for images, E5 for text, and keep the door open for a shared space later. The index layer doesn't care about modality — it just stores and retrieves float vectors. The modality-specific logic lives in the encoder and threshold configuration.

Late fusion for multimodal items (e.g., an image-text pair): compute image similarity and text similarity independently, then combine with a learned or hand-tuned weighting. This is more interpretable and easier to debug than a single fused embedding. You can also use the disagreement between modalities as a signal — if images are similar but text is different, it's probably not a duplicate.

### 4. How to normalize, version, store embeddings

**Normalization:** L2-normalize all embeddings to unit vectors. This converts cosine similarity to inner product (dot product), which is computationally cheaper and supported by all ANN libraries. L2 normalization also makes the distance metric invariant to embedding magnitude, which is important when comparing embeddings from different model checkpoints or input lengths.

**Versioning:** Every embedding is stored with metadata: `{item_id, tenant_id, model_version, modality, embedding_vector, created_at, hash}`. The model_version field is critical — it determines which index partition to search and enables gradual migration between model versions. Use a naming convention like `dinov2-vit-l14-v2.3` that encodes architecture and fine-tuning version.

**Storage:** Hot storage: the ANN index (in-memory, serves queries). Warm storage: embeddings in a columnar store (Parquet files in S3 or a vector database like Milvus/Qdrant) for re-indexing and offline analysis. Cold storage: raw items (images, text) in blob storage, linked by item_id. Never store only the embedding — always retain the ability to re-embed from raw data when the model changes.

**Practical considerations:** At 768d x 4 bytes x 1B items = 3TB raw embeddings. With PQ compression (64 bytes/vector), the index fits in ~64GB RAM. Store raw embeddings separately from the compressed index — raw embeddings are needed for re-ranking and re-indexing, compressed embeddings are for ANN search. Use memory-mapped files for the index to handle datasets that exceed RAM with graceful degradation.

### 5. ANN index choice: HNSW vs IVF-PQ vs flat vs hybrid — why

**Flat (brute force):** Exact k-NN, no approximation. O(n) per query. Only viable for corpora < 1M items or offline batch jobs where latency doesn't matter. Use as the gold standard for evaluating ANN recall.

**IVF-PQ (Inverted File Index with Product Quantization):** Partitions the space into Voronoi cells (IVF) and compresses vectors within each cell (PQ). Very memory-efficient — 64 bytes/vector is typical. Good for large corpora (1B+) where memory is the bottleneck. Downsides: requires training (k-means on a sample), lower recall than HNSW at the same query latency, and recall degrades at distribution boundaries.

**HNSW (Hierarchical Navigable Small World):** Graph-based index with excellent recall-latency tradeoffs. At 1M items, HNSW gives >0.99 recall with <1ms query time. At 100M items, still >0.95 recall at ~5-10ms. Downsides: high memory (stores full vectors + graph edges, ~1.5-2x raw vector size), expensive to build (hours for 100M items), and adding new items is O(log n) per insert but can cause write amplification.

**Recommendation for this system:** Use HNSW for the online serving path (corpora up to ~100M per tenant partition) and IVF-PQ for the offline batch path and for very large tenants (1B+). Specifically, FAISS's `HNSW32` with M=32, efConstruction=200, efSearch=64 gives >0.95 recall@10 with p99 < 10ms at 100M scale. For the offline path, use `IVF4096,PQ64` with nprobe=64 for higher throughput at slightly lower recall.

**Hybrid approach:** Use HNSW for the primary search and flat re-ranking on the top-100 candidates with uncompressed vectors. This gives HNSW's speed with flat's precision on the final candidate set. The re-ranking step adds ~1-2ms and significantly improves precision.

### 6. How to choose k, distance metric, retrieval cutoffs

**Distance metric:** Use cosine similarity (implemented as inner product on L2-normalized vectors). Cosine is the standard for learned embeddings because embedding magnitude is often an artifact of input length or model architecture, not semantic content. Euclidean distance on normalized vectors is monotonically related to cosine similarity (d_euclidean = sqrt(2 - 2*cos_sim)), so it's equivalent — but inner product is faster to compute.

**k (number of neighbors to retrieve):** For online blocking, k=10-20 is sufficient. You only need to find the closest match — k>1 provides robustness against ANN recall loss and enables cluster-aware decisions (if 5 of the top 10 neighbors are from the same cluster, confidence is higher). For offline clustering, k=50-100 to build a denser similarity graph for connected-components analysis.

**Retrieval cutoffs:** Don't use a fixed top-k — use a similarity threshold. Return all neighbors with cosine similarity > tau, where tau is calibrated per (modality, domain, tenant) slice. Typical tau values: 0.90-0.95 for images (DINOv2), 0.85-0.92 for text (E5). But always cap at k_max=100 to bound compute.

**Two-threshold system:** Define tau_high (auto-duplicate, e.g., cosine > 0.95) and tau_low (auto-not-duplicate, e.g., cosine < 0.85). Items with max similarity in [tau_low, tau_high] go to human review. This "gray zone" approach maximizes precision of automated decisions while concentrating human effort where it matters most.

### 7. Calibrating thresholds for stable near-duplicate meaning across slices

Threshold calibration is one of the hardest practical problems in this system. A cosine similarity of 0.92 might mean "definitely duplicate" for natural photographs but "maybe duplicate" for text with domain-specific jargon. The similarity distribution varies dramatically across data slices (modality, domain, source, language, image resolution, text length).

**Approach 1 — Per-slice threshold calibration:** Partition the corpus into slices (e.g., AV camera images, satellite imagery, English RLHF text, Japanese RLHF text). For each slice, collect a small labeled set of duplicate/not-duplicate pairs (100-500 pairs). Fit a logistic regression on cosine similarity to predict duplicate probability. Set the threshold at the operating point that achieves the desired precision (e.g., 0.95 precision). Store thresholds in a configuration service keyed by slice ID.

**Approach 2 — Score normalization:** Instead of calibrating thresholds per slice, normalize the similarity scores to a common scale. For each slice, compute the empirical CDF of pairwise similarities on a random sample. Convert raw cosine similarity to a percentile (e.g., "this pair is more similar than 99.5% of random pairs in this slice"). Apply a universal threshold on the percentile (e.g., >99th percentile = duplicate). This is more robust to distribution shifts but requires maintaining per-slice statistics.

**Approach 3 — Learned re-ranker:** Train a lightweight binary classifier (logistic regression or small MLP) on features: [cosine_similarity, modality, domain, embedding_magnitude_ratio, text_length_ratio, etc.]. This classifier outputs a calibrated duplicate probability. It's more expressive than a single threshold and naturally handles slice-dependent behavior. Retrain weekly on accumulated human labels.

Use Approach 3 for production and Approach 1 as a fallback/interpretability tool. Monitor threshold stability — if the optimal threshold for a slice drifts by >0.02 over a week, alert and investigate (possible distribution shift in incoming data).

### 8. Metadata filters before/after ANN search

**Before (pre-filtering):** Apply metadata filters (tenant_id, modality, date range, source) before ANN search. This reduces the search space and ensures results are relevant. Most vector databases support pre-filtered search natively (e.g., Milvus, Qdrant). In FAISS, implement pre-filtering by maintaining separate indexes per partition or using IDSelector.

**After (post-filtering):** Apply metadata filters after ANN retrieval. Simpler to implement but wasteful — you may retrieve k=20 neighbors and then filter all of them away, returning zero results. This requires over-fetching (retrieve k=200, filter, return top-20) which is slower.

**Recommendation:** Use pre-filtering for mandatory filters (tenant_id — never search across tenants) and post-filtering for optional filters (date range, source). Partition the index by tenant_id (separate HNSW graphs per tenant) and apply other filters on the retrieved candidate set. This gives the best balance of correctness, efficiency, and flexibility.

**Important nuance:** Pre-filtering can degrade ANN recall because it constrains the graph traversal. If a tenant has only 10K items, a dedicated HNSW graph may be too small for optimal connectivity. Solution: for small tenants (<100K items), use flat search instead of HNSW — it's fast enough at that scale and gives exact results.

### 9. Avoiding expensive similarity search against entire corpus on every write

At 1B items with 10K writes/second, naive per-write ANN search is ~10K QPS against a 1B-item index. This is feasible with HNSW (a well-tuned HNSW serving ~5K QPS per node at 100M scale would need ~10 nodes per 1B-item tenant), but expensive. Strategies to reduce cost:

**Tiered dedup:** First, check perceptual hash (pHash, aHash for images; MinHash/SimHash for text) against a hash table. This catches exact and near-exact duplicates in O(1) with <1ms latency. Only items that pass the hash check proceed to embedding-based ANN search. In practice, 30-50% of duplicates are caught by hashing alone, halving the ANN QPS.

**Temporal windowing:** Most duplicates arrive within a short time window (same data upload batch, same crawl session). Check new items against only the last N days of ingested items first (a smaller, recent-items index). If no match, optionally defer full-corpus search to the offline pipeline. This trades recall for latency/cost.

**Bloom filter pre-check:** Maintain a locality-sensitive hashing (LSH) Bloom filter. If the LSH hash of a new item doesn't match any bucket, it's definitely not a duplicate — skip ANN search entirely. This adds a fast rejection path. False positive rate is tunable; even a 50% pass-through rate halves QPS.

**Batch coalescing:** Buffer incoming items and search them in batches (every 100ms or every 100 items). Batch ANN search is significantly faster than individual queries because of vectorized distance computation and shared graph traversal. FAISS batch search can achieve 10-50x throughput improvement over individual queries.

### 10. Supporting both low-latency online checks and deep offline cleanup

**Online path:** Optimized for latency. Uses HNSW index in RAM, k=10, conservative threshold (high precision). Returns a binary duplicate/not-duplicate decision within 10-50ms. Accepts some recall loss. Items flagged as duplicates are either rejected (hard block) or tagged for downstream filtering (soft flag). The online index is updated incrementally as new items arrive (HNSW supports online insertion).

**Offline path:** Optimized for thoroughness. Runs as a batch job (hourly/daily). Uses larger k (50-100), multiple passes with different thresholds, transitive closure for cluster building, and classifier-based re-ranking. Can afford 10-100x the per-item compute of the online path. Catches duplicates missed by the online path (lower-confidence matches, cross-batch duplicates, items that arrived during index rebuild).

**Consistency model:** The online and offline paths may disagree. The offline path is authoritative — if it determines an item is a duplicate, it overrides the online decision. But the online path's decisions are not reversed retroactively (items already sent for annotation are not recalled). Instead, duplicate labels are applied to the item's metadata for future filtering.

**Shared infrastructure:** Both paths share the same embedding service and the same embedding store. They differ in index configuration (HNSW vs IVF-PQ), search parameters (k, threshold, nprobe), and post-processing logic. This shared-nothing design for the index layer allows independent scaling and deployment.

---

## Follow-up Pushes (Edge Cases)

### 1. Preventing false matches from common templates/backgrounds/watermarks

Common visual templates (e.g., a standard form layout, a branded background, a dashboard screenshot) produce high embedding similarity even when the actual content differs. Similarly, watermarked images may cluster by watermark rather than content. This is a major source of false positives.

**Solution 1 — Template subtraction:** Maintain a library of known templates/backgrounds. Before embedding, detect if the image matches a template (using template matching or a classifier) and either mask the template region or subtract the template embedding from the image embedding. This is domain-specific but very effective for known templates.

**Solution 2 — Attention-based embeddings:** Use embedding models that attend to content regions rather than backgrounds. DINOv2's attention maps naturally focus on salient objects. Extract embeddings from the CLS token or from attention-weighted patches excluding low-attention (background) regions. This reduces template influence without explicitly modeling templates.

**Solution 3 — Hard-negative training:** Fine-tune the embedding model with hard negatives that include template-similar-but-content-different pairs. The model learns to ignore shared template features and focus on discriminative content. This requires a labeled set of "same template, different content" pairs, which can be mined from the false-positive log.

**For text:** Common boilerplate (legal disclaimers, standard greetings, template responses) causes the same problem. Use TF-IDF weighting on embedding tokens or train with hard negatives that share boilerplate but differ in substance.

### 2. Handling embedding model drift and old thresholds

When you fine-tune or update the embedding model, the similarity distribution shifts. A threshold of 0.92 that gave 95% precision with model V1 might give 80% precision with model V2 (if V2 compresses the similarity range) or 99.5% precision (if V2 expands it). Old thresholds are invalid for new embeddings.

**Mitigation:** Never reuse thresholds across model versions. When deploying a new model, re-run threshold calibration on the labeled evaluation set. Automate this: the model deployment pipeline includes a threshold calibration step that computes optimal thresholds per slice and publishes them to the configuration service.

**Monitoring:** Track the similarity score distribution for each model version. Alert if the distribution mean or variance shifts by more than a configurable delta. Maintain a "threshold freshness" metric — the time since thresholds were last calibrated against current data. If threshold freshness > 7 days, trigger re-calibration.

**Backward compatibility:** During model transitions, some items have V1 embeddings and some have V2. You cannot compare across versions. Either re-embed all items (expensive but clean) or maintain dual indexes and merge results. The dual-index approach requires careful handling: search both indexes, combine candidates, re-rank with uncompressed embeddings from the new model (re-embed old candidates on the fly for a small candidate set).

### 3. Backfilling a new model version safely

Re-embedding 1B items with a ViT-L model at ~200 items/sec/GPU requires ~5M GPU-seconds, or ~1400 GPU-hours. At $1/GPU-hour (spot), that's ~$1,400 per backfill — manageable but not trivial. With 100 GPUs, it takes ~14 hours.

**Safe backfill process:** (1) Deploy new embedding model to a shadow fleet. (2) Start backfill job, writing V2 embeddings to a staging store (not the production index). (3) Build a new ANN index from V2 embeddings in the staging store. (4) Run evaluation: compare V2 duplicate decisions against the labeled eval set and against V1 decisions on a holdout. (5) If V2 meets quality bar (precision/recall improvements or parity), promote the V2 index to production with a traffic ramp (10% -> 50% -> 100%). (6) Keep V1 index as a fallback for 7 days. (7) Archive V1 embeddings to cold storage.

**Incremental backfill:** Instead of re-embedding everything, prioritize: (a) items in known duplicate clusters (re-check with V2), (b) items in the gray zone (near-threshold with V1), (c) recent items first (more likely to be queried). Stale items can be re-embedded lazily when they appear as ANN candidates.

**Version coexistence:** During backfill, the system must handle queries where the query item has a V2 embedding but some index items still have V1 embeddings. Options: (a) search only the V2 partition (lower recall until backfill completes), (b) search both V1 and V2 partitions and merge (complex but higher recall), (c) re-embed query candidates on the fly (expensive per query but exact). Option (a) is simplest; option (c) is best for accuracy on a small candidate set.

### 4. Detecting transitive chaining errors in duplicate clusters

Transitive closure on similarity is dangerous: if A~B (sim=0.93) and B~C (sim=0.91), single-linkage clustering would group {A, B, C}. But A and C might have sim=0.75, far below the duplicate threshold. This "chaining" effect produces mega-clusters of unrelated items connected through a chain of borderline similarities.

**Detection:** After building clusters via connected components, compute the cluster diameter (maximum pairwise distance within the cluster) and the minimum cut (weakest link in the chain). Flag clusters where diameter > 2x threshold or where any pair in the cluster has similarity < threshold - margin.

**Prevention:** Use average-linkage or complete-linkage clustering instead of single-linkage. Average-linkage requires every pair in the cluster to have average similarity above the threshold. Complete-linkage requires every pair to be above threshold — the most conservative. Alternatively, use a graph-based approach: build a similarity graph with edges only for pairs above threshold, then find cliques or dense subgraphs rather than connected components.

**Practical approach:** Use DBSCAN or HDBSCAN with the cosine distance matrix. DBSCAN's min_samples parameter prevents singleton chaining, and its density-based approach naturally avoids elongated clusters. Set eps = 1 - threshold (e.g., eps=0.08 for threshold=0.92) and min_samples=2.

**Monitoring:** Track cluster size distribution. A healthy system has mostly small clusters (2-5 items). If you see clusters of 100+ items, investigate — they're likely chaining errors or genuinely redundant data sources (which may be fine but should be verified).

### 5. Handling adversarial/borderline examples

Adversarial near-duplicates are items intentionally modified to evade dedup — e.g., images with subtle perturbations, text with synonym substitution or character-level obfuscation. Borderline examples are legitimately ambiguous cases where reasonable humans disagree on duplicate status.

**Adversarial defense:** Embedding-based dedup is inherently more robust than hash-based methods because learned embeddings capture semantic similarity. Small pixel perturbations that fool perceptual hashes often don't change the embedding significantly. However, adversarial attacks targeting the embedding model (e.g., PGD attacks that maximize embedding distance) can evade detection. Defenses: (a) use ensembles of embedding models — an item must evade all of them, (b) augment the embedding with a perceptual hash — if the hash matches but embedding doesn't, flag for review, (c) add random augmentations (crop, blur, color jitter) before embedding and check consistency.

**Borderline handling:** For items where the duplicate probability is near 0.5, don't make an automated decision. Route to human review with context (side-by-side display, similarity score, nearest cluster members). Use the human decision as a training signal to sharpen the boundary. Over time, the classifier should learn the boundary for each slice.

**Active learning:** Prioritize borderline examples for labeling. Use uncertainty sampling (examples with predicted probability closest to 0.5) or query-by-committee (examples where different model versions disagree). This maximizes the information gained per labeled example.

### 6. Building a human review queue that learns from scarce labeling budget

The review queue is the engine of continual improvement. Design it to maximize information gain per human judgment. A typical labeling budget might be 500-2000 pairs reviewed per day.

**Sampling strategy:** (1) Uncertainty sampling: pairs with duplicate probability in [0.4, 0.6]. These are the examples the model is most confused about, so labels here improve the decision boundary the most. (2) Diversity sampling: pairs from underrepresented slices (rare domains, new data sources, new languages). These prevent the model from overfitting to the dominant slice. (3) Disagreement sampling: pairs where the embedding model and rule-based heuristics disagree. These highlight systematic model failures. Allocate budget: 50% uncertainty, 30% diversity, 20% disagreement.

**UI design matters:** Show the annotator both items side-by-side with highlighted differences. Show the similarity score (as a confidence indicator, not a decision). Provide three options: "Duplicate," "Not Duplicate," "Unsure." Track inter-annotator agreement on a 10% overlap set to measure label quality.

**Feedback integration:** Aggregate human labels daily. Retrain the threshold/classifier weekly. Feed hard examples into the embedding model fine-tuning dataset monthly. Close the loop: after retraining, re-score the review queue to reprioritize. Items that were uncertain before retraining should become clear after — if not, investigate.

### 7. Tenant-safe service design

Tenant isolation is non-negotiable. A query from Tenant A must never return results from Tenant B, and the system must not leak information about Tenant B's data (including its existence or size).

**Index isolation:** Use per-tenant HNSW indexes. Each tenant's embeddings live in a separate HNSW graph, stored in a separate memory region. Queries are routed to the correct tenant's index by a tenant-aware router. This is the safest approach — no shared state, no filtering bugs, no side-channel leaks.

**Efficiency concern:** Per-tenant indexes waste memory for small tenants (HNSW has per-graph overhead). For tenants with <100K items, use flat search (exact k-NN) which has negligible overhead. For tenants with 100K-10M items, use HNSW with smaller M parameter (M=16 instead of 32). For tenants with >10M items, use full HNSW.

**Multi-tenant shared index (alternative):** If cost requires sharing, implement strict tenant partitioning within the index. Every vector stores a tenant_id. The search function takes tenant_id as a mandatory parameter and uses partition-aware traversal (FAISS IDSelector, Qdrant partition key). But this is risky — a bug in the filtering logic could leak data. Prefer per-tenant isolation unless cost is prohibitive.

**Audit logging:** Log every query with tenant_id, item_id, results returned, decision made. Enable audit queries: "Show all queries that touched tenant X's data in the last 30 days." This is both a security control and a compliance requirement.

### 8. Cost reasoning at billions of items

**Storage:** 1B items x 768d x 4 bytes = 3TB raw embeddings. With PQ (64 bytes/vector), index size = 64GB. HNSW graph overhead ~2x, so ~128GB total index RAM. At $10/GB-month for cloud RAM, index cost = ~$1,280/month per billion items. Raw embedding storage in S3 at $0.023/GB-month = ~$70/month for 3TB. Total storage: ~$1,350/month per billion items.

**Compute — embedding generation:** ViT-L at ~200 items/sec/GPU (A100). 1B items / 200 = 5M GPU-seconds = 1,389 GPU-hours. At $1/GPU-hour (spot A100), one-time embedding cost = ~$1,400. Daily incremental: 10M new items/day / 200 = 50K seconds = 14 GPU-hours = ~$14/day.

**Compute — serving:** HNSW search at ~5K QPS per node (8-core, 128GB RAM). For 1K QPS sustained: need ~1 node (with headroom, 2 nodes). At ~$500/month per node, serving cost = ~$1,000/month.

**Compute — offline batch:** Daily full-corpus scan with IVF-PQ at ~50K QPS throughput. 1B items / 50K = 20K seconds ~= 6 hours on one node. Cost: negligible compared to serving.

**Human review:** 1,000 pairs/day at $0.10/pair = $100/day = $3,000/month. This is often the dominant cost.

**Total for 1B items:** ~$5,000-8,000/month. At Scale's revenue per labeled item, this is easily justified if it prevents even 1% redundant labeling.

### 9. Disaster plan for corrupted/stale index

**Corrupted index:** If the HNSW index becomes corrupted (e.g., disk failure, OOM during build, bit rot), queries return garbage results. Detection: monitor recall against a known-pair test set. If recall drops below a threshold, declare the index corrupted.

**Recovery:** (1) Stop serving from the corrupted index. (2) Fall back to a known-good snapshot (keep the last 3 index snapshots in cold storage). (3) If no snapshot is available, rebuild from raw embeddings in the embedding store. Rebuild time for 100M items: ~2-4 hours for HNSW. (4) During recovery, the online path is degraded — either serve from a stale snapshot (with a "results may be incomplete" flag) or disable online dedup and let the offline pipeline catch up.

**Stale index:** If the index hasn't been updated with recent items (e.g., insertion pipeline lag), new items won't be checked against recent additions. Detection: monitor the gap between the latest item timestamp in the index and the current time. Alert if gap > 1 hour.

**Prevention:** (1) Replicate the index across multiple nodes (read replicas). (2) Use checkpointing — snapshot the index every hour. (3) Keep the raw embedding store as the source of truth — the index is always rebuildable. (4) Run a continuous reconciliation job that compares item counts in the embedding store vs. the index.

### 10. Retrieval looks good but final precision is bad

This means the ANN search is returning the right candidates (high recall), but the final duplicate decision is wrong (low precision). The problem is in the decision layer, not the retrieval layer.

**Diagnosis:** Examine false positives. Are they items that are similar but not duplicates? (e.g., same template different content, same scene different time, same topic different opinion). If so, the embedding model captures the wrong notion of similarity, or the threshold is too low.

**Fix 1 — Tighter threshold:** Raise the threshold. This is the simplest fix but reduces recall. Plot the precision-recall curve and find the knee.

**Fix 2 — Re-ranking classifier:** Add a lightweight classifier after retrieval. Input: [cosine_similarity, embedding_diff, metadata_features (same source? same date? same length?)]. Output: duplicate probability. Train on human-labeled pairs. This classifier can learn patterns that cosine similarity alone misses (e.g., "same source + high similarity = duplicate" but "different source + high similarity = probably not duplicate").

**Fix 3 — Better embeddings:** Fine-tune the embedding model with hard negatives from the false-positive log. Use contrastive learning with triplets: (anchor, positive=true_duplicate, negative=false_positive_match). This pushes the model to separate near-duplicates from merely similar items.

**Fix 4 — Feature augmentation:** Compute additional similarity signals beyond embedding cosine: structural similarity (SSIM for images), exact substring overlap (for text), metadata overlap (same source URL, same upload batch). Use these as additional features in the re-ranking classifier.

---

## Evaluation Questions

### 1. Offline evaluation set — building positive and hard-negative pairs

The evaluation set is the foundation of all quality measurement. It must be representative, high-quality, and regularly refreshed.

**Positive pairs (true duplicates):** (1) Mine from human review labels accumulated over time. (2) Generate synthetically: take an item, apply realistic augmentations (crop, resize, compress, add watermark for images; paraphrase, truncate, reorder for text), and use the (original, augmented) pair as a positive. (3) Use known duplicates from data ingestion (same item uploaded twice, same URL crawled twice).

**Hard negatives (similar but not duplicate):** (1) Take ANN nearest neighbors that humans labeled as "not duplicate." These are the most valuable — they represent the exact failure modes of the system. (2) Take items from the same category/class that are genuinely different (two different dogs, two different news articles on the same topic). (3) Use template-matched pairs (same form layout, different content).

**Dataset construction:** Aim for 5,000-10,000 labeled pairs. Balance: 40% true duplicates (varying difficulty), 40% hard negatives, 20% easy negatives (random pairs, for sanity checking). Stratify by modality, domain, and source. Refresh quarterly by adding new pairs from the latest human review queue.

**Anti-contamination:** Never use evaluation pairs for model training or threshold calibration. Maintain strict train/eval/test splits. Track data provenance to prevent leakage.

### 2. Estimating recall with incomplete ground truth

In a corpus of 1B items, it's impossible to identify all duplicate pairs (the number of possible pairs is O(n^2) = 5*10^17). You can only estimate recall, not measure it exactly.

**Method 1 — Sampling-based recall estimation:** Draw a random sample of N items (e.g., N=10,000). For each sampled item, exhaustively search the full corpus using flat (exact) search. Compare the results to what the ANN system returns. The fraction of true neighbors recovered by the ANN system is the estimated recall. This measures ANN recall, not system-level recall.

**Method 2 — Known-pair injection:** Inject synthetic duplicate pairs into the corpus (items with known duplicates). Check whether the system identifies them. The fraction identified is a lower bound on recall. Advantage: directly measures end-to-end recall including threshold effects. Disadvantage: synthetic pairs may not represent real duplicates.

**Method 3 — Capture-recapture:** Use two independent dedup methods (e.g., hash-based and embedding-based). Count duplicates found by each and by both. Apply the Lincoln-Petersen estimator to estimate the total number of duplicates. This gives a population estimate, from which you can compute recall for each method.

**Method 4 — Human audit:** Sample 500 items flagged as "not duplicate" by the system. Have humans verify. The fraction that humans identify as actually duplicate is the system's false negative rate on that sample. Recall = 1 - FNR. This is the most direct measurement but expensive and subject to human error.

### 3. Which metrics matter most

**Primary metrics:** (1) Precision@threshold: of items flagged as duplicate, what fraction are truly duplicate? Target: >0.95 for auto-decisions. (2) Recall@threshold: of true duplicates, what fraction does the system catch? Target: >0.80 for online, >0.95 for offline. (3) F1 or F-beta with beta chosen to reflect the FP/FN cost ratio (beta=0.5 if false positives are 2x as costly as false negatives).

**Operational metrics:** (4) Latency: p50 and p99 query latency for online path. Target: p50 < 10ms, p99 < 50ms. (5) Throughput: sustained QPS for online path. (6) Index freshness: lag between item ingestion and index availability. Target: <5 minutes. (7) Human review queue depth and throughput.

**Business metrics:** (8) Labeling cost savings: estimated reduction in redundant annotation work. (9) Model quality impact: does dedup improve downstream model performance? Measure on held-out benchmarks after training with/without dedup. (10) Edge case preservation: are rare, high-value items (e.g., unusual AV scenarios) being preserved or incorrectly removed?

**Don't over-optimize a single metric.** Precision is most important for trust, recall is most important for cost savings, and edge case preservation is most important for safety. Report a balanced scorecard.

### 4. Slice metrics by source, modality, language, domain

Global metrics hide problems. A system with 95% precision overall might have 99% precision on English text and 70% precision on Japanese text. Slice-level metrics expose these gaps.

**Mandatory slices:** (1) By modality: image, text, video, etc. (2) By domain/use case: AV perception, RLHF, content moderation, satellite imagery. (3) By source: web crawl, customer upload, synthetic generation. (4) By language (for text): English, Chinese, Japanese, etc. (5) By data volume: high-volume slices vs. long-tail slices.

**Implementation:** Tag every evaluation pair with slice labels. Report precision, recall, and F1 per slice. Set minimum performance bars per slice (e.g., precision > 0.90 on every slice, not just overall). Alert when any slice drops below its bar.

**Bias detection:** Compare false positive rates across slices. If the system disproportionately flags items from a specific source or language as duplicates, investigate — it may reflect a bias in the embedding model (e.g., the model collapses representations for an underrepresented language).

### 5. Guardrail metrics for catastrophic regressions

Guardrail metrics are hard limits that must never be violated, regardless of aggregate improvements. They protect against catastrophic failures that could harm users, data quality, or trust.

**Guardrail 1 — No-delete rate for labeled rare events:** Maintain a curated set of rare, high-value items (e.g., unusual AV scenarios, safety-critical edge cases). The system must never flag these as duplicates. Check this on every model/threshold update.

**Guardrail 2 — Precision floor per slice:** No slice should have precision below 0.85, even if overall precision is 0.97. A single low-precision slice means the system is actively harming data quality for that segment.

**Guardrail 3 — Cluster size cap:** No duplicate cluster should exceed 1,000 items. A cluster larger than this almost certainly contains chaining errors. Alert and investigate.

**Guardrail 4 — Dedup rate sanity:** The overall dedup rate (fraction of items flagged as duplicate) should stay within a historical range (e.g., 5-15%). If it suddenly jumps to 50%, something is wrong (threshold too low, model regression, data pipeline bug). If it drops to 0%, the system is broken.

**Guardrail 5 — Cross-tenant contamination:** Zero tolerance. Any query that returns results from a different tenant is a critical severity incident.

### 6. A/B or shadow testing for new embedding model

**Shadow testing (preferred for this use case):** Deploy the new model alongside the old one. Both process every incoming item. The old model makes real decisions; the new model's decisions are logged but not acted upon. Compare the two models' decisions on the same items over 1-2 weeks.

**Analysis:** (1) Agreement rate: how often do the two models agree? If >95% agreement, the new model is unlikely to cause regressions. (2) Disagreement analysis: for cases where they disagree, which model is correct? Sample disagreements and send to human review. (3) Precision/recall comparison: using the evaluation set, measure both models' precision and recall. (4) Distribution comparison: compare similarity score distributions — if the new model's distribution has shifted significantly, thresholds need recalibration.

**A/B testing is harder** because duplicate detection has long-term effects: an item incorrectly allowed into the corpus affects all future comparisons. You can't easily undo a false negative. Instead, use shadow testing for the dedup decision and A/B testing for downstream model training: train two models on corpora deduped by V1 and V2 respectively, compare on benchmarks.

**Rollout:** After shadow testing confirms improvement, deploy the new model with a gradual rollout: 10% of traffic for 24 hours (monitor for regressions), 50% for 48 hours, 100%. Keep the old model index warm for 7 days as a rollback target.

### 7. Success metric when downstream consumer is model quality

The ultimate goal of dedup is not to find duplicates — it's to improve the quality of models trained on the deduped data. This is the hardest metric to measure because the causal chain is long: better dedup -> cleaner training data -> better model -> better product.

**Direct measurement (expensive but definitive):** Train the same model architecture on (a) the raw corpus and (b) the deduped corpus. Compare on held-out benchmarks. If the deduped model is better or equal with fewer training examples, dedup is adding value. This requires a full training run, which may cost $10K-$100K, so do it quarterly.

**Proxy metrics (cheaper, faster):** (1) Dataset diversity: measure embedding space coverage (e.g., number of clusters, uniformity of cluster sizes) before and after dedup. Higher diversity = better. (2) Training efficiency: does the model converge faster (fewer epochs) on deduped data? (3) Influence analysis: use data influence functions (e.g., TracIn, Datamodels) to identify which training examples are redundant. Dedup should preferentially remove low-influence examples.

**Negative signal:** If dedup removes too aggressively and model quality drops, the system is being too aggressive. This is the most important feedback loop — the system must be tuned to improve, not degrade, downstream model quality. Build a "dedup impact dashboard" that tracks model benchmarks over time alongside dedup rate changes.

---

## Continual Improvement

### 1. Where human feedback enters the loop

Human feedback enters at three points: (1) **Explicit review queue:** Pairs near the decision boundary are presented to human reviewers who label them as duplicate/not-duplicate. This is the primary feedback source, generating 500-2000 labeled pairs per day. (2) **Implicit signals from annotation:** When annotators encounter items they've previously labeled (or that look familiar), they may flag them. This signal is noisy but free — capture it from the annotation tool's UX. (3) **Downstream model training feedback:** If a data scientist discovers that certain training examples are redundant (via data analysis or influence functions), they can retroactively label them as duplicates.

Each feedback type has different noise levels and latency. Explicit review labels are high quality but expensive and delayed (hours to days). Implicit annotator signals are free but noisy (annotators may misremember). Downstream training feedback is definitive but very delayed (weeks to months).

Design the feedback ingestion pipeline to handle all three sources with different confidence weights. Explicit labels get weight 1.0. Implicit signals get weight 0.3 (require multiple annotators to agree). Downstream feedback gets weight 1.0 but is rare. Store all feedback with provenance metadata for auditability.

The feedback doesn't just improve thresholds — it also identifies systematic failures. If 80% of false positives come from a specific template, that's a signal to add template-aware preprocessing. If 80% of false negatives come from a specific data source, that source may need a source-specific threshold.

### 2. Sampling review candidates for max information gain

With a budget of 1,000 human reviews per day, you must choose wisely. Random sampling wastes budget on easy cases. Pure uncertainty sampling over-fits to the current decision boundary. A balanced approach maximizes information gain.

**Uncertainty sampling (50% of budget):** Select pairs with predicted duplicate probability closest to 0.5. These are the examples the model is most confused about. Labels here directly improve the decision boundary. Implementation: after scoring all candidate pairs through the re-ranking classifier, sort by |probability - 0.5| ascending, take top-500.

**Diversity sampling (30% of budget):** Select pairs from underrepresented slices. If the evaluation set has 10 slices and one slice has only 20 labeled pairs, prioritize that slice. Implementation: compute the number of labeled pairs per slice, compute a sampling weight inversely proportional to count, sample 300 pairs according to these weights.

**Surprise sampling (20% of budget):** Select pairs where the system's decision changed between model versions or time periods. A pair that was "duplicate" last week but "not duplicate" this week (or vice versa) indicates instability — human labels resolve the ambiguity. Also include pairs where different system components disagree (hash says duplicate, embedding says not, or vice versa).

**Anti-gaming:** Rotate sampling strategies periodically. If the model learns to be confident everywhere (because uncertainty sampling reduced its uncertainty), the system stops improving. Inject a 5% random sample to maintain calibration.

### 3. Using disagreement between retrieval/classifier/rules

Run multiple dedup signals in parallel: (1) perceptual hash match, (2) embedding cosine similarity above threshold, (3) re-ranking classifier prediction, (4) rule-based checks (same URL, same file size, same metadata). When these signals disagree, the item is informative.

**Disagreement types and actions:**
- Hash match + embedding mismatch: possibly a hash collision or a case where visual similarity doesn't imply semantic similarity. Route to review.
- Embedding match + classifier reject: the classifier has learned a pattern that raw similarity misses (e.g., same template different content). Investigate — this is a fine-grained signal.
- Rule match + embedding mismatch: metadata suggests duplicate but content differs (e.g., same URL crawled at different times with different content). Update rules.
- All agree "duplicate": high confidence, auto-decide.
- All agree "not duplicate": high confidence, auto-decide.

**Ensemble voting:** In production, use a weighted vote across signals for the final decision. Weight each signal by its historical precision on the evaluation set. The classifier typically gets the highest weight (0.5), embedding similarity gets 0.3, hash gets 0.15, rules get 0.05.

**Disagreement logging:** Log every disagreement with full context. This creates a rich dataset for debugging and model improvement. Weekly analysis of disagreement patterns reveals systematic issues.

### 4. Retrain embedder vs recalibrate thresholds

This is a cost-benefit decision. Threshold recalibration is cheap (minutes, no GPU, no re-indexing). Embedder retraining is expensive (GPU hours, re-embedding the corpus, rebuilding the index). Choose the right lever for the situation.

**Recalibrate thresholds when:** (1) The similarity distribution has shifted (new data sources with different characteristics). (2) The precision-recall trade-off is suboptimal but the embedding space itself is good (the right neighbors are retrieved, just at the wrong score). (3) You have new labeled data that refines the operating point but doesn't change the fundamental similarity structure.

**Retrain embedder when:** (1) The right items aren't even in the top-k neighbors (retrieval recall is low). (2) Hard negatives are embedded too close to true positives, and no threshold can separate them. (3) A new modality or domain is added that the current model doesn't handle well. (4) The false positive analysis shows systematic failures that trace to the embedding space structure.

**Decision framework:** After collecting new labeled data, first try threshold recalibration. Measure improvement. If precision or recall improves by >2%, ship the new thresholds. If improvement is <1%, the problem is in the embedding space — consider retraining. If retraining, use the new labeled data as hard negatives/positives in contrastive fine-tuning, not as the entire training set (preserve the pretrained model's general knowledge).

**Cadence:** Recalibrate thresholds weekly. Retrain embedder quarterly (or when triggered by significant quality degradation that threshold tuning can't fix).

### 5. Preventing feedback loops from collapsing diversity

A dangerous failure mode: the system removes "duplicates," reducing diversity. The next training iteration sees less diverse data, learns narrower representations. The narrower representations make more items look similar, causing more dedup. Repeat until the training set is a small homogeneous subset.

**Detection:** Monitor dataset diversity metrics over time: (1) Embedding space coverage: number of occupied Voronoi cells, average inter-cluster distance, uniformity of cluster size distribution. (2) Dedup rate trend: if the dedup rate is monotonically increasing (5% -> 8% -> 12% -> 20%), the system may be collapsing. (3) Category distribution: track the distribution of items across categories/domains. If long-tail categories are shrinking, diversity is being lost.

**Prevention:** (1) Never auto-delete — only flag and let downstream consumers decide. This preserves the full corpus for retraining. (2) Stratified dedup: within each category, limit the dedup rate to prevent over-removal from any single category. (3) Diversity-preserving dedup: when choosing which member of a duplicate cluster to keep, prefer the one that maximizes coverage (furthest from existing non-duplicate items in embedding space). (4) Periodic recalibration against the original (pre-dedup) corpus, not just the current (post-dedup) corpus.

**Architectural safeguard:** The dedup system should soft-label (tag as duplicate) rather than hard-delete. The downstream data pipeline applies the dedup labels as a filter, and the filter can be loosened if diversity metrics drop.

### 6. Maintaining reproducibility with changing index and models

Reproducibility is critical for debugging, auditing, and regulatory compliance. If someone asks "why was item X flagged as a duplicate on March 1?", you must be able to recreate the decision.

**What to version:** (1) Embedding model: version ID, weights snapshot. (2) ANN index: snapshot with timestamp, model version used, item IDs included. (3) Thresholds: per-slice threshold values with effective date range. (4) Decision logic: version of the re-ranking classifier, rule set, ensemble weights. (5) Labeled data: version of the evaluation set and training set.

**How to version:** Use immutable artifacts in object storage (S3). Each artifact is identified by a content hash (SHA-256). A version manifest ties together all artifacts for a given system version: `{model: sha256:abc, index: sha256:def, thresholds: sha256:ghi, classifier: sha256:jkl}`. Store the manifest in a version registry (DynamoDB, PostgreSQL).

**Reproducibility protocol:** For any past decision, retrieve the version manifest that was active at that time. Load the corresponding model, index, thresholds, and classifier. Re-run the query. The result should match the logged decision. Test this protocol monthly with spot checks.

**Practical limitation:** Exact reproducibility for ANN search is difficult because HNSW traversal can be non-deterministic (due to concurrent updates or floating-point ordering). Log the exact candidate set returned by ANN search so the re-ranking step can be exactly reproduced even if the ANN traversal differs slightly.

### 7. Versioning for embeddings, thresholds, policies, labels

**Embeddings:** Version format: `{model_architecture}-{training_version}-{finetune_version}`. Example: `dinov2-vitl14-ft3`. Store in the embedding record: `{item_id, tenant_id, embedding_version, vector, created_at}`. When the model changes, old embeddings coexist with new ones in the store, each tagged with their version. The index is version-specific: `index-dinov2-vitl14-ft3-tenant42`.

**Thresholds:** Version format: `{embedding_version}-{calibration_date}-{slice_id}`. Example: threshold for `dinov2-vitl14-ft3` calibrated on `2026-03-10` for slice `av-camera-front` = 0.923. Store in a configuration service with history. Never overwrite — append new versions with effective dates.

**Policies:** Business rules (e.g., "never dedup items from source X," "always dedup items with identical filenames") are versioned as code (git commit hash) and as configuration (policy version in the config service). Changes to policies go through code review and are deployed with the same rollout safeguards as model changes.

**Labels:** The labeled dataset is versioned in a dataset registry. Each version records: date, source (which review queue, which feedback pipeline), number of pairs, class distribution. When labels are added, a new version is created (never modify in place). Model training and threshold calibration reference a specific label version. This enables reproducibility: "model V3 was trained on labels V7."

---

## Operational and Reliability

### 1. What to log per request

Every query to the dedup service should produce a structured log entry containing: (1) **Request metadata:** timestamp, request_id, tenant_id, item_id, modality, source. (2) **Embedding:** model_version used, embedding generation latency, embedding hash (for dedup of the dedup request itself). (3) **Retrieval:** index_version queried, number of candidates retrieved, top-k item_ids with similarity scores, ANN search latency. (4) **Decision:** threshold applied, re-ranker score (if applicable), final decision (duplicate/not-duplicate/review), decision confidence, applied policy rules. (5) **Performance:** total end-to-end latency, component-level latencies (embedding, retrieval, re-ranking, decision).

**Storage:** Log to a structured logging system (e.g., structured JSON to Kafka -> S3/BigQuery). Retain for 90 days in hot storage (queryable), 1 year in cold storage. Total log volume: ~1KB per request x 10K QPS = ~10MB/s = ~860GB/day. Budget for this.

**Privacy:** Do not log the raw embedding vector (too large and potentially invertible to recover the original item). Log the embedding hash for debugging. If the item contains PII, log only the item_id, not the content.

**Usage:** Logs are used for (a) debugging individual decisions, (b) computing aggregate metrics, (c) training data for the re-ranking classifier, (d) audit trails for compliance, (e) anomaly detection (sudden changes in score distributions, latency spikes, etc.).

### 2. Observability for queue lag, index freshness, model freshness

**Dashboard metrics (real-time):**
- Online QPS and latency (p50, p95, p99) with 1-minute resolution
- ANN index freshness: seconds since last successful index update
- Embedding service GPU utilization and queue depth
- Human review queue depth and drain rate (items/hour)
- Dedup rate (fraction of items flagged) by tenant and modality

**Health checks (automated):**
- Canary queries: every 60 seconds, send a known duplicate pair and a known non-duplicate pair through the system. Verify correct decisions. Alert if either fails.
- Index consistency: compare item count in the index vs. the embedding store. Alert if divergence > 0.1%.
- Model serving health: verify the embedding service returns valid embeddings (correct dimensionality, L2 norm ~= 1.0) for test inputs.

**Alerting thresholds:**
- p99 latency > 100ms: warning
- p99 latency > 500ms: critical
- Index freshness > 30 minutes: warning
- Index freshness > 2 hours: critical
- Review queue depth > 10,000: warning (falling behind)
- Canary failure: critical (immediate page)

**Long-term trends (weekly review):**
- Precision and recall on the evaluation set
- Threshold stability per slice
- Similarity score distribution shift
- Dedup rate trends
- Human review agreement rate

### 3. SLOs for latency, precision floor, freshness

**Latency SLO:** p50 < 10ms, p99 < 50ms for online dedup decisions (excluding embedding generation, which is a separate SLO at p50 < 30ms, p99 < 100ms for images). Measured at the API gateway. Error budget: 0.1% of requests per month may exceed p99.

**Precision floor SLO:** Overall precision > 0.95 on the evaluation set, measured weekly. Per-slice precision > 0.85 on every defined slice. This is measured offline (not real-time) but violations trigger immediate investigation and potential threshold adjustment.

**Freshness SLO:** New items are searchable in the index within 5 minutes of embedding completion (online path). Offline cluster updates complete within 24 hours. Error budget: 99.5% of items meet the 5-minute target per month.

**Availability SLO:** 99.9% uptime for the online dedup API. Downtime is measured as periods where error rate > 5% or latency > 500ms p50. This allows ~43 minutes of downtime per month.

**Review queue SLO:** 95% of items in the review queue are reviewed within 48 hours. This is a human-dependent SLO that requires adequate staffing.

### 4. Handling partial outages

**Embedding service down:** Items flow through without dedup (fail-open). Tag items as "dedup-pending." When the embedding service recovers, backfill embeddings and run dedup retroactively via the offline pipeline. Alert the oncall immediately — this is a data quality gap.

**ANN index service down:** If the primary index is unavailable, try the read replica. If all replicas are down, fall back to hash-based dedup only (catches exact duplicates but misses near-duplicates). Tag items as "ann-dedup-pending." This provides degraded but non-zero dedup coverage.

**Decision service down:** If the threshold/classifier service is unavailable, use cached thresholds (last known good values, refreshed daily). If cached values are stale (>24 hours old), fail-open with "dedup-pending" tag.

**Human review queue overloaded:** If queue depth exceeds capacity, stop routing borderline cases to review. Instead, apply a conservative threshold (higher precision, lower recall) to auto-decide borderline cases. Log these as "auto-decided-due-to-queue-pressure" for later audit.

**Design principle:** Always fail-open (allow items through) rather than fail-closed (block items). Missing a duplicate is recoverable; blocking a legitimate item may cause pipeline stalls and customer impact.

### 5. What to cache and what never to cache

**Cache:**
- Embedding model weights (in GPU memory, warm on startup)
- Per-slice thresholds (in application memory, refresh every 5 minutes)
- Perceptual hashes of recent items (in Redis/Memcached, TTL 24 hours) for fast exact dedup
- ANN index (in memory, this is the entire point)
- Recently computed embeddings (if the same item is submitted twice in quick succession)

**Never cache:**
- Dedup decisions (the decision for an item may change as more items are added to the index — a new item may make a previously non-duplicate item become a duplicate)
- Human review labels (must always be read from the source of truth to prevent stale data informing model updates)
- Cross-tenant results (caching results across tenant boundaries is a privacy violation vector)
- Threshold calibration parameters during active recalibration (use the new values immediately once validated)

**Cache invalidation:** Threshold cache is invalidated when a new calibration is published. Embedding cache is invalidated when the model version changes. Hash cache is size-bounded (LRU eviction). Index cache is rebuilt on the configured schedule.

### 6. Canaries and rollback for new index builds

**Index build pipeline:** (1) Build new index from current embeddings on a staging cluster. (2) Run the canary test suite: 1,000 known-pair queries (500 true duplicates, 500 known non-duplicates). Verify precision > 0.95 and recall > 0.80. (3) Run regression test: compare new index decisions against old index decisions on 10,000 random recent items. Flag if disagreement rate > 5%. (4) If canary passes, deploy to 1 read replica (canary node). Route 5% of production traffic to the canary node. (5) Monitor canary node for 1 hour: latency, error rate, precision on live traffic. (6) If canary node is healthy, promote to all replicas. (7) Keep old index on standby for 24 hours.

**Rollback:** If canary fails (precision drop, latency spike, error rate increase), immediately route all traffic back to the old index. No manual intervention required — the deployment system automatically rolls back if the canary health check fails 3 consecutive times.

**Rollback granularity:** Rollback targets the index only, not the embedding model or thresholds. Each component can be rolled back independently. This isolation prevents cascading rollbacks.

**Frequency:** New index builds happen on every model change, threshold change, or when the incremental update lag exceeds 1 hour. Routine rebuilds happen weekly to compact the HNSW graph (which degrades with many incremental insertions).

### 7. Security/privacy controls

**Data access control:** (1) Tenant isolation enforced at the API layer (every request must include a valid tenant_id, authenticated via API key or JWT). (2) Index partitioning by tenant (as discussed earlier). (3) No cross-tenant queries — the API rejects any request that specifies a different tenant_id than the authenticated identity.

**Embedding privacy:** Embeddings are one-way projections of the original data, but they may be partially invertible (recent work shows images can be approximately reconstructed from CLIP embeddings). Treat embeddings as sensitive data with the same access controls as the original items. Encrypt embeddings at rest (AES-256) and in transit (TLS 1.3).

**Access logging:** Log every data access (embedding read/write, index query, label access) with the identity of the accessor. Enable audit queries for compliance.

**PII handling:** If items contain PII, the embedding may encode PII-related features. Apply PII detection before embedding and tag items accordingly. PII-tagged items may require special handling: separate index, restricted access, automatic expiration.

**Model security:** The embedding model is an asset — protect the weights from exfiltration. Serve the model from a secure enclave. Do not expose model internals (gradients, intermediate representations) in API responses or logs.

---

## Robotics/AV/Physical AI Twist

### 1. How design changes for temporally adjacent AV frames

AV cameras capture frames at 10-30 Hz. Adjacent frames are almost identical — same scene, same objects, nearly the same positions. This creates a massive dedup challenge: 90%+ of frames may be "near-duplicates" of their temporal neighbors, but they contain subtly different information (object motion, lighting changes, ego-vehicle position changes).

**Key insight:** For AV, the notion of "duplicate" is not just visual similarity — it's informational redundancy for perception model training. Two nearly identical highway frames are redundant. Two visually similar intersection frames with different pedestrian positions are not.

**Design changes:** (1) Add a temporal gating layer: before embedding-based dedup, check if the candidate pair comes from the same drive sequence and is within N seconds (e.g., 5 seconds). Temporally adjacent frames are dedup candidates; temporally distant similar frames are usually not duplicates (same intersection on different days = different training examples). (2) Use motion-aware embeddings: augment the visual embedding with optical flow or ego-motion features. Two frames that look similar but have different object motions should be farther apart in embedding space. (3) Frame sampling as a first pass: before per-frame dedup, apply intelligent subsampling (e.g., keep every Kth frame, or keep frames where significant scene change occurs).

### 2. Dedupe at frame vs clip vs trajectory vs scene level

**Frame-level:** Easiest to implement (each frame is an item). But ignores temporal context — a turning sequence is not a duplicate of a similar turning sequence if the dynamics are different.

**Clip-level (2-5 second windows):** More semantically meaningful. A clip captures a complete micro-action (lane change, pedestrian crossing). Embed clips using a video encoder (e.g., VideoMAE, InternVideo) or by pooling frame embeddings (mean, attention-weighted). Dedup at clip level preserves temporal dynamics while removing redundant clips (e.g., 100 clips of empty highway driving).

**Trajectory-level (full drive sequences):** Dedup at the route level — if two drives cover the same route with similar conditions, one may be redundant. Use GPS/route embeddings (encode the sequence of road segments) combined with condition features (weather, time of day, traffic density).

**Scene-level (semantic scenario):** Most abstract. A "scene" is a semantic scenario: "unprotected left turn with pedestrian." Dedup at this level requires scene understanding, which is the downstream task itself. Use scenario classifiers or language-based scene descriptions and dedup on the description embedding.

**Recommendation:** Multi-level dedup. Frame-level dedup for fast first pass (remove obviously redundant adjacent frames). Clip-level dedup for the primary pass (remove redundant micro-actions). Trajectory-level dedup for periodic cleanup (remove redundant drives). Scene-level dedup informing data collection strategy (identify underrepresented scenarios for targeted collection).

### 3. Preserving rare edge cases while removing boring redundancy

This is the central tension in AV data dedup. The system must aggressively remove the 90% of data that's boring (straight highway, empty parking lot) while perfectly preserving the 0.1% of data that's rare and safety-critical (child running into road, construction zone, emergency vehicle).

**Approach 1 — Rarity scoring:** For each item, compute a rarity score = 1 / (density in embedding space). Items with high rarity scores (few nearby neighbors) are presumed rare/interesting and should never be deduped. Items with low rarity scores (many nearby neighbors) are common and candidates for dedup. Implementation: use the ANN index itself — the number of neighbors within a radius r is a density proxy.

**Approach 2 — Scenario-aware dedup budget:** Define scenario categories (highway, intersection, pedestrian, cyclist, construction, weather, etc.). Set a minimum retention count per category (e.g., keep at least 10K examples of each scenario, but no more than 100K of any single scenario). Dedup within each category separately, respecting the budget.

**Approach 3 — Value-based retention:** Use downstream model performance to estimate each item's training value (via influence functions or gradient similarity). High-value items are retained regardless of duplicate status. Low-value items in large clusters are aggressively deduped.

**Safety net:** Maintain a "protected" item list — items manually curated or flagged as important that the dedup system is forbidden from removing. Audit the dedup output for rare scenario preservation before applying to production datasets.

### 4. Using camera metadata, time, geolocation, route context

AV data comes with rich metadata that is absent in generic image dedup: GPS coordinates, IMU data, camera intrinsics/extrinsics, timestamp, weather conditions, vehicle speed, road type. Use this metadata to improve dedup quality.

**Geolocation-based pre-clustering:** Two frames from the same GPS location (within 10m) on different days are strong dedup candidates — same intersection, same view. Cluster items by geo-cell (e.g., S2 cells at level 18 ≈ 50m resolution) and search for duplicates only within the same geo-cell. This dramatically reduces the search space and makes semantic sense.

**Time-of-day and weather as features:** A daytime sunny frame and a nighttime rainy frame of the same intersection are NOT duplicates — they represent different training conditions. Include time-of-day and weather features in the dedup decision (either as metadata filters or as features in the re-ranking classifier).

**Route-level dedup:** If the same vehicle drives the same route daily, each drive produces similar data. Use route matching (sequence of GPS waypoints) to identify repeated routes. Within repeated routes, keep one "golden" drive and sample from others only for diverse conditions (different times, weather, traffic levels).

**Camera pose awareness:** Multi-camera setups produce partially overlapping views. Don't dedup across cameras — the multi-view information is valuable for 3D reconstruction and multi-view training. Only dedup within the same camera's temporal stream.

### 5. Avoiding discarding useful multi-view/multi-sensor diversity

AV platforms have multiple cameras (6-12), LiDAR (1-5), radar, and IMU. The same scene is captured from multiple viewpoints and sensor modalities simultaneously. This multi-sensor data is extremely valuable — it enables sensor fusion, cross-modal learning, and 3D scene understanding.

**Rule 1:** Never dedup across sensor types. A camera image and a LiDAR point cloud of the same scene are not duplicates — they provide complementary information. The dedup system should treat each sensor stream independently.

**Rule 2:** Be cautious deduping across camera angles. Front-left and front-right cameras capture overlapping but different views. If both are needed for stereo depth estimation or surround-view training, they must both be retained. Only dedup within the same camera stream (e.g., two consecutive frames from the front-center camera).

**Rule 3:** Preserve temporal synchronization. When deduping frames from one camera, ensure the corresponding data from all other sensors is also kept or removed together. Never break the temporal alignment between sensors — downstream models expect synchronized multi-sensor inputs. Implementation: the atomic unit of dedup is a "frame group" (all sensor data from the same timestamp), not an individual image.

**Exception:** For large-scale pretraining where individual images are used (not multi-sensor fusion), you can dedup images across cameras. But label them with camera_id so the training pipeline can ensure view diversity.

### 6. What notion of duplicate is actually harmful for training VLA/perception models

For Vision-Language-Action (VLA) models and perception models, not all redundancy is harmful. Some redundancy is actually beneficial — it reinforces common patterns and improves robustness. The harmful duplicates are specifically those that:

**Class imbalance amplification:** If a common scenario (e.g., empty highway) has 10x more data than it needs, duplicates amplify this imbalance, causing the model to over-represent the common scenario and under-represent rare ones. Solution: class-balanced sampling during training, informed by dedup cluster sizes.

**Memorization risk:** Exact or near-exact duplicates that appear in both training and validation sets cause inflated evaluation metrics (data leakage). This is the most directly harmful form of duplication. Solution: ensure dedup clusters are split consistently across train/val/test.

**Label noise amplification:** If a duplicated item has inconsistent labels across copies (common in crowd-sourced annotation), the model receives conflicting supervision. Dedup with label reconciliation (majority vote across duplicates) resolves the conflict.

**Compute waste:** Redundant items waste training FLOPS without improving model quality. For a VLA model training run costing $100K+, even 10% redundancy = $10K wasted. This is the simplest economic argument for dedup.

**NOT harmful:** Moderate redundancy (2-5x) of rare scenarios is actually beneficial — it ensures the model sees enough examples of critical cases. The dedup system should have a lower bound on retention: never reduce a scenario to fewer than K examples.

---

## Math and Retrieval Depth

### 1. Cosine similarity vs inner product vs Euclidean distance

**Cosine similarity:** cos(a, b) = (a . b) / (||a|| * ||b||). Ranges from -1 to 1. Invariant to vector magnitude. The standard metric for learned embeddings because magnitude often reflects input length or model artifacts, not semantic content.

**Inner product (dot product):** a . b = ||a|| * ||b|| * cos(a, b). Equivalent to cosine similarity when vectors are L2-normalized (||a|| = ||b|| = 1). Faster to compute (no normalization step at query time). All major ANN libraries (FAISS, ScaNN, HNSWlib) support inner product natively. **Recommendation: L2-normalize all embeddings at embedding time, then use inner product everywhere.**

**Euclidean distance:** d(a, b) = ||a - b||. For L2-normalized vectors, d(a, b) = sqrt(2 - 2*cos(a, b)). Monotonically related to cosine similarity, so ranking is identical. Some ANN libraries default to Euclidean (e.g., FAISS IndexFlatL2). The threshold interpretation is less intuitive than cosine (what does d=0.3 mean?), so prefer cosine for human-facing thresholds.

**When inner product ≠ cosine:** If embeddings are not normalized (e.g., maximum inner product search for recommendation systems where magnitude encodes popularity), inner product and cosine give different rankings. For near-duplicate detection, always normalize — magnitude is noise.

### 2. Calibrating threshold when similarity distributions differ by slice

The fundamental problem: cosine similarity is not calibrated across slices. A score of 0.90 might be "very similar" for natural images (where random pairs have cosine ~0.1) but "moderately similar" for text prompts (where random pairs may have cosine ~0.4 due to shared vocabulary).

**Empirical distribution approach:** For each slice, compute the similarity distribution on random pairs (sample 10K random pairs, compute cosine similarity, fit a distribution — typically approximately Gaussian after normalization). Set the threshold at the Nth percentile (e.g., 99.5th percentile of the random-pair distribution). This means "duplicate = more similar than 99.5% of random pairs in this slice." The absolute threshold will differ by slice, but the statistical meaning is consistent.

**Formal calibration:** Model the problem as hypothesis testing. Null hypothesis: the pair is not a duplicate (drawn from the random-pair distribution). Alternative: the pair is a duplicate (drawn from the duplicate-pair distribution). Set the threshold to control the false positive rate at alpha (e.g., alpha = 0.05). This requires knowing both distributions, which is where labeled data comes in.

**Practical implementation:** Store per-slice distribution parameters (mean, std of random-pair similarity) in the configuration service. At query time, convert raw cosine similarity to a z-score: z = (sim - mu_slice) / sigma_slice. Apply a universal z-score threshold (e.g., z > 4.0 = duplicate). This normalizes across slices automatically.

**Caveat:** This assumes the random-pair distribution is stable over time. If the data distribution shifts (new data sources, different content mix), the distribution parameters become stale. Re-estimate monthly or when drift is detected.

### 3. Estimating recall with incomplete ground-truth duplicate graph

As discussed in the evaluation section, the full duplicate graph is unknowable at billion scale. Here are the mathematical frameworks for recall estimation:

**Capture-recapture (Lincoln-Petersen):** Run two independent detection methods (Method A and Method B). Let a = duplicates found by A only, b = duplicates found by B only, c = duplicates found by both. Estimated total duplicates N_hat = (a + c) * (b + c) / c. Recall_A = (a + c) / N_hat. Assumptions: independence of methods, equal catchability. In practice, methods are correlated (both catch easy cases), so this overestimates recall. Use Chapman's correction for small samples.

**Sampling-based recall:** Draw a random sample of S items. For each, do exhaustive (exact) search against the full corpus. Let T = total true duplicates found in the sample. Let D = duplicates that the production system identified. Recall_est = D / T. Confidence interval: use the binomial proportion CI. For 95% confidence with recall ~0.85 and S=1000, the CI is approximately +/- 2.2%.

**Mark-recapture with synthetic injection:** Inject K known duplicate pairs into the corpus. Run the production system. Let K' = number detected. Recall_est = K'/K. Advantage: ground truth is known. Disadvantage: synthetic duplicates may be easier/harder than real ones, biasing the estimate. Use realistic augmentations to minimize this bias.

### 4. How ANN recall affects product-level precision

ANN recall is the fraction of true nearest neighbors returned by the approximate search (vs. exact search). Product-level precision is the fraction of items flagged as duplicate that are truly duplicate. These interact in a subtle way.

**ANN recall < 1 means missed candidates:** Some true duplicates in the corpus are not returned by the ANN search, so the system doesn't even consider them. This directly reduces system recall (missed duplicates). But it can also indirectly affect precision: if the ANN search returns a biased subset of neighbors (e.g., only very close neighbors), the precision on the returned set may be artificially high.

**Quantifying the impact:** Let ANN_recall = 0.95 (5% of true top-k neighbors are missed). If the system's precision on exact search results is P_exact, then: System_recall = ANN_recall * P_exact_recall (roughly). System_precision ≈ P_exact (slightly higher because missed candidates are more likely to be borderline). So ANN recall primarily affects recall, not precision.

**Practical implication:** An HNSW index with recall@10 = 0.95 means you miss 5% of true duplicates in the retrieval stage. If your target system recall is 0.90, you need your decision layer to have recall > 0.90 / 0.95 = 0.947 on the retrieved candidates. This is the "recall budget" — you allocate some recall loss to ANN approximation and require the rest from the decision layer.

**Mitigation:** Increase ANN recall by tuning search parameters (higher efSearch for HNSW, higher nprobe for IVF). But this trades off against latency. At efSearch=64, HNSW gives ~0.95 recall with ~5ms latency. At efSearch=128, ~0.98 recall with ~8ms latency. At efSearch=256, ~0.99 recall with ~15ms latency. Choose based on your recall budget and latency SLO.

### 5. Transitive closure applied naively — what goes wrong

Given pairs (A,B) with sim=0.93 and (B,C) with sim=0.91, transitive closure groups {A,B,C}. But sim(A,C) might be only 0.78 — below the duplicate threshold. The cluster contains a false positive (A and C are not duplicates).

**Mathematical analysis:** In a d-dimensional embedding space with cosine similarity, the triangle inequality for cosine distance (1 - cos) gives: dist(A,C) <= dist(A,B) + dist(B,C). With thresholds: dist(A,B) = 1 - 0.93 = 0.07, dist(B,C) = 1 - 0.91 = 0.09. Upper bound on dist(A,C) = 0.16, so sim(A,C) >= 0.84. But this is a loose bound — the actual sim(A,C) can be anywhere in [0.84, 1.0]. If your threshold is 0.90, transitive closure can incorrectly include pairs with similarity as low as 0.84.

**Chain length amplification:** With a chain of length L, the worst-case similarity between endpoints is: sim_min = 1 - L * (1 - threshold). For threshold = 0.92 and L = 5 hops, sim_min = 1 - 5 * 0.08 = 0.60. This is catastrophic — items with 0.60 similarity (barely related) end up in the same cluster.

**Solutions:** (1) Complete-linkage clustering: require all pairs in a cluster to exceed the threshold. This is O(n^2) within each cluster but clusters are typically small. (2) Star clustering: one "centroid" item per cluster; all members must be within threshold of the centroid (not of each other). Prevents chaining. (3) Correlation clustering: formulate as an optimization problem (minimize disagreements between pairwise labels), solve with LP relaxation or greedy algorithms.

### 6. Mining hard negatives for retraining

Hard negatives are pairs that the current model scores as similar (high cosine) but are actually not duplicates. They are the most valuable training examples for improving the embedding model because they target the model's specific failure modes.

**Mining strategy 1 — From false positive logs:** The production system's false positive log (pairs flagged as duplicate that humans labeled as not-duplicate) is a direct source of hard negatives. Each false positive is a (anchor, hard_negative) pair. Combine with known duplicates as positives for contrastive training.

**Mining strategy 2 — In-batch hard negatives:** During training, for each anchor, find the closest non-duplicate in the same mini-batch. This is computationally cheap (similarity matrix is already computed) and provides increasingly harder negatives as training progresses.

**Mining strategy 3 — ANN-mined negatives:** For each item in the training set, query the ANN index and retrieve the top-k neighbors. Filter out known duplicates. The remaining neighbors are hard negatives (the model thinks they're similar but they're not). This scales to large corpora.

**Training protocol:** Use a contrastive loss (InfoNCE, NT-Xent) or triplet loss with hard negatives. Be careful with the hard negative ratio — too many hard negatives can destabilize training (the model oscillates between pushing apart hard negatives and pulling together positives). A ratio of 1 positive : 3-5 hard negatives : 5-10 random negatives works well in practice. Use temperature scheduling to gradually increase the difficulty of negatives during training.

### 7. Quantifying whether dedup improved downstream model quality

**Controlled experiment (gold standard):** Train model M1 on the full (un-deduped) dataset D. Train model M2 on the deduped dataset D' (same architecture, same hyperparameters, same compute budget). Compare M1 and M2 on a held-out benchmark. If M2 >= M1, dedup helped. If M2 < M1, dedup was too aggressive (removed useful data).

**Efficiency framing:** If M2 ≈ M1 but |D'| < |D|, dedup saved training compute without losing quality. Compute savings = (|D| - |D'|) / |D| * training_cost. If training costs $100K and dedup removes 20% of data, you saved $20K per training run.

**Data quality metrics (proxies):** (1) Dataset entropy: compute the entropy of the label distribution or the embedding distribution. Dedup should increase entropy (more diverse dataset). (2) Effective dataset size: the number of independent data points (accounting for redundancy). Related to the trace of the data covariance matrix or the number of significant principal components. Dedup should increase the effective-to-actual size ratio. (3) Training loss curvature: if the model's loss curve plateaus earlier on the deduped data, it's reaching the same information content faster.

**Causal attribution challenge:** In practice, multiple changes ship simultaneously (new data, new dedup, new training recipe). Isolating dedup's contribution requires A/B testing on the data pipeline, which is expensive. A pragmatic approach: maintain a dedup impact log that tracks (dedup_version, training_run_id, benchmark_results). Over many training runs, regress benchmark results against dedup parameters to estimate the causal effect.

---

## Ten Nasty Interviewer Pivots

### 1. "The corpus is now multilingual — 40% English, 20% Chinese, 15% Japanese, 10% Korean, 15% other."

Multilingual corpora break assumptions about similarity distributions. English sentences are more spread out in embedding space (large pretraining data) while low-resource languages are more clustered (smaller pretraining data). A universal threshold gives poor precision on low-resource languages and poor recall on English.

**Response:** Use per-language thresholds calibrated on per-language evaluation sets. For the embedding model, use a multilingual encoder (multilingual-E5-large, mE5) rather than an English-only model. Critically, cross-lingual dedup is usually not desired — a Chinese translation of an English prompt is a different training example. Add a language_id metadata filter so that dedup only happens within the same language.

If cross-lingual dedup IS desired (e.g., detecting translated duplicates), use a cross-lingual embedding model (LaBSE, multilingual CLIP for images with text) and be aware that cross-lingual similarity scores are systematically lower than within-language scores. Set separate thresholds for within-language and cross-language comparisons.

Monitor per-language dedup rates. If Chinese items are being deduped at 3x the rate of English items, investigate — it may be a calibration issue or a genuine data quality issue (more redundancy in Chinese data sources).

### 2. "Now we need strict tenant isolation — no shared index, no shared GPU."

This is the most expensive isolation requirement. Per-tenant dedicated infrastructure means no resource sharing, no economies of scale for small tenants.

**Response:** Implement a tiered isolation model. Tier 1 (enterprise, contractual requirement): fully dedicated infrastructure — dedicated GPU fleet for embedding, dedicated index servers, dedicated storage. Cost: $5K-20K/month per tenant. Tier 2 (standard): shared GPU fleet with job-level isolation (separate containers, no shared memory), per-tenant indexes on shared hardware. Cost: $500-2K/month per tenant. Tier 3 (self-serve): fully shared infrastructure with software-level isolation. Cost: $50-200/month per tenant.

For Tier 1, the architecture is a per-tenant deployment of the entire stack, orchestrated by Kubernetes namespaces or separate clusters. The embedding model weights can be shared (they're public pretrained models) but must be served from dedicated GPU instances. The index is on dedicated memory. Logs go to a tenant-specific log sink.

Discuss the operational complexity: N tenants x M components = N*M deployments to manage. Use infrastructure-as-code (Terraform, Pulumi) and GitOps to manage this at scale. Each tenant gets a configuration file that specifies their tier and resource allocation.

### 3. "We shipped a new embedder and it's 5% better overall but 20% worse on our most critical AV safety slice."

This is a classic metric conflict. The global metric improved but a critical slice regressed. You cannot ship this model.

**Response:** Do not deploy. The AV safety slice is the highest-priority slice — regression there is a guardrail violation. Investigate why the new model is worse on this slice: (1) Did the fine-tuning data under-represent AV safety scenarios? (2) Did the new model collapse representations for rare safety-critical objects? (3) Is the regression in the embedding space itself or in the threshold calibration?

If the regression is in calibration, try per-slice threshold re-calibration with the new model. If the regression is in the embedding space, you have two options: (a) Continue fine-tuning with oversampled AV safety data (targeted improvement). (b) Use a mixture-of-experts approach: route AV safety queries to the old model and all other queries to the new model.

Establish a formal quality gate: no model ships unless it meets minimum performance on every critical slice. Define critical slices upfront with stakeholders.

### 4. "The index build now takes 18 hours and it's blocking deployment."

HNSW index construction is O(n * log(n)) with large constants. At 1B items with 768d, M=32, build time is ~18 hours on a single machine. This is too slow for rapid iteration.

**Response:** (1) **Parallelize construction:** Use FAISS's GPU-based IVF training (k-means on GPU) and parallel HNSW construction. FAISS supports multi-threaded HNSW building. With 32 cores, build time drops to ~2-3 hours. (2) **Incremental updates:** Don't rebuild from scratch every time. HNSW supports online insertion — add new items incrementally. Rebuild only when the graph quality degrades significantly (monitor recall over time; rebuild when recall drops >2% from the fresh-build baseline). (3) **Use IVF-PQ for the primary index and HNSW for a smaller "hot" partition:** The hot partition contains recent items (last 30 days) in an HNSW index that's fast to build (<1 hour). The cold partition uses IVF-PQ over the full corpus, which trains faster. (4) **Precompute and shard:** Split the corpus into K shards, build HNSW on each shard in parallel, and use distributed search across shards. K=10 shards of 100M items each build in ~2 hours each, and can be built in parallel.

### 5. "The legal team needs us to explain why a specific item was flagged as duplicate."

Embedding-based decisions are inherently opaque. "The cosine similarity was 0.93" is not a satisfying explanation for legal/compliance teams.

**Response:** Build an explanation layer on top of the decision. (1) **Retrieve the matched item:** Show the item and its identified duplicate side-by-side. This is the most powerful explanation — humans can see the similarity. (2) **Highlight similar regions:** For images, use attention map overlays (GradCAM or DINOv2's attention heads) to show which regions the model focused on. For text, highlight shared n-grams or paraphrased passages. (3) **Provide the similarity score in context:** "This item has a similarity score of 0.93, which is in the top 0.1% of all pairwise similarities for this data type. Our threshold is 0.92." (4) **Show the decision trail:** Log entry showing the exact model version, threshold, and decision logic that produced the flag.

For GDPR/regulatory compliance, the explanation must be "meaningful information about the logic involved." The combination of side-by-side display + attention highlighting + contextual score is generally sufficient. Document this explanation framework and review with legal counsel.

### 6. "The annotation team says the system is flagging items that operators agree are different."

This is a precision problem, and the operators' judgment is the ground truth that matters. The system is producing false positives that waste operators' time and erode trust.

**Response:** (1) Collect the flagged items — these are high-value training examples (hard negatives from production). (2) Analyze patterns: are the false positives concentrated in a specific slice, source, or template? (3) If pattern exists, adjust the threshold for that slice or add template-aware preprocessing. (4) If no pattern, the model's embedding space doesn't discriminate well enough for this data. Trigger a fine-tuning cycle with the operator-labeled negatives. (5) Short-term fix: raise the threshold by 0.02-0.03 to reduce false positives immediately, accepting some recall loss. Communicate the change to the operators and ask for continued feedback.

**Process improvement:** Create a direct feedback channel between operators and the dedup system. An "I disagree" button in the annotation tool that logs the disagreement, exempts the item from dedup, and adds the pair to the training/evaluation set. This turns every false positive into a system improvement.

### 7. "We're losing long-tail diversity — rare categories are being deduped into oblivion."

This is the feedback loop collapse scenario from the continual improvement section. The dedup system is correctly identifying similarity but incorrectly equating "similar rare item" with "duplicate."

**Response:** (1) Implement category-aware dedup budgets: for each category/scenario, set a minimum retention count. If a category has fewer than K items (e.g., K=1000), disable dedup for that category entirely. (2) Use rarity-weighted thresholds: raise the threshold for rare items (require higher similarity to flag as duplicate). Rarity = 1 / (number of items in the same cluster/category). (3) Diversity audit: weekly report on per-category item counts and dedup rates. Flag categories where dedup rate > 50% or item count < minimum threshold. (4) Preserve-first policy for rare items: if an item's nearest cluster has fewer than K members, never auto-dedup — always route to human review.

### 8. "Adversary discovers that embedding vectors can be inverted to reconstruct original items."

Recent research (e.g., Vec2Text for text embeddings, image reconstruction from CLIP embeddings) shows that embeddings can be partially inverted. This is a privacy concern if embeddings are stored for items containing PII, proprietary content, or sensitive data.

**Response:** (1) Treat embeddings as sensitive data — apply the same access controls and encryption as the original items. (2) Consider dimensionality reduction: PCA to 128-256 dims reduces invertibility while preserving similarity structure for dedup. The information loss from PCA makes inversion significantly harder. (3) Differential privacy: add calibrated noise to embeddings before storage. This provably limits inversion accuracy but also reduces dedup precision. Tune the noise level to achieve an acceptable privacy-utility tradeoff. (4) Avoid storing embeddings longer than necessary. If an item is deduped and removed, delete its embedding too. (5) For the most sensitive tenants, compute similarity online (embed, search, discard embedding) without persisting the embedding. This is more expensive but eliminates the stored-embedding attack surface.

### 9. "We need the system to work for both online ingest blocking AND offline deep cleanup simultaneously."

This is covered in core design question 10 but the interviewer is pushing for consistency and interaction between the two modes.

**Response:** The key challenge is consistency — an item flagged as "not duplicate" at ingest (fast, shallow check) may later be identified as duplicate in offline cleanup (slow, deep check). How does the system handle this conflict?

Design a state machine for each item: `INGESTED -> CHECKED_ONLINE -> {DUPLICATE_ONLINE | PENDING_OFFLINE | CLEAN_ONLINE} -> CHECKED_OFFLINE -> {DUPLICATE_OFFLINE | CLEAN}`. The online check produces an intermediate decision. The offline check produces the authoritative decision. If they conflict (online said clean, offline says duplicate), the offline decision wins — but the item may have already been sent for annotation. In that case, mark the annotation as "from-duplicate-source" metadata (don't discard completed annotations, but weight them lower in downstream aggregation).

Shared components: embedding service, embedding store, feedback pipeline. Separate components: online index (HNSW, low latency, incremental updates) and offline index (IVF-PQ, high throughput, periodic rebuild). The offline index is a superset of the online index and catches items the online index missed due to index staleness or ANN recall loss.

### 10. "Image dedup works great, but text dedup is terrible — too many false positives."

Text embeddings have higher baseline similarity than image embeddings. Two random English sentences might have cosine similarity of 0.3-0.5 (due to shared vocabulary and syntax), while two random natural images might have cosine similarity of 0.0-0.2. This means text dedup requires much higher thresholds or a fundamentally different approach.

**Response:** (1) Re-calibrate text thresholds independently — don't reuse image thresholds. Typical text dedup thresholds are 0.92-0.97 (much higher than image thresholds of 0.88-0.93). (2) Use a text-specific re-ranking signal: after embedding-based retrieval, compute exact token overlap (Jaccard on word n-grams) as a second signal. High embedding similarity + low token overlap = probably not a duplicate (semantically similar but different text). (3) Consider text-specific hashing (MinHash with n-grams) as the primary dedup method, with embedding-based search as a secondary signal. MinHash is well-suited for text dedup and has tunable sensitivity. (4) For instruction-tuning data specifically, use instruction-response decomposition: embed the instruction and response separately. Two items are duplicates only if BOTH instruction and response are similar (not just one). This significantly reduces false positives from different responses to similar prompts.

---

## Quick Reference: Key Numbers

| Parameter | Online Path | Offline Path |
|---|---|---|
| Latency budget | p50 < 10ms, p99 < 50ms | N/A (throughput-optimized) |
| k (neighbors) | 10-20 | 50-100 |
| Typical threshold (images) | 0.92-0.95 cosine | 0.88-0.92 cosine |
| Typical threshold (text) | 0.94-0.97 cosine | 0.90-0.95 cosine |
| Embedding dim | 768 (images), 1024 (text) | Same |
| Corpus size target | 100M-1B | Same |
| Memory per 1B items (PQ) | ~128GB (index + graph) | ~64GB (IVF-PQ) |
| Embedding cost (1B items) | ~$1,400 (one-time) | Same |
| Serving cost (1B items) | ~$1,000/month | ~$200/month |
| Human review budget | 500-2000 pairs/day | Same |
| Index rebuild time (100M) | 2-4 hours (HNSW) | 1-2 hours (IVF-PQ) |
| ANN recall target | >0.95 | >0.99 |
| Precision target | >0.95 | >0.90 (routes to review) |
| Recall target | >0.80 | >0.95 |
