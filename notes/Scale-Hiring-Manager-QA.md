# Hiring Manager Round (Ben Levin) — Scale AI Interview Prep

**Focus**: Physical AI, data engines, execution leverage, robotics direction fit

**Target tone**: Calm, business-aware, technical but not performative, honest about failure, concrete about impact, clear on why Scale specifically

---

## 1. What drew you specifically to Scale's AV/CV and robotics group?

Three things converged. First, I have spent the last several years building perception and planning systems for autonomous vehicles and robotics — camera-based 3D detection, BEV fusion, trajectory prediction, sensor calibration. I know firsthand that the bottleneck in these systems is almost never the architecture; it is the data loop. The team that controls data quality, annotation throughput, and evaluation rigor controls the pace of the entire field. Scale is that team.

Second, Ben's blog post on expanding the data engine for physical AI made the thesis concrete: as embodied models scale up (VLAs, world models, sim-to-real transfer), the annotation surface area explodes — you go from 2D bounding boxes to dense 3D trajectories, contact labels, force annotations, and multi-modal alignment checks. The infrastructure to generate, curate, and evaluate that data at scale does not exist in most robotics companies. They build bespoke pipelines that rot. Scale is building the general platform, and that is the higher-leverage position.

Third, the customer surface is uniquely interesting. Working with multiple OEMs and robotics companies simultaneously means the team sees failure modes across the entire industry, not just one stack. That cross-pollination of edge cases, label taxonomies, and evaluation strategies is something you cannot get at a single AV or robotics company. I want to work where the learning rate is highest, and that is at the infrastructure layer.

Finally, on a personal level, I have always gravitated toward the intersection of rigorous engineering and real-world deployment. Physical AI is one of the few remaining domains where you cannot hide behind benchmark gaming — the car has to actually drive, the arm has to actually grasp. Scale's position as the data and evaluation backbone for these systems means the work has immediate, measurable consequence.

---

## 2. Why are you interested in physical AI infrastructure rather than only model research?

Model research is important, but I have seen enough production ML to know that the lasting competitive advantage rarely lives in the model checkpoint. It lives in the data flywheel, the evaluation harness, and the infrastructure that lets you iterate faster than your competitor. A 2% architecture improvement gets matched within months; a data engine that surfaces and fixes tail-case failures 3x faster compounds indefinitely.

In physical AI specifically, infrastructure matters even more than in language or vision. The data is high-dimensional (multi-sensor, temporal, spatial), the annotation is expensive and error-prone, and the evaluation requires domain-specific simulation or replay. A research paper can propose a new loss function for grasp prediction, but someone has to build the pipeline that collects contact-force labels at scale, detects annotator disagreement on ambiguous grasps, routes those to expert review, and feeds the corrected labels back into training. That is the real system, and it is largely an infrastructure and systems problem.

I also believe we are at an inflection point where the model architectures for physical AI (VLAs, diffusion policies, world models) are converging faster than the data infrastructure to feed them. The RT-2 to pi0 to Gemini Robotics progression shows that scaling up data quality and diversity matters more than architectural novelty. Scale is positioned to be the enabling layer for that scaling, and I would rather work on the binding constraint than on incremental architecture changes.

From a career perspective, I want to build systems that outlast any single model generation. Models get replaced; the data engine, the annotation ontology, and the evaluation framework persist and accumulate value. That is the kind of work I find most fulfilling.

---

## 3. Which project best demonstrates your ability to create leverage across data, modeling, and systems?

The best example is building an end-to-end camera-based 3D perception pipeline for autonomous driving. The system had to go from raw surround-view camera images to production-quality 3D detections and BEV occupancy predictions, deployed on embedded hardware with strict latency constraints.

The leverage came from treating data, model, and system as one unified problem. On the data side, I built an automated mining pipeline that identified high-value training frames — scenes with rare object classes, unusual occlusion patterns, or edge-case geometries. Instead of labeling everything uniformly, we focused annotation budget on the frames that would most improve tail-case performance. This reduced annotation cost by roughly 40% while improving mAP on rare classes by 8 points.

On the modeling side, I iterated between BEVFormer-style architectures and simpler lift-splat-shoot baselines, choosing based on the actual deployment latency budget rather than benchmark numbers. The key insight was that the architecture choice was secondary to the training data distribution — once we fixed the data pipeline to over-sample hard cases, the simpler model nearly matched the more complex one at a fraction of the latency.

On the systems side, I built the evaluation harness that closed the loop: production inference results were compared against high-confidence ground truth in a nightly pipeline, failure cases were automatically triaged into annotation queues, and model retraining was triggered when enough new labeled data accumulated. This created a genuine flywheel — every production mile improved the next model iteration without manual intervention. The leverage was not in any one component but in the tight coupling between all three.

---

## 4. Tell me about a time you turned a messy technical problem into a shipped capability.

At one point, I inherited a camera-based depth estimation module that was critical for 3D object detection but notoriously unreliable. The existing system used a monocular depth network trained on LiDAR-projected ground truth, but it had severe artifacts: depth discontinuities at object boundaries were noisy, far-range estimates were biased, and the model would silently degrade in rain or low-light conditions without any detection mechanism. Engineers had been patching it for months with heuristic post-processing filters, each one introducing new failure modes.

The first thing I did was not touch the model. I built a diagnostic pipeline that systematically measured depth error as a function of range, weather, time-of-day, and object class. This took about a week and immediately revealed that 70% of the error budget came from two root causes: (1) the LiDAR ground truth itself was noisy at long range due to sparse returns, and (2) the training set was heavily biased toward daytime, clear-weather scenes.

With that diagnosis, the fix was straightforward but required coordination across teams. I worked with the data team to re-filter the ground truth using multi-frame LiDAR accumulation and confidence thresholds, eliminating the noisy supervision signal. I worked with the collection team to prioritize night and rain drives. And I swapped the monolithic depth network for a simpler architecture with an explicit aleatoric uncertainty head, so downstream consumers could discount low-confidence depth estimates rather than trusting everything equally.

The shipped result was a depth module that reduced median depth error by 35%, included calibrated uncertainty estimates, and had a monitoring dashboard that flagged distribution shift in production. The key lesson was that "messy" problems are usually messy because the diagnosis was never done properly, not because the underlying fix is hard.

---

## 5. Tell me about a project failure that made you better.

Early in my career, I spent three months building a real-time multi-object tracker for an autonomous vehicle stack. I was proud of the architecture — it used a graph neural network for association, a Kalman filter bank for state estimation, and a learned re-identification module for handling occlusions. It worked beautifully on the evaluation dataset.

When we deployed it on the vehicle, it fell apart. The tracker would lose objects when they went behind pillars, hallucinate tracks from sensor noise, and occasionally swap identities between adjacent vehicles. The framerate also dropped below spec on the target hardware. The evaluation dataset had not captured any of these failure modes because it was curated from clean, highway driving — the production environment was dense urban with frequent occlusions and sensor artifacts.

The failure taught me three permanent lessons. First, evaluation must be adversarial and representative, not convenient. I now always insist on building evaluation sets that include the hardest known cases before claiming a system works. Second, complexity is a liability in production. The graph neural network was elegant but fragile; a simpler Hungarian-assignment tracker with good heuristics would have shipped months earlier and been easier to debug. I had over-indexed on architectural novelty because it was more interesting, not because it was the right engineering decision. Third, I learned to separate "works on my machine" from "works in production" by deploying early prototypes into shadow mode and measuring real-world failure rates before committing to a design.

That failure changed how I approach every project since. I now front-load the ugly work — building robust evaluation, profiling on target hardware, and testing against adversarial inputs — before investing in model sophistication. It is less exciting but dramatically more effective.

---

## 6. How do you decide whether to improve data, architecture, loss, or infrastructure first?

I use a simple diagnostic framework. Before changing anything, I run three checks: (1) What does the error analysis say? (2) Where is the current bottleneck in iteration speed? (3) What is the cheapest experiment that would give the most information?

For error analysis, I break down model failures into categories: is the model seeing the right data and still getting it wrong (model capacity or loss issue), or is the model never seeing the relevant patterns (data issue)? If I look at the worst-performing slices and the training set has poor coverage of those cases, the answer is data. If the training set covers those cases but the model still fails, I look at whether the loss function properly incentivizes the right behavior or whether the architecture lacks the representational capacity. This triage takes a day or two and prevents weeks of wasted effort.

For iteration speed bottleneck, I ask: how long does it take to go from "I have an idea" to "I have a number that tells me if the idea worked"? If training takes a week because the data pipeline is slow, fixing infrastructure first has the highest ROI because it accelerates every future experiment. If training is fast but we do not have the right evaluation metric, fixing evaluation comes first. If we can iterate quickly but every experiment gives the same result, it is time to step back and question the data or the problem formulation.

The cheapest-experiment heuristic guards against over-thinking. If I can test whether the problem is data quality by manually cleaning 500 labels and retraining on a small subset overnight, that is always better than spending a week designing a new architecture that might not address the root cause. I bias heavily toward experiments that are fast, informative, and disposable. In practice, the answer is usually data first, then loss/evaluation, then architecture, then infrastructure — but the framework ensures I am not just defaulting to a habit.

---

## 7. What is your framework for dealing with ambiguity in early-stage product or research environments?

My framework has three pillars: constrain the problem, find the fastest path to a concrete artifact, and create reversible decisions by default.

Constraining the problem means getting explicit about what we are not doing. In ambiguous environments, the danger is not that people do nothing — it is that they try to do everything simultaneously and make no progress on anything. I push for a written one-pager that states: here is the specific user/customer problem, here is the minimum artifact that would prove we can solve it, here is what is explicitly out of scope for the next two weeks. The act of writing it down forces clarity and surfaces disagreements early.

Finding the fastest path to a concrete artifact means building the simplest version of the thing that could generate real signal. In an ML context, this might mean training a small model on a small dataset and evaluating it on a handful of real customer cases — not to get good numbers, but to learn what breaks. The artifact is a forcing function: it gives everyone something concrete to react to instead of debating abstractions. I have seen teams spend weeks arguing about system architecture when building a rough prototype in three days would have resolved the argument empirically.

Creating reversible decisions means biasing toward choices that are easy to undo. Use a simple model that can be swapped later rather than committing to a complex framework. Use a flat data format that can be restructured rather than an elaborate schema. Deploy behind a feature flag rather than directly to customers. The goal is to maintain optionality while still making forward progress. The one exception is when a decision has clear increasing cost of reversal — like a data format that customers are already ingesting — in which case I invest more upfront in getting it right.

Throughout all of this, I communicate uncertainty explicitly. I tell stakeholders: "Here is what I know, here is what I do not know, here is the experiment that will resolve the uncertainty, and here is when we will have the answer." People can handle ambiguity; what they cannot handle is surprise.

---

## 8. Describe a situation where the technically elegant answer was not the right business answer.

I was working on a perception system where we had the opportunity to implement an end-to-end differentiable pipeline — a single neural network that took raw sensor data and produced planning-ready outputs, eliminating the hand-designed intermediate representations. Technically, this was beautiful: fewer moving parts, end-to-end optimization, and the potential for the model to learn representations we would never have designed manually.

The business reality was different. Our safety case depended on being able to explain intermediate outputs to regulators and internal safety reviewers. The end-to-end model produced a latent representation that was not human-interpretable — we could not point to it and say "the system detected this vehicle at this location with this velocity" in a way that satisfied the safety team. Additionally, when the system made errors, debugging was nearly impossible because there were no intermediate checkpoints to inspect.

I advocated instead for a modular architecture with explicit 3D detection, tracking, and prediction stages. Each stage had interpretable outputs, could be evaluated independently, and could be improved in isolation. It was less elegant — there were hand-designed interfaces between modules, and the system could not jointly optimize across stage boundaries. But it shipped, it was debuggable, it satisfied the safety case, and each module could be upgraded incrementally without revalidating the entire pipeline.

The lesson I took away is that "elegant" and "right" are different optimization objectives. Elegance optimizes for intellectual coherence; "right" optimizes for the full set of constraints including debuggability, organizational readability, regulatory requirements, and speed of iteration. In a production setting, especially in safety-critical physical AI, I will always choose the solution that is boring, debuggable, and shippable over the one that is clever and fragile. The technically elegant solution often becomes the right solution later, once the problem is better understood and the organizational infrastructure exists to support it — but premature elegance is a common cause of project failure.

---

## 9. How have you worked with annotation, evaluation, or human feedback loops before?

Annotation quality has been central to almost every ML system I have built. In autonomous driving perception, the ground truth comes from human annotators labeling 3D bounding boxes, lane markings, and semantic segmentation in point clouds and images. I have worked directly with annotation teams to design label ontologies, write annotation guidelines, build consensus workflows, and measure inter-annotator agreement.

One concrete example: I discovered that our 3D bounding box annotations had a systematic size bias — annotators were consistently making boxes too large because the annotation tool made it easier to expand than to contract. This was subtle enough that it did not show up in aggregate quality metrics, but it was degrading our model's ability to estimate object dimensions accurately. The fix was a combination of tooling changes (better snapping heuristics in the labeling UI), guideline updates (explicit size calibration examples), and a targeted quality audit with golden-set comparisons. After the fix, our dimension estimation error dropped by 20%.

I have also built evaluation pipelines that use human judgment as the final arbiter. For tasks where automated metrics are insufficient — like evaluating whether a predicted trajectory is "reasonable" even if it does not match the ground truth exactly — I designed human evaluation protocols with calibrated raters, inter-rater reliability checks, and stratified sampling to ensure coverage of edge cases. The key insight is that human feedback loops need the same engineering rigor as the ML pipeline: you need version control on guidelines, monitoring on annotator consistency, and automated detection of label drift.

This experience maps directly to Scale's core business. Scale is fundamentally in the business of making human feedback loops scalable, consistent, and high-quality. I understand the failure modes — annotator fatigue, guideline ambiguity, consensus gaming, distribution shift in the task stream — and I know how to build systems that detect and correct for them. For physical AI specifically, the annotation challenge is even harder because the data is 3D, temporal, and often requires domain expertise (understanding physics, contact dynamics, force magnitudes). I am excited to work on these harder annotation problems at Scale's infrastructure level.

---

## 10. What type of impact do you want to have in your first six months?

In the first six months, I want to accomplish three things: ship a measurable improvement to a customer-facing capability, build internal credibility through technical depth, and develop a point of view on where the physical AI data engine should go next.

For the first goal, I want to find a concrete problem — a quality gap, a throughput bottleneck, a missing evaluation capability — and close it end-to-end. I am not particular about whether this is a modeling problem, a data pipeline problem, or an infrastructure problem. The point is to demonstrate that I can take a problem from diagnosis through solution to measurable customer impact within weeks, not months. In my experience, the fastest way to earn trust on a new team is to ship something useful quickly, even if it is small.

For the second goal, I want to become the person the team turns to for perception and robotics domain expertise. This means doing thorough code reviews, writing clear technical documents when I find something non-obvious, and being a resource for questions about 3D perception, sensor fusion, BEV representations, or VLA architectures. I want my teammates to feel like the team got stronger when I joined.

For the third goal, I want to develop a thesis about what physical AI customers will need in 12-18 months that Scale is not yet providing. This might be around evaluation methodology for VLA models, data curation strategies for sim-to-real transfer, or annotation tools for contact-rich manipulation. Having a forward-looking point of view is how I transition from "useful contributor" to "strategic asset." I would want to share this thesis with Ben and the team by month six, even if it is rough — having a direction to debate is more valuable than waiting for certainty.

---

## 11. What parts of your background translate most directly to Scale's robotics work?

Four parts of my background translate directly. First, my deep experience in camera-based 3D perception for autonomous driving — BEV representations, multi-camera fusion, depth estimation, occupancy prediction. These are the exact sensing modalities and representations that physical AI systems use, and they are the exact data types that Scale needs to annotate, evaluate, and curate at scale. I understand the failure modes, the data requirements, and the evaluation challenges intimately.

Second, my work on production ML deployment. I have taken models from research through optimization (quantization, distillation, latency profiling) to deployment on embedded hardware with real-time constraints. This means I understand the full lifecycle, not just the training loop. At Scale, this translates to understanding what customers actually need from the data engine — not just labels, but labels that are consistent with the model's deployment constraints, evaluation metrics that correlate with real-world performance, and data pipelines that can iterate at the speed the customer's development cycle requires.

Third, my familiarity with the VLA and robotics perception literature. I have studied the progression from RT-1 through OpenVLA to pi0 and Gemini Robotics. I understand the architectural landscape, the data requirements of different approaches, and where the field is heading. This means I can anticipate what Scale's physical AI customers will need before they ask for it, and I can have technical conversations with customer teams at their level.

Fourth, my experience building data flywheels — automated failure mining, annotation quality loops, evaluation-driven retraining. This is the operational core of what Scale's physical AI team does, and I have built versions of these systems at smaller scale. I understand the difference between a data pipeline that runs and a data engine that learns, and I know how to build the latter.

---

## 12. What would you do if a customer demanded performance on a slice that the current training data barely covers?

This is a common and important problem, and rushing to a solution without proper diagnosis is the most common mistake. My approach has four steps.

First, quantify the gap precisely. How big is the slice? What is the current performance on it? What performance does the customer need? Is the gap due to insufficient training data, or does the model see the data and still fail? This distinction matters enormously — if the model has some training data for the slice and still performs poorly, adding more of the same data may not help. I need to understand whether this is a coverage problem, a quality problem, or a distribution problem.

Second, find the fastest path to a credible estimate of what it would take to close the gap. Can I simulate the slice by augmenting existing data? Can I mine unlabeled data for similar cases and run a quick experiment with a small annotation batch? The goal is to give the customer (and the internal team) a realistic estimate of the cost and timeline, not an optimistic promise that we will "work on it." If closing the gap requires 10,000 new annotations in a rare domain, the customer needs to know that upfront.

Third, explore creative alternatives to brute-force data collection. Can we use synthetic data or simulation to bootstrap coverage? Can we use active learning to identify the most informative examples to label? Can we transfer from a related domain where we have more data? Can we adjust the model's loss function to upweight the rare slice without requiring proportional data? These approaches often get 80% of the benefit at 20% of the cost.

Fourth, communicate transparently with the customer. Here is the gap, here is what we can do in the short term (fast but partial), here is what the full solution requires (slower but complete), and here are the tradeoffs. In my experience, customers respect honesty and a clear plan far more than vague reassurances. If the full solution is expensive, that is a product conversation, not just a technical one — and it is better to have it early.

---

## 13. How do you communicate risk upward when the system is not ready but the pressure is high?

The worst thing you can do is either (a) stay silent and let leadership be surprised, or (b) raise alarms without actionable information. My approach is to communicate risk in a structured format that respects the audience's time while giving them what they need to make decisions.

I use a simple framework: State, Risk, Options, Recommendation. "The system is currently at X level of performance. The risk of shipping at this level is Y (with specific failure mode examples, not abstract concerns). We have three options: (1) delay by N weeks to fix the top issues, (2) ship with guardrails that mitigate the worst failure modes, (3) ship to a limited audience and expand as quality improves. I recommend option 2 because it balances customer expectations with quality, and here is specifically what the guardrails would look like."

The key is being concrete about failure modes. "The model is not ready" is not useful. "The model has a 15% false positive rate on pedestrians in low-light conditions, which means approximately 1 in 7 nighttime scenes will contain a phantom detection that could trigger unnecessary braking" is useful. It gives leadership the information they need to make an informed risk decision, rather than forcing them to either trust my vague judgment or override it blindly.

I also separate my role from the decision. My job is to give leadership the clearest possible picture of the risk and the options. Their job is to make the business decision about which option to take, because they have context I do not (customer relationships, competitive pressure, contractual obligations). If they decide to ship with known limitations, I document the limitations clearly, build monitoring to track the expected failure modes, and create a plan to remediate. I do not take it personally or treat it as a failure of communication — sometimes the right business decision is to ship something imperfect. But I make sure the decision is informed, not accidental.

---

## 14. Describe a time when you had to choose between a platform investment and a short-term feature.

We had a recurring pattern where every new model experiment required significant manual effort to set up: copying configuration files, adjusting data loading scripts, modifying evaluation pipelines, and manually tracking results in spreadsheets. Each individual experiment was not that painful, but we were running 3-5 experiments per week, and the cumulative overhead was substantial — probably 30% of engineering time was going to experiment logistics rather than actual experimentation.

A customer request came in that required a new feature — adding a specific object class to our detection taxonomy. This was a well-understood, two-week task: annotate training data for the new class, add it to the model head, retrain, evaluate, deploy. The business pressure to deliver it was real.

I made the case to invest two weeks in building a proper experiment management system instead: standardized configs, automated data pipeline setup, experiment tracking with metrics dashboards, and one-command evaluation against the full test suite. The argument was arithmetic — the experiment platform would save 1-2 days per experiment, and we were running 15-20 experiments per month. Within two months, the platform investment would pay for itself, and every month after that was pure acceleration.

The compromise we reached was to spend one week on the most impactful parts of the platform (config management and automated evaluation) and one week on the customer feature, delivered slightly later than originally promised. This worked because I was explicit about the tradeoff: "We can deliver the feature one week late, or we can deliver it on time and be 30% slower on everything for the next year." Framing it as a concrete cost rather than an abstract "technical debt" argument made the decision straightforward. The platform investment paid off within six weeks and became one of the most-appreciated internal tools on the team.

---

## 15. What would you optimize first in a robotics data engine: throughput, quality, cost, or iteration speed?

Iteration speed, without hesitation. The reasoning is that iteration speed is a meta-optimization — it accelerates your ability to improve throughput, quality, and cost. If you can go from "identify a problem" to "test a fix" to "measure the result" in hours instead of days, you will converge on good solutions for all the other dimensions much faster.

Concretely, in a robotics data engine, iteration speed means: how quickly can I change an annotation guideline and see the effect on model performance? How quickly can I add a new data source and evaluate whether it helps? How quickly can I modify the quality control pipeline and measure the impact on label consistency? If these feedback loops take days or weeks, you are flying blind — you make changes and hope they work, then wait to find out. If they take hours, you can run multiple experiments per day and make decisions based on evidence.

The practical investments that improve iteration speed are not glamorous: standardized data formats so new sources can be onboarded quickly, modular pipeline components that can be reconfigured without rebuilding everything, lightweight evaluation suites that give directional signal in minutes rather than comprehensive signal in days, and dashboards that surface quality metrics in near-real-time. None of this is novel research, but the teams that build this infrastructure well consistently outperform teams with better models but slower loops.

The one caveat is that quality has a floor. If label quality is so bad that model training is dominated by noise, iteration speed does not help because every experiment gives meaningless results. So the actual answer is: ensure quality is above the noise floor, then optimize for iteration speed, then use that speed to improve throughput and cost. But in my experience, quality below the noise floor is an acute crisis that gets addressed quickly. The chronic problem — the one that silently kills teams — is slow iteration.

---

## 16. How do you think about research taste versus execution discipline?

Both matter, but they matter at different times, and the failure mode of each is different. Bad research taste means you spend months executing perfectly on the wrong problem. Bad execution discipline means you identify the right problem but never ship a solution. In my experience, most teams have more research taste than execution discipline — they can identify interesting problems but struggle to convert insights into shipped systems.

Research taste, to me, means the ability to distinguish between problems that are interesting and problems that are important. In physical AI, an interesting problem might be "can we train a single foundation model for all manipulation tasks?" An important problem might be "our annotation pipeline for contact labels has 30% disagreement rate, and it is degrading every model we train." Research taste tells you that the second problem, while less exciting, is more impactful right now — and that solving it might be a prerequisite for the first problem to even be tractable.

Execution discipline means breaking a problem into concrete milestones, estimating effort honestly, shipping incrementally, and resisting the temptation to redesign mid-flight. It means knowing when a solution is good enough to ship and when it needs more work. It means writing tests, documenting decisions, and building systems that your teammates can maintain after you move on.

The way I balance them: I spend research taste at the beginning of a project (problem selection, approach design, key technical bets) and execution discipline through the middle and end (implementation, evaluation, shipping). I explicitly separate the "exploration" phase from the "exploitation" phase and try not to mix them. During exploration, I prototype freely, read papers, try wild ideas. During exploitation, I commit to an approach and execute methodically. The transition point is when I have enough signal to believe I am working on the right problem with a viable approach — after that, switching approaches has to clear a high bar.

---

## 17. What are you unusually good at that would compound well on this team?

I am unusually good at translating between the model world and the data world. Most ML engineers think primarily in terms of architectures, losses, and training dynamics. Most data engineers think in terms of pipelines, schemas, and throughput. I operate naturally at the interface: I can look at a model's failure modes and trace them back to specific data deficiencies, and I can look at a data pipeline and predict what model behaviors it will produce. This skill compounds on a team building data engines for ML because every decision about data collection, annotation, or curation has downstream model implications, and someone who can reason about both simultaneously avoids expensive mistakes.

I am also unusually good at building evaluation systems that actually correlate with real-world performance. This is harder than it sounds — most evaluation setups measure something, but the something they measure is only loosely connected to what the customer cares about. I have developed a habit of asking "if this metric improves, does the customer's experience actually get better?" and then building the measurement infrastructure to verify the answer. In a data engine for physical AI, evaluation quality is existential — if you cannot reliably measure whether a data change helped, you do not have a flywheel, you have a random walk.

Finally, I am good at writing things down clearly. Technical decisions, failure analyses, system architectures, experiment results — I document them in a way that is useful to people who were not in the room. This is a mundane skill, but on a fast-growing team it compounds significantly. Clear documentation means less repeated work, faster onboarding, and better-informed decisions. In my experience, the teams that write things down well move faster than the teams that keep everything in people's heads, especially as the team scales.

---

## 18. Where are you still developing, and how are you closing the gap?

I am still developing my ability to make product-level decisions that balance customer needs, technical feasibility, and business impact. I am strong on the technical side — I can assess what is feasible and estimate effort accurately. But I am less practiced at the product judgment side: which of three technically feasible features will drive the most customer value? How do you price a new capability? When do you say no to a customer request because it would distort the product? I am closing this gap by deliberately seeking out conversations with product managers and business leaders, reading about how infrastructure companies (Stripe, Twilio, Datadog) made product decisions, and volunteering for projects that require customer-facing judgment rather than purely technical work.

I am also developing my leadership of larger technical initiatives. I have led projects with 2-3 engineers effectively, but I have less experience coordinating work across 5-10 people or across team boundaries. The skills that work at small scale — leading by example, detailed code reviews, pair programming — do not scale to larger groups. I am working on this by focusing on writing clearer project plans, being more explicit about delegation, and investing in communication infrastructure (status documents, decision logs) rather than trying to stay personally involved in every detail.

On the purely technical side, I am deepening my understanding of simulation-to-real transfer for robotics. My background is primarily in real-world sensor data, but as physical AI increasingly relies on synthetic data — domain randomization, physics simulation, procedural generation — I need to be more fluent in the failure modes and best practices of sim-to-real pipelines. I am closing this gap through targeted paper reading (the NVIDIA Isaac ecosystem, Google's simulation work for RT-X) and by building small prototype sim-to-real pipelines to develop hands-on intuition.

I share these honestly because I believe self-awareness about gaps is more valuable than pretending they do not exist. And in every case, I have a specific plan for closing the gap rather than a vague intention to "get better."

---

## 19. What are the warning signs that a data flywheel is fake instead of real?

The most reliable warning sign is that the team talks about the flywheel in presentations but cannot show you a graph of model performance improving monotonically as a function of data iterations. A real flywheel produces measurable, compounding improvement. If the model is not getting better with each data cycle, you do not have a flywheel — you have a data pipeline with a motivational poster.

Second warning sign: the flywheel does not have a failure-driven data collection mechanism. If the data being collected is random or convenience-sampled rather than targeted at the model's current failure modes, the flywheel is not actually learning from its mistakes. A real flywheel identifies what the model gets wrong, collects or annotates more data specifically for those cases, retrains, and verifies improvement. If the collection strategy is not conditioned on the model's errors, you are just accumulating data, not compounding knowledge.

Third warning sign: there is no quality control on the annotation feedback loop. If the labels going back into training have high noise rates, each flywheel iteration can actually make the model worse — you are amplifying errors rather than correcting them. I check for this by asking: what is the inter-annotator agreement on the latest batch? Is there a golden set that measures annotator accuracy? Is there a mechanism to detect and correct label drift? If the answer to any of these is "we don't measure that," the flywheel may be running, but it is not spinning in the right direction.

Fourth warning sign: the evaluation metric does not capture what customers care about. If the flywheel is optimizing for aggregate accuracy but the customer's problem is in the tail — specific scenarios, rare objects, edge cases — then improving the aggregate metric may not improve the customer experience at all. A real flywheel has evaluation metrics that are decomposed by customer-relevant slices, so you can verify that improvement is happening where it matters.

Fifth warning sign: the flywheel has no iteration clock. Real flywheels have a cadence — weekly, biweekly — where a full cycle (collect, label, train, evaluate, identify failures, feed back) completes. If you ask "how often does the flywheel complete a full turn?" and the answer is vague, the system is probably not actually cycling. It is just a batch data pipeline with aspirational branding.

---

## 20. If you joined and discovered the model quality issue is mostly labeling noise, what would you do?

First, I would verify the hypothesis rigorously before acting on it. "Mostly labeling noise" is a strong claim, and acting on an incorrect diagnosis wastes months. I would design a targeted experiment: take a small subset of the training data (maybe 1,000-2,000 examples), have it re-labeled by expert annotators with high inter-annotator agreement, retrain the model on only the clean subset, and compare performance against the model trained on the full noisy dataset. If the clean-subset model performs comparably or better despite having far less data, the labeling noise hypothesis is confirmed. If it performs worse, the issue is more nuanced.

Second, assuming the hypothesis is confirmed, I would characterize the noise. Not all labeling noise is equal. Is it random noise (annotators are inconsistent but unbiased), systematic bias (annotators consistently make the same error, like over-sizing bounding boxes), or task ambiguity (the annotation guidelines are unclear and reasonable annotators disagree)? Each type requires a different fix. Random noise can be addressed with consensus mechanisms — multiple annotators per example with majority voting. Systematic bias requires guideline updates and calibration. Task ambiguity requires ontology redesign and possibly accepting that some labels should be probabilistic rather than deterministic.

Third, I would build monitoring to prevent recurrence. This means: golden sets that measure annotator accuracy continuously, automated detection of annotator drift (is a particular annotator's agreement with the consensus declining over time?), and statistical tests that flag when a batch of new labels has significantly different distributional properties from validated data. The goal is to make label quality a continuously measured quantity, not something you discover is broken months later.

Fourth, I would think about this as a product opportunity, not just a bug fix. If labeling noise is the dominant quality issue, then the ability to detect, measure, and correct label noise at scale is a differentiating capability. Scale's entire value proposition is high-quality data — if we can build tooling that guarantees label quality at a level competitors cannot match, that is a competitive moat. I would propose building this not just as an internal fix but as a customer-facing quality guarantee: "here is our label quality dashboard, here is the measured noise rate, here is how we detect and correct for it." This turns a problem into a feature.

Finally, I would communicate the finding and the plan transparently. Labeling noise as the root cause of model quality issues is a finding that affects multiple teams — the annotation team, the model team, the customer success team. I would write up the analysis, share it broadly, and make sure everyone understands both the problem and the plan. In my experience, quality issues that are visible and well-understood get fixed; quality issues that are hidden or poorly communicated persist indefinitely.
