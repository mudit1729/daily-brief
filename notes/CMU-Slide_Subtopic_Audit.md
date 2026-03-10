# Slide Subtopic Audit

This file is a lecture-by-lecture coverage checklist derived from the slide decks in [course_slides](../course_slides). The intent is operational:

- extract the actual subtopics from each deck
- compare them against the lecture summary markdown
- patch missing concepts, mechanisms, caveats, and visuals

`Status` is the target state for the summary after patching, not a claim that every slide has identical wording.

## Lecture 1
Source PDF: `course_slides/lec1.intro.pdf`
Status: target coverage

- neural networks as a dominant AI paradigm
- contrast with earlier AI / ML approaches
- historical arc from associationism to connectionism
- Aristotle's laws of association
- Hartley / Bain / "information is in the connections"
- brain-inspired vs Von Neumann computation
- McCulloch-Pitts neuron as a threshold logic model
- excitatory vs inhibitory inputs and simple recurrence
- Hebbian learning intuition and instability
- perceptron learning and linearly separable problems
- XOR limitation and why hidden layers matter
- modern successes in speech, vision, segmentation, captioning, and games

## Lecture 2
Source PDF: `course_slides/lec2.universal.pdf`
Status: target coverage

- recap of connectionist intuition and perceptrons
- soft perceptron with sigmoid / tanh activation
- formal MLP structure: source nodes, sink nodes, depth, layers
- linear decision boundaries from a single perceptron
- Boolean composition using multiple threshold units
- XOR as a minimal non-linearly separable example
- square-pulse / interval construction with threshold units
- continuous function approximation in 1D
- extension of function approximation to higher dimensions
- role of width in direct approximation
- role of depth in parameter efficiency
- universal approximation as an existence theorem, not a training theorem
- necessity of nonlinearity for expressive depth

## Lecture 3
Source PDF: `course_slides/lec3.learning.pdf`
Status: target coverage

- learning problem setup from input-output training pairs
- perceptron parameters: weights and bias
- bias implemented as a constant input
- perceptron learning rule and correction on mistakes
- perceptron convergence under linear separability
- hidden-layer credit assignment problem
- ADALINE as differentiable linear learning
- MADALINE as greedy multilayer workaround
- empirical risk minimization viewpoint
- need for a smooth divergence instead of classification error
- scalar derivative intuition
- gradient as steepest-ascent direction
- Hessian and curvature / critical-point type
- gradient descent with learning-rate dependence

## Lecture 4
Source PDF: `course_slides/lec4.learning.presented.pdf`
Status: target coverage

- empirical risk minimization recap
- divergence vs 0/1 classification error
- local linearization intuition for derivatives
- scalar derivative and vector derivative notation
- gradient as transpose / directional derivative geometry
- inner-product argument for steepest descent
- Hessian intuition and stationary points
- chain rule along multiple dependency paths
- forward pass through a layered network
- output-layer gradient from the divergence
- propagation from outputs to affine variables
- gradients for weights and biases
- recursive propagation to earlier layers
- why backprop is efficient dynamic programming
- vanishing-gradient intuition from repeated multiplication

## Lecture 5
Source PDF: `course_slides/lec5.pdf`
Status: target coverage

- full notation for multilayer networks
- total training loss as sum / average over examples
- total derivative as sum of per-example derivatives
- calculus refresher: differential approximation
- multivariate chain rule in matrix form
- forward equations for affine values and activations
- output-layer derivative for regression
- output-layer derivative for classification
- recursive delta equations for hidden layers
- weight-gradient matrix shapes and orientation
- bias-gradient computation
- activation derivatives for sigmoid / tanh / ReLU
- softmax plus cross-entropy simplification
- complete forward-backward algorithm
- minibatch averaging, gradient checking, and complexity

## Lecture 6
Source PDF: `course_slides/lec6.pdf`
Status: target coverage

- training recap and backprop-derived gradients
- backprop does not directly optimize classification error
- perceptron vs backprop counterexample / failure to separate
- quadratic convergence analysis in 1D
- monotone vs oscillatory vs divergent updates
- optimal learning rate for a quadratic
- multivariate quadratic and diagonal Hessian intuition
- condition number as the source of slow convergence
- Newton method and Hessian inversion idea
- why second-order methods are expensive for deep nets
- non-convexity and negative curvature issues
- learning-rate decay schedules
- RProp sign-based step adaptation
- QuickProp as per-parameter curvature approximation
- momentum and Nesterov acceleration

## Lecture 7
Source PDF: `course_slides/lec7.stochastic_gradient.pdf`
Status: target coverage

- batch updates as an expensive baseline
- effect of number of samples on update quality
- incremental update intuition
- stochastic gradient descent definition
- need for random shuffling of examples
- noisy gradients as exploration and regularization
- decaying step-size schedules for SGD
- minibatch gradients as unbiased estimators
- variance scaling with batch size
- convergence-vs-compute tradeoff
- hardware vectorization benefit of minibatches
- trend-based optimizers to smooth oscillations
- RMSprop intuition from squared-gradient tracking
- Adam as first-moment plus second-moment adaptation
- practical batch-size and learning-rate defaults

## Lecture 8
Source PDF: `course_slides/lec8.optimizersandregularizers.pdf`
Status: target coverage

- recap of incremental methods and trend-based updates
- desiderata for a good divergence
- L2 loss for regression
- KL / cross-entropy for classification
- convexity claims and where they hold
- standard output-layer pairings that yield `y - d`
- internal activation drift / batch-normalization motivation
- batch normalization forward equations
- train-time vs test-time statistics
- backpropagation through batch normalization
- L2 regularization / weight decay
- L1 regularization and sparsity
- dropout as stochastic co-adaptation control
- early stopping as implicit regularizer
- data augmentation and gradient clipping

## Lecture 9
Source PDF: `course_slides/Lec9.CNN1.pdf`
Status: target coverage

- scanning for patterns in speech / time signals
- why flat MLPs are poor shift-tolerant detectors
- need for shift / translation invariance
- scanning-window detector as repeated subnet
- giant-network interpretation of scanning
- parameter sharing across positions
- sparse local connectivity
- 2D analogue for images
- filters / kernels / feature maps terminology
- output-size computation with kernel, stride, padding
- training shared parameters by summing gradient contributions
- convolution as the generalization of scanning
- higher-dimensional extension to video / volumetric data

## Lecture 10
Source PDF: `course_slides/lec10.CNN2.pdf`
Status: target coverage

- visual-cortex inspiration from Hubel and Wiesel
- receptive fields and orientation selectivity
- simple cells vs complex cells
- hierarchical composition of responses
- NeoCognitron S planes and C planes
- weight sharing within planes
- increasing receptive field with depth
- decreasing spatial resolution with depth
- Hebbian / unsupervised feature learning in the NeoCognitron
- shift tolerance from complex / pooling-like responses
- LeNet as supervised end-to-end CNN
- multi-map convolution over all input maps
- padding to preserve spatial size
- pooling / subsampling rationale
- classical architecture patterns: AlexNet, VGG, ResNet

## Lecture 11
Source PDF: `course_slides/Lec11.CNN3.pdf`
Status: target coverage

- recap of the generic CNN architecture
- exact convolution equation with map and spatial indices
- interpretation of weight tensor dimensions
- parameter count as kernel size times number of input maps
- max pooling definition
- mean pooling definition
- stride as integrated downsampling
- transposed / fractional-stride intuition
- backpropagation through activation functions
- weight-gradient accumulation over all output positions
- backpropagation to previous maps with flipped filters
- dependence graph for a single map / local receptive field
- receptive-field growth through deeper stacks

## Lecture 12
Source PDF: `course_slides/Lec12.CNN4.pdf`
Status: target coverage

- detailed backpropagation through convolutional layers
- backpropagation through pooling layers
- derivative with respect to affine maps
- derivative with respect to filter weights
- derivative with respect to previous-layer maps
- flipped-filter view of backward convolution
- max-pooling gradient routing to argmax locations
- mean-pooling gradient distribution
- stride inside pooling and convolution
- transposed convolution as learned upsampling
- deformation / jitter sensitivity
- how pooling and stride reduce sensitivity to tiny shifts

## Lecture 13
Source PDF: `course_slides/lec13.recurrent.pdf`
Status: target coverage

- sequence modeling examples from text, sports, and stocks
- language generation as next-symbol prediction
- beginnings and ends of generated sequences
- time-delay neural network as finite-context baseline
- finite response vs infinite response systems
- NARX / output-feedback viewpoint
- explicit recurrent hidden state
- Jordan network
- Elman network
- representational shortcut of unrolling recurrence
- one-input / one-output and sequence classification task types
- why recurrence beats fixed windows on long dependencies

## Lecture 14
Source PDF: `course_slides/lec14.recurrent.pdf`
Status: target coverage

- short-term vs long-term temporal dependence
- parity and addition as canonical recurrence examples
- recurrence can solve variable-size problems with small models
- sequence divergence over outputs
- unrolling the recurrent computation through time
- analyzing recursion in linear systems
- eigenvalue view of memory persistence
- bounded-input bounded-output stability intuition
- saturation in nonlinear recurrence
- vanishing gradients and exploding gradients
- distinction between memory failure and learning failure
- gating as the remedy
- introduction to LSTM cell state and gates

## Lecture 15
Source PDF: `course_slides/lec15.recurrent.pdf`
Status: target coverage

- vanishing memory from repeated recurrent multiplication
- linear-activation memory decay with small eigenvalues
- non-linear-activation fixed-point saturation
- gradient dependence on activation Jacobians and singular values
- why hidden-state behavior is dominated by recurrent weights
- need for input-dependent memory control
- constant error carousel
- forget gate
- input gate and candidate content
- output gate
- full LSTM update equations
- language modeling as a running application
- GRU as a simplified gated alternative
- gradient clipping and forget-gate initialization

## Lecture 16
Source PDF: `course_slides/lec16.recurrent.pdf`
Status: target coverage

- variants of recurrent nets for different sequence problems
- MLP baseline for sequence-to-sequence divergence
- time-synchronous recurrence
- one output per input timestep
- sequence classification setup
- divergence over sequences rather than single examples
- unrolled computation graph
- backpropagation through time first step
- recursive gradient flow through hidden states
- shared-weight accumulation across timesteps
- truncated BPTT and memory/computation tradeoffs
- gradient clipping / practical training issues

## Lecture 17
Source PDF: `course_slides/lec17.recurrent.pdf`
Status: target coverage

- sequence-to-sequence task families
- order-aligned but time-asynchronous case
- recap of training with explicit alignments
- characterization of an alignment path
- timing information as the missing supervision
- problem statement when alignment is absent
- Viterbi / best-alignment approximation
- need to marginalize over all valid alignments
- blank symbol in CTC
- collapse rule: merge repeats then remove blanks
- forward probability recursion
- backward probability recursion
- posterior probability of a symbol at a frame
- CTC loss and gradient intuition
- greedy decoding vs beam search

## Lecture 18
Source PDF: `course_slides/lec18.attention.pdf`
Status: target coverage

- seq2seq recap and fixed-context bottleneck
- language modeling recap with recurrent generation
- next-symbol prediction setup
- start token and end token handling
- encoder-decoder architecture without attention
- why fixed-length context vectors fail on long inputs
- attention scores over encoder states
- normalized attention weights as soft alignment
- context vector as weighted sum of values / states
- decoder update with previous output plus context
- alignment heatmap interpretation
- training vs autoregressive inference
- attention as the bridge to transformers

## Lecture 19
Source PDF: `course_slides/lec19.Txfmr_GCN.pdf`
Status: target coverage

- non-recurrent encoder motivation
- context-specific token embeddings without recurrence
- query / key / value construction
- self-attention score computation
- attention-weighted sum of values
- multi-head attention and diversity of relations
- need for positional encoding
- transformer encoder block components
- transformer decoder block, masking, and cross-attention
- full seq2seq transformer data flow
- moving from sequences and grids to graphs
- graph message passing / graph convolution
- neighborhood aggregation and update rule
- depth as k-hop neighborhood growth
- node-level, edge-level, and graph-level tasks

## Lecture 20
Source PDF: `course_slides/lec20.representations.pdf`
Status: target coverage

- function learning from finite samples and locality assumptions
- trivial linear case and smooth interpolation intuition
- non-linearly separable 1D example
- why tiny data perturbations should not rewrite the whole function
- local windows and neighborhood-based reasoning
- hidden layers as learned feature transforms
- manifold hypothesis
- layerwise probing of learned representations
- logistic regression / probabilistic interpretation of final layer
- neurons as correlation-based feature detectors
- autoencoder objective and bottleneck
- linear autoencoder and PCA subspace
- non-linear autoencoder as manifold learner
- decoder as dictionary / source model
- source separation example
- reconstruction vs generative use

## Lecture 21
Source PDF: `course_slides/lec21.VAE.pdf`
Status: target coverage

- data manifold picture for generation
- why ordinary autoencoders do not guarantee sampleable latent codes
- hidden representation as a distribution, not a single point
- encoder outputting posterior parameters
- decoder mapping latent samples back to data
- prior constraint on latent variables
- Gaussian prior and isotropic-Gaussian properties
- affine projection property of isotropic Gaussians
- KL divergence to the prior
- reconstruction term vs latent regularization term
- reparameterization trick
- training with statistical constraints
- latent interpolation and sampling use cases

## Lecture 22
Source PDF: `course_slides/lec_22_GAN1.pdf`
Status: target coverage

- discriminative vs generative modeling
- conditional vs joint / marginal modeling
- explicit-density vs implicit-sample models
- VAE recap and why sample quality can be blurry
- GAN setup: generator and discriminator
- adversarial min-max objective
- alternating training loop
- optimal discriminator form
- JS-divergence interpretation under ideal assumptions
- non-saturating generator objective
- Nash equilibrium intuition
- GANs vs VAEs as complementary generative families

## Lecture 23
Source PDF: `course_slides/Gans_TA.pdf`
Status: target coverage

- GAN framework recap
- training dynamics in the adversarial game
- qualitative effects of JS divergence
- mode collapse
- vanishing gradients with an over-strong discriminator
- oscillation / non-convergence in min-max games
- feature matching
- minibatch discrimination
- historical averaging
- one-sided label smoothing
- virtual batch normalization
- KL vs JS vs Wasserstein comparison
- WGAN critic and 1-Lipschitz constraint
- weight clipping vs gradient penalty

## Lecture 24
Source PDF: `course_slides/lec26.hopfield.pdf`
Status: target coverage

- loopy binary recurrent network
- local field and sign-matching flip rule
- asynchronous single-unit updates
- why one flip can change the field of others
- energy of a Hopfield net
- monotonic energy decrease under asynchronous updates
- local minima as attractors
- content-addressable memory
- noisy pattern completion
- partial observation completion
- Hebbian storage rule
- spurious minima / parasitic memories
- storage capacity intuition and the `0.14N` result
- stochastic updates, temperature, and annealing intuition

## Lecture 25
Source PDF: `course_slides/lec27.BM.pdf`
Status: target coverage

- Hopfield recap: energy and deterministic evolution
- content-addressable examples and noisy completion
- training the Hopfield energy landscape
- target patterns vs parasitic / confusing patterns
- SGD-style shaping of valleys and attraction basins
- boredom / broad-valley intuition in energy shaping
- shift from deterministic attractors to stochastic updates
- Boltzmann update probabilities
- equilibrium distribution over states
- partition function and why learning is hard
- visible vs hidden units
- positive phase vs negative phase
- restricted Boltzmann machines
- contrastive divergence
- Hopfield vs Boltzmann interpretation
