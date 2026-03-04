# Classical Machine Learning Interview Questions

*Comprehensive Guide to Traditional ML Concepts*

This document covers fundamental classical machine learning concepts commonly asked in technical interviews. Topics include supervised learning algorithms, unsupervised learning, model evaluation, regularization, ensemble methods, and more.

---

## Q1: Mention three ways to make your model robust to outliers

**Answer:**

1. **Use robust loss functions:** Instead of Mean Squared Error (MSE) which squares the errors and is highly sensitive to outliers, use Mean Absolute Error (MAE) or *Huber loss*. Huber loss combines the best of both worlds -- it behaves like MSE for small errors and like MAE for large errors, making it less sensitive to outliers.

2. **Outlier detection and removal:** Use statistical methods like Z-score (remove data points beyond 3 standard deviations), IQR method (remove points below Q1-1.5xIQR or above Q3+1.5xIQR), or isolation forests to detect and remove outliers before training. However, be cautious as outliers might contain valuable information.

3. **Use robust algorithms:** Choose algorithms that are inherently robust to outliers such as tree-based methods (Random Forests, Gradient Boosting), which make decisions based on splits rather than absolute values. Alternatively, use data transformations like log transformation to reduce the impact of extreme values.

> **Interview Tip:** Interviewers often follow up with "but what if the outliers are meaningful?" The key is to investigate whether outliers are data errors vs. genuine rare events before deciding to remove them. Always ask: "Is this outlier a data quality issue or a real signal?"

---

## Q2: What are L1 and L2 regularization? What are the differences between the two?

**Answer:**

Regularization techniques add a penalty term to the loss function to prevent overfitting by constraining model complexity.

**L1 Regularization (Lasso):**

- Adds the sum of absolute values of coefficients: `lambda * sum(|w_i|)`
- Produces sparse models by driving some coefficients exactly to zero
- Performs implicit feature selection

**L2 Regularization (Ridge):**

- Adds the sum of squared coefficients: `lambda * sum(w_i^2)`
- Shrinks coefficients toward zero but rarely makes them exactly zero
- Distributes penalty among all features

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|-----------|-----------|
| **Penalty Term** | lambda * sum(\|w_i\|) | lambda * sum(w_i^2) |
| **Feature Selection** | Yes (sets coefficients to 0) | No (shrinks but doesn't eliminate) |
| **Solution** | Sparse (many zeros) | Dense (small non-zero values) |
| **Computational Cost** | Higher (no closed form) | Lower (closed form solution) |
| **Best Use Case** | High-dimensional, sparse features | Multicollinearity, all features relevant |

> **Geometric Intuition (Most Important):** L1 regularization has a diamond-shaped constraint region in coefficient space. The corners of the diamond lie on the axes, so optimal solutions tend to land exactly on the axes, driving coefficients to exactly zero. This creates sparsity. In contrast, L2 regularization has a circular constraint region, so solutions shrink toward but rarely touch the axes, meaning coefficients approach zero gradually but almost never reach it exactly.

---

## Q3: Mention three ways to handle missing or corrupted data in a dataset

**Answer:**

1. **Deletion Methods:** Remove rows with missing values (*listwise deletion*) if the missing data is minimal (<5%) and Missing Completely At Random (MCAR). Alternatively, remove columns/features with high percentage of missing values (>50%). This is simple but can lead to information loss.

2. **Imputation:** Fill missing values with statistical measures: mean/median for numerical features (median is more robust to outliers), mode for categorical features, or use more sophisticated methods like *K-Nearest Neighbors* (KNN) imputation which fills missing values based on similar instances, or forward/backward fill for time series data.

3. **Advanced Methods:** Use algorithms that handle missing values naturally (like XGBoost, LightGBM), create a separate category/indicator variable for missingness (which can capture patterns in why data is missing), or use predictive models like *MICE* (Multiple Imputation by Chained Equations) to predict missing values based on other features.

---

## Q4: Explain briefly the logistic regression model and state an example of when you have used it recently

**Answer:**

**Logistic regression** is a supervised learning algorithm used for binary classification problems. Despite its name, it's a classification algorithm, not regression.

**How it works:**

- Uses the sigmoid/logistic function: `sigma(z) = 1 / (1 + e^(-z))` to map linear combination of features to probability [0,1]
- Model: `P(y=1|x) = sigma(w_0 + w_1*x_1 + w_2*x_2 + ... + w_n*x_n)`
- Uses log loss (cross-entropy) as the cost function
- Optimized using gradient descent or other optimization algorithms

> **Decision Boundary Interpretation:** The coefficients define a linear decision boundary in feature space. Each coefficient w_i tells you how much a one-unit increase in feature x_i changes the log-odds of the positive class. This makes logistic regression highly interpretable compared to black-box methods.

**Key advantages:** Interpretable coefficients, outputs calibrated probabilities, computationally efficient, works well with linearly separable data.

**Example use case:** Email spam classification (spam vs not spam), customer churn prediction (will churn vs won't churn), credit default prediction, disease diagnosis (positive vs negative), or click-through rate prediction in digital advertising.

---

## Q5: Explain the linear regression model and discuss its assumptions

**Answer:**

**Linear regression** is a supervised learning algorithm that models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation.

**Model equation:** `y = beta_0 + beta_1*x_1 + beta_2*x_2 + ... + beta_n*x_n + epsilon`

Where beta coefficients are learned by minimizing Mean Squared Error (MSE) using ordinary least squares or gradient descent.

**Key Assumptions (LINE + No Multicollinearity):**

1. **Linearity:** The relationship between independent and dependent variables is linear. Check using residual plots.

2. **Independence:** Observations are independent of each other (no autocorrelation). Particularly important for time series data.

3. **Normality:** Residuals (errors) are normally distributed. Check using Q-Q plots or Shapiro-Wilk test.

4. **Equal Variance (Homoscedasticity):** Residuals have constant variance across all levels of independent variables. Check using residual vs fitted plots.

5. **No Multicollinearity:** Independent variables are not highly correlated with each other. Check using VIF (Variance Inflation Factor).

**What Happens When Assumptions Are Violated:**

| Assumption Violated | Consequence | Remedy |
|---------------------|-------------|--------|
| **Linearity** | Model underfits; predictions are systematically biased | Polynomial features or non-linear models |
| **Independence** | Standard errors underestimated; confidence intervals too narrow | Robust standard errors or GLS |
| **Normality** | Hypothesis tests and confidence intervals become unreliable | Robust methods or transformation |
| **Homoscedasticity** | Coefficients unbiased but inefficient | Weighted least squares or robust standard errors |
| **Multicollinearity** | Coefficients unstable and hard to interpret | Regularization (Ridge/Lasso) or remove correlated features |

---

## Q6: What are the differences between a model that minimizes squared error and one that minimizes absolute error? In which cases is each error metric more appropriate?

**Answer:**

| Aspect | Squared Error (MSE/L2) | Absolute Error (MAE/L1) |
|--------|------------------------|-------------------------|
| **Formula** | (1/n) * sum((y_i - y_hat_i)^2) | (1/n) * sum(\|y_i - y_hat_i\|) |
| **Outlier Sensitivity** | High (squares magnify large errors) | Low (treats all errors equally) |
| **Differentiability** | Smooth, differentiable everywhere | Not differentiable at zero |
| **Optimization** | Easier (gradient descent) | Harder (non-smooth) |
| **Best for** | Clean data, when large errors are very bad | Noisy data with outliers |

**When to use each:**

- **MSE:** When outliers are rare and should be heavily penalized, when you need smooth gradients for optimization, for standard regression with Gaussian noise.
- **MAE:** When data contains many outliers, when all errors should be treated equally, for robust regression where extreme predictions are acceptable.

---

## Q7: Define and compare parametric and non-parametric models and give two examples for each

**Answer:**

**Parametric Models:**

- Make strong assumptions about the functional form of the data
- Have a fixed number of parameters regardless of training data size
- Faster to train and predict, require less data
- Risk of underfitting if assumptions are wrong

**Examples:** Linear Regression (assumes linear relationship), Logistic Regression, Naive Bayes, Perceptron.

**Non-Parametric Models:**

- Make few or no assumptions about data distribution
- Number of parameters grows with training data size
- More flexible, can model complex relationships
- Require more data, risk of overfitting, computationally expensive

**Examples:** K-Nearest Neighbors (KNN), Decision Trees, Random Forests, Kernel SVM.

---

## Q8: You are working on a clustering problem. What are different evaluation metrics that can be used, and how to choose between them?

**Answer:**

Clustering evaluation metrics fall into two categories: *internal metrics* (no ground truth) and *external metrics* (with ground truth).

### Internal Metrics (No Labels)

1. **Silhouette Score (-1 to 1):** Measures how similar an object is to its own cluster compared to other clusters. Higher is better. Good for convex clusters.

2. **Davies-Bouldin Index:** Ratio of within-cluster to between-cluster distances. Lower is better. Fast to compute.

3. **Calinski-Harabasz Index:** Ratio of between-cluster to within-cluster dispersion. Higher is better.

### External Metrics (With Ground Truth)

4. **Adjusted Rand Index (ARI):** Measures similarity between two clusterings, adjusted for chance. Range: -1 to 1, higher is better.

5. **Normalized Mutual Information (NMI):** Information theoretic measure. Range: 0 to 1, higher is better.

6. **Fowlkes-Mallows Index:** Geometric mean of precision and recall for pairs of points.

> **How to choose:** Use internal metrics when you don't have labels (most common). Silhouette is most interpretable. Use external metrics only for validation if you have ground truth labels. Consider combining multiple metrics for robust evaluation.

---

## Q9: What is the ROC curve and when should you use it?

**Answer:**

The **ROC** (Receiver Operating Characteristic) curve is a graphical plot that illustrates the diagnostic ability of a binary classifier as its discrimination threshold varies.

**Components:**

- **X-axis:** False Positive Rate (FPR) = FP / (FP + TN)
- **Y-axis:** True Positive Rate (TPR) = TP / (TP + FN), also called Recall or Sensitivity
- **AUC-ROC:** Area Under the Curve, ranges from 0 to 1 (0.5 = random, 1.0 = perfect)

> **Key Intuition:** A random classifier gives a diagonal line (AUC = 0.5). The further the curve bows toward the top-left corner, the better the model. The top-left corner represents the ideal classifier: 100% true positives and 0% false positives.

**When to use ROC:**

- When you need to evaluate model performance across different thresholds
- When classes are relatively balanced
- When both false positives and false negatives are important
- For comparing multiple models objectively

**When NOT to use ROC:** For highly imbalanced datasets (use *Precision-Recall curve* instead), when you care more about positive predictions than the full spectrum of thresholds. Precision-Recall curves are more informative for imbalanced data because they focus on the minority class performance.

---

## Q10: What is the difference between hard and soft voting classifiers in the context of ensemble learners?

**Answer:**

Voting classifiers combine predictions from multiple models. The difference lies in how they aggregate predictions.

**Hard Voting:**

- Each classifier votes for a class (majority vote wins)
- Prediction = mode of all classifier predictions
- Example: 3 classifiers predict [Class A, Class A, Class B] -> Final: Class A
- Works with any classifier, doesn't require probability estimates

**Soft Voting:**

- Averages predicted probabilities from all classifiers
- Prediction = argmax of averaged probabilities
- Example: Average probabilities [0.9, 0.7, 0.4] for Class A -> Final: Class A
- Requires classifiers to output probability estimates (`predict_proba`)

> **Key Insight:** Soft voting generally performs better because it gives more weight to highly confident predictions and can handle uncertainty better than hard voting.

---

## Q11: In what cases would you use vanilla PCA, Incremental PCA, Randomized PCA, or Kernel PCA?

**Answer:**

| PCA Type | When to Use | Advantages | Limitations |
|----------|-------------|-----------|-------------|
| **Vanilla PCA** | Dataset fits in memory, standard use case | Most accurate, complete eigenvalue decomposition | High memory requirement, slow for large datasets |
| **Incremental PCA** | Large datasets that don't fit in memory, online learning | Low memory, can process mini-batches | Slower than vanilla PCA if data fits in memory |
| **Randomized PCA** | High-dimensional data, need speed, few components | Very fast, good approximation | Approximate (stochastic), not deterministic |
| **Kernel PCA** | Non-linear relationships, complex manifolds | Handles non-linear structures well | Computationally expensive, kernel selection needed |

---

## Q12: Discuss two clustering algorithms that can scale to large datasets

**Answer:**

### 1. Mini-Batch K-Means

- Variant of K-Means that uses mini-batches instead of the entire dataset
- Updates centroids using small random batches of data
- Time complexity: O(n x k x i x b) where b is batch size << n
- **Advantages:** 3-10x faster than standard K-Means, low memory footprint, suitable for online learning
- **Trade-off:** Slightly less accurate than full K-Means but often negligible in practice

### 2. BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)

- Builds a tree structure (CF-Tree) that summarizes the data
- Processes data incrementally in a single pass
- Time complexity: O(n) -- linear in number of samples
- **Advantages:** Very fast, memory efficient (only stores cluster features), handles outliers well, works with streaming data
- **Limitation:** Works best with spherical clusters, sensitive to data insertion order

---

## Q13: Do you need to scale your data if you will be using the SVM classifier? Discuss your answer

**Answer:**

**Yes, you absolutely should scale your data before using SVM.** Here's why:

**Why scaling is critical for SVM:**

1. **Distance-based algorithm:** SVM finds the optimal hyperplane by maximizing the margin between classes. Features with larger scales will dominate the distance calculations.

2. **Example:** If one feature ranges from 0-1000 (income) and another from 0-1 (age/100), the income feature will dominate, making age almost irrelevant.

3. **Kernel functions:** RBF and polynomial kernels compute distances/similarities between samples. Unscaled features lead to numerical instability and poor kernel performance.

4. **Training time:** Unscaled data can significantly slow down convergence during optimization.

**Recommended scaling approaches:**

| Scaler | Description | Best For |
|--------|-------------|----------|
| **StandardScaler** | mean=0, variance=1 | Most common choice for SVM |
| **MinMaxScaler** | Scale to [0,1] or [-1,1] | When you want bounded features |
| **RobustScaler** | Uses median and IQR | Best if you have outliers |

---

## Q14: What are Loss Functions and Cost Functions? Explain the key difference between them

**Answer:**

**Loss Function:**

- Measures the error for a *single* training example
- Computed on individual predictions
- Example: `L(y_hat_i, y_i) = (y_hat_i - y_i)^2` for one sample

**Cost Function (also called Objective Function):**

- Measures the average error over the *entire* training set
- Aggregation of all individual losses
- Example: `J(w) = (1/n) * sum(L(y_hat_i, y_i)) + regularization`

> **Key Difference:** Loss function is per-sample, cost function is per-dataset. During training, we minimize the cost function (not individual losses). The cost function may also include regularization terms beyond just the average loss.
>
> **Analogy:** Loss is like grading one student's exam, cost is like calculating the class average (plus any curve/adjustment).

---

## Q15: What is the importance of batch in machine learning and explain batch-dependent gradient descent algorithms?

**Answer:**

**Importance of Batches:**

- Enable training on datasets larger than memory
- Control trade-off between computation speed and gradient accuracy
- Affect convergence speed and model generalization
- Enable parallel processing on GPUs/TPUs

| Aspect | Batch Gradient Descent | Stochastic GD (SGD) | Mini-Batch GD |
|--------|------------------------|---------------------|---------------|
| **Batch Size** | Entire dataset | 1 sample | Small subset (32, 64, 128, 256) |
| **Updates per Epoch** | 1 | n (dataset size) | n / batch_size |
| **Pros** | Stable convergence, accurate gradient | Fast updates, escape local minima | Balance of both |
| **Cons** | Slow, high memory, can't use online | Noisy updates, erratic convergence | GPU optimization, better generalization |

---

## Q16: What are the different methods to split a tree in a decision tree algorithm?

**Answer:**

Decision trees use various splitting criteria to determine the best feature and threshold for partitioning data:

### 1. Information Gain (ID3, C4.5)

- Based on entropy reduction
- Entropy: `H(S) = -sum(p_i * log2(p_i))`
- Information Gain = H(parent) - sum((|S_v|/|S|) x H(S_v))
- Higher information gain = better split

### 2. Gini Impurity (CART)

- Measures probability of incorrect classification
- `Gini = 1 - sum(p_i^2)`
- Gini = 0 (pure node), Gini = 0.5 (maximum impurity for binary)
- Lower Gini impurity = better split

### 3. Chi-Square

- Statistical test measuring independence between feature and target
- `chi^2 = sum((Observed - Expected)^2 / Expected)`
- Higher chi^2 = stronger relationship = better split

### 4. Variance Reduction (for Regression)

- Used for continuous target variables
- Variance Reduction = Var(parent) - sum((|S_v|/|S|) x Var(S_v))
- Higher reduction = better split

---

## Q17: Why is boosting a more stable algorithm compared to other ensemble algorithms?

**Answer:**

**Boosting provides consistent, incremental improvement** through its sequential approach:

**Why boosting can be considered stable:**

1. **Bias reduction:** Boosting sequentially corrects errors, systematically reducing bias. This leads to strong predictive performance.

2. **Weighted learning:** Focuses on hard-to-classify samples, ensuring consistent improvement across iterations.

3. **Regularization:** Modern boosting (XGBoost, LightGBM) includes built-in regularization, making predictions more stable.

**However, boosting is NOT always more stable:**

- **Overfitting risk:** Boosting can overfit if not properly tuned (too many iterations, no early stopping)
- **Noise sensitivity:** Outliers and noisy labels can destabilize boosting since it focuses on hard samples
- **Variance:** Random Forests typically have lower variance due to bagging and randomness

> **Better answer:** Boosting provides stable improvement in bias reduction and consistent performance gains when properly regularized, but Random Forests may be more stable in terms of variance and robustness to noisy data.

---

## Q18: What are the different approaches to implementing recommendation systems?

**Answer:**

### 1. Content-Based Filtering

- Recommends items similar to those the user liked in the past
- Uses item features (genre, keywords, attributes)
- Techniques: TF-IDF, cosine similarity, feature vectors
- **Pros:** No cold start for new users, transparent recommendations
- **Cons:** Limited discovery, requires rich item metadata, filter bubble

### 2. Collaborative Filtering

- **User-based:** Find similar users, recommend what they liked
- **Item-based:** Find similar items based on user interactions
- Techniques: Matrix factorization (SVD, NMF), nearest neighbors
- **Pros:** No item features needed, discovers unexpected patterns
- **Cons:** Cold start problem, sparsity, scalability challenges

### 3. Hybrid Systems

- Combine multiple approaches (content + collaborative)
- Examples: Weighted hybrid, switching hybrid, cascade hybrid
- Used by Netflix, Amazon, Spotify

### 4. Deep Learning Approaches

- Neural Collaborative Filtering (NCF)
- Autoencoders for collaborative filtering
- Sequence models (RNNs) for session-based recommendations
- Graph Neural Networks for social recommendations

### 5. Context-Aware & Reinforcement Learning

- Incorporate time, location, device
- Use RL to optimize long-term user engagement

---

## Q19: What are the evaluation metrics that can be used for multi-label classification?

**Answer:**

**Multi-label classification** is when each sample can belong to multiple classes simultaneously (e.g., a movie can be both 'Action' and 'Comedy'). Standard metrics need to be adapted:

### 1. Hamming Loss

- Fraction of wrong labels (either missing or extra)
- Lower is better (0 = perfect)
- Formula: `(1 / (n x L)) * sum(XOR(y_ij, y_hat_ij))`

### 2. Subset Accuracy (Exact Match Ratio)

- Percentage where predicted labels exactly match true labels
- Very strict -- all labels must be correct

### 3. Label-based Metrics (Macro/Micro/Weighted)

- **Micro-averaged:** Aggregate all predictions globally, then compute metric (good for imbalanced datasets)
- **Macro-averaged:** Compute metric per label, then average (treats all labels equally)
- **Weighted:** Like macro but weighted by label frequency

### 4. Ranking-based Metrics

- **Coverage Error:** How far you need to go in ranked predictions to cover all true labels
- **Label Ranking Average Precision:** Average precision based on label ranking
- **NDCG:** Normalized Discounted Cumulative Gain

### 5. F1 Scores

- Can compute micro-F1, macro-F1, or per-sample F1 (comparing predicted vs true label sets for each sample)

---

## Q20: What is the difference between concept drift and data drift, and how to overcome each of them?

**Answer:**

Both refer to changes in data distribution over time, but they differ in *what* changes:

**Data Drift (Covariate Shift):**

- Definition: Distribution of input features P(X) changes, but P(Y|X) stays the same
- Example: Training on summer data but deploying in winter, user demographics change
- Impact: Model may still work but with degraded performance

**Concept Drift:**

- Definition: Relationship between features and target P(Y|X) changes
- Example: Customer behavior changes, economic conditions shift, competitors emerge
- Impact: Model predictions become fundamentally incorrect

| Solution | Data Drift | Concept Drift |
|----------|-----------|---------------|
| **Detection** | Statistical tests (KS-test, PSI), feature distribution monitoring | Monitor prediction accuracy, precision/recall over time |
| **Mitigation** | Feature transformation, reweighting, importance sampling | Retrain model with recent data, online learning |
| **General Strategies** | Regular monitoring, sliding window approach, ensemble methods | Active learning, continuous training pipeline, A/B testing |

---

## Q21: Can you explain the ARIMA model, its components, and assumptions?

**Answer:**

**ARIMA** (AutoRegressive Integrated Moving Average) is a popular time series forecasting model that combines three components:

### Components -- ARIMA(p, d, q)

1. **AR (AutoRegressive) -- p:** The model uses p past values (lags) to predict the current value. Captures linear dependencies on its own past values.

2. **I (Integrated) -- d:** The number of differencing operations needed to make the series stationary. Removes trend and seasonality.

3. **MA (Moving Average) -- q:** Uses q past forecast errors to improve predictions. Models dependency on past error terms.

**Mathematical form:** `(1 - sum(phi_i * L^i)) * (1-L)^d * y_t = (1 + sum(theta_i * L^i)) * epsilon_t`

### Key Assumptions

1. **Stationarity (after differencing):** Mean, variance, and autocorrelation structure are constant over time. Check using ADF test or KPSS test.

2. **Linear relationships:** ARIMA assumes linear dependencies between observations and errors.

3. **No autocorrelation in residuals:** Residuals should be white noise (independent, identically distributed). Check using Ljung-Box test.

4. **Homoscedasticity:** Constant variance of errors over time.

5. **Normality of residuals:** Errors are normally distributed (important for confidence intervals).

### Model Selection

- Use ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots
- AIC (Akaike Information Criterion) or BIC for model comparison
- Grid search over p, d, q parameters

**Extensions:** SARIMA adds seasonal components, ARIMAX includes exogenous variables, Vector ARIMA (VARIMA) for multivariate time series.

> **Practical Note:** In practice, use `auto_arima` (pmdarima library) or `statsmodels` SARIMAX for automatic parameter selection rather than manual grid search. This saves time and often finds better parameters.

**Use cases:** Stock prices, sales forecasting, demand prediction, weather forecasting (short-term), economic indicators.

---

## Study Tips

As you prepare for interviews, practice explaining each concept aloud in 2 minutes (simulating interview conditions). This helps you internalize the material and develop the communication skills that are just as important as technical knowledge. Focus on understanding the "why" behind each algorithm, not just the mechanics. Interviewers value candidates who can explain trade-offs and connect concepts together.
