# Module 2: Mathematics and Statistics Fundamentals

## Overview
This module provides a comprehensive foundation in the mathematical and statistical concepts essential for data science. Understanding these fundamentals is crucial for developing machine learning algorithms, interpreting results, and making data-driven decisions.

## Learning Objectives
By the end of this module, you will be able to:
- Understand linear algebra concepts used in data science
- Apply calculus principles to optimization problems
- Work with probability distributions and statistical inference
- Perform hypothesis testing and confidence intervals
- Understand regression analysis and correlation
- Apply mathematical concepts to real-world data problems

## 1. Linear Algebra

### 1.1 Vectors and Matrices

#### Vectors
A vector is an ordered collection of numbers that can represent data points, features, or coefficients.

**Types of Vectors:**
- **Row Vector**: `[1, 2, 3]` - horizontal arrangement
- **Column Vector**: `[1, 2, 3]ᵀ` - vertical arrangement
- **Unit Vector**: A vector with magnitude 1, often denoted as û
- **Zero Vector**: A vector with all elements equal to zero

**Vector Operations:**
- **Addition**: `v + w = [v₁ + w₁, v₂ + w₂, ..., vₙ + wₙ]`
- **Scalar Multiplication**: `c × v = [c × v₁, c × v₂, ..., c × vₙ]`
- **Dot Product**: `v · w = Σ(vᵢ × wᵢ)`
- **Cross Product**: Only for 3D vectors, results in a vector perpendicular to both

#### Matrices
A matrix is a rectangular array of numbers arranged in rows and columns.

**Types of Matrices:**
- **Square Matrix**: Equal number of rows and columns (n × n)
- **Identity Matrix (I)**: Square matrix with 1s on diagonal, 0s elsewhere
- **Diagonal Matrix**: Square matrix with non-zero elements only on diagonal
- **Symmetric Matrix**: Matrix equal to its transpose (A = Aᵀ)
- **Orthogonal Matrix**: Matrix whose inverse equals its transpose (A⁻¹ = Aᵀ)

**Matrix Operations:**
- **Addition/Subtraction**: Element-wise operations
- **Scalar Multiplication**: Multiply each element by a scalar
- **Matrix Multiplication**: `Cᵢⱼ = Σₖ(Aᵢₖ × Bₖⱼ)`
- **Transpose**: Flip matrix over its diagonal (Aᵢⱼ → Aⱼᵢ)
- **Inverse**: Matrix A⁻¹ such that A × A⁻¹ = I (only for square matrices)

### 1.2 Eigenvalues and Eigenvectors

For a square matrix A, a non-zero vector v is an eigenvector if:
`A × v = λ × v`

Where λ (lambda) is the eigenvalue corresponding to eigenvector v.

**Applications in Data Science:**
- Principal Component Analysis (PCA)
- Dimensionality reduction
- Google's PageRank algorithm
- Quantum mechanics simulations

### 1.3 Matrix Decomposition

#### Singular Value Decomposition (SVD)
Any m × n matrix A can be decomposed as:
`A = U × Σ × Vᵀ`

Where:
- U: m × m orthogonal matrix (left singular vectors)
- Σ: m × n diagonal matrix (singular values)
- Vᵀ: n × n orthogonal matrix (right singular vectors)

**Applications:**
- Image compression
- Recommendation systems
- Natural language processing

## 2. Calculus

### 2.1 Differential Calculus

#### Derivatives
The derivative measures the rate of change of a function.

**Basic Rules:**
- **Power Rule**: d/dx(xⁿ) = n × xⁿ⁻¹
- **Product Rule**: d/dx(f × g) = f' × g + f × g'
- **Chain Rule**: d/dx(f(g(x))) = f'(g(x)) × g'(x)
- **Quotient Rule**: d/dx(f/g) = (f' × g - f × g') / g²

#### Partial Derivatives
For multivariable functions, partial derivatives measure change in one variable while holding others constant.

`∂f/∂x` - partial derivative with respect to x

#### Gradients
The gradient is a vector of all partial derivatives:
`∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]`

**Applications in Data Science:**
- Gradient descent optimization
- Backpropagation in neural networks
- Sensitivity analysis

### 2.2 Integral Calculus

#### Definite Integrals
The definite integral calculates the area under a curve between two points:

`∫ₐᵇ f(x) dx`

#### Fundamental Theorem of Calculus
`d/dx ∫ₐˣ f(t) dt = f(x)`

**Applications:**
- Probability calculations
- Expected value computations
- Area under ROC curves

### 2.3 Optimization

#### Local vs Global Optima
- **Local Optimum**: Best value in a neighborhood
- **Global Optimum**: Best value in entire domain

#### Convex vs Non-Convex Functions
- **Convex Function**: Line segment between any two points lies above the function
- **Non-Convex Function**: May have multiple local optima

#### Gradient Descent
Iterative optimization algorithm:
1. Start with initial parameters θ
2. Compute gradient ∇J(θ)
3. Update: θ = θ - α × ∇J(θ)
4. Repeat until convergence

**Types:**
- **Batch Gradient Descent**: Uses entire dataset
- **Stochastic Gradient Descent (SGD)**: Uses one sample per iteration
- **Mini-batch Gradient Descent**: Uses small batches

## 3. Probability Theory

### 3.1 Basic Concepts

#### Sample Space and Events
- **Sample Space (S)**: All possible outcomes
- **Event (E)**: Subset of sample space
- **Probability**: P(E) = |E| / |S| for equally likely outcomes

#### Probability Axioms
1. 0 ≤ P(E) ≤ 1 for any event E
2. P(S) = 1
3. For mutually exclusive events: P(E₁ ∪ E₂) = P(E₁) + P(E₂)

#### Conditional Probability
Probability of event A given that event B has occurred:
`P(A|B) = P(A ∩ B) / P(B)`

#### Bayes' Theorem
`P(A|B) = [P(B|A) × P(A)] / P(B)`

**Applications:**
- Spam filtering
- Medical diagnosis
- Document classification

### 3.2 Random Variables

#### Discrete Random Variables
Take countable values (e.g., number of heads in coin flips)

#### Continuous Random Variables
Take uncountable values (e.g., height, weight, temperature)

#### Probability Distributions

**Discrete Distributions:**
- **Bernoulli**: Single trial (success/failure)
- **Binomial**: Multiple independent Bernoulli trials
- **Poisson**: Events in fixed interval
- **Geometric**: Number of trials until first success

**Continuous Distributions:**
- **Normal (Gaussian)**: Bell-shaped curve, defined by μ and σ²
- **Uniform**: Equal probability over interval
- **Exponential**: Time between events
- **Beta**: Probabilities and proportions

### 3.3 Expected Value and Variance

#### Expected Value (Mean)
For discrete RV: `E[X] = Σ(xᵢ × P(X = xᵢ))`
For continuous RV: `E[X] = ∫x × f(x) dx`

#### Variance
Measure of spread: `Var(X) = E[(X - μ)²]`
Standard Deviation: `σ = √Var(X)`

#### Covariance and Correlation
- **Covariance**: `Cov(X,Y) = E[(X - μ_X)(Y - μ_Y)]`
- **Correlation**: `ρ = Cov(X,Y) / (σ_X × σ_Y)`

## 4. Statistical Inference

### 4.1 Sampling and Sampling Distributions

#### Population vs Sample
- **Population**: Entire group of interest
- **Sample**: Subset of population
- **Parameter**: Numerical characteristic of population
- **Statistic**: Numerical characteristic of sample

#### Central Limit Theorem
For large samples (n ≥ 30), the sampling distribution of the mean is approximately normal, regardless of the population distribution.

#### Standard Error
`SE = σ / √n` (for known population standard deviation)
`SE = s / √n` (for estimated standard deviation)

### 4.2 Hypothesis Testing

#### Steps in Hypothesis Testing
1. **State Hypotheses**:
   - H₀ (null): No effect or no difference
   - H₁ (alternative): Effect or difference exists

2. **Choose Significance Level (α)**: Typically 0.05, 0.01, or 0.10

3. **Select Test Statistic**: z-test, t-test, F-test, χ²-test

4. **Determine Critical Region**: Reject H₀ if test statistic falls in this region

5. **Make Decision**: Reject or fail to reject H₀

6. **Interpret Results**: Consider practical significance

#### Common Tests

**One-Sample t-test**: Compare sample mean to known value
**Two-Sample t-test**: Compare means of two groups
**Paired t-test**: Compare means of related samples
**ANOVA**: Compare means of three or more groups
**Chi-Square Test**: Test independence of categorical variables

#### Type I and Type II Errors
- **Type I Error (α)**: Reject H₀ when H₀ is true (false positive)
- **Type II Error (β)**: Fail to reject H₀ when H₀ is false (false negative)
- **Power**: 1 - β (probability of correctly rejecting false H₀)

### 4.3 Confidence Intervals

#### Interpretation
A 95% confidence interval means: If we repeated the sampling process many times, 95% of the resulting confidence intervals would contain the true population parameter.

#### Formula for Mean (Large Sample)
`CI = x̄ ± z × (σ/√n)`

#### Formula for Mean (Small Sample)
`CI = x̄ ± t × (s/√n)`

Where t comes from t-distribution with n-1 degrees of freedom.

## 5. Regression Analysis

### 5.1 Simple Linear Regression

#### Model
`Y = β₀ + β₁X + ε`

Where:
- Y: Dependent variable
- X: Independent variable
- β₀: Intercept
- β₁: Slope
- ε: Error term

#### Parameter Estimation (Least Squares)
- `β₁ = Σ((xᵢ - x̄)(yᵢ - ȳ)) / Σ((xᵢ - x̄)²)`
- `β₀ = ȳ - β₁x̄`

#### Model Evaluation
- **R²**: Proportion of variance explained (0 ≤ R² ≤ 1)
- **MSE**: Mean squared error
- **MAE**: Mean absolute error

### 5.2 Multiple Linear Regression

#### Model
`Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε`

#### Assumptions
1. **Linearity**: Relationship between X and Y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of errors
4. **Normality**: Errors are normally distributed
5. **No Multicollinearity**: Predictors are not highly correlated

#### Feature Selection
- **Forward Selection**: Start with no variables, add one at a time
- **Backward Elimination**: Start with all variables, remove one at a time
- **Stepwise Selection**: Combination of forward and backward

### 5.3 Regularization

#### Ridge Regression (L2)
Adds penalty term: `λΣβⱼ²`
Prevents overfitting by shrinking coefficients toward zero.

#### Lasso Regression (L1)
Adds penalty term: `λΣ|βⱼ|`
Can force some coefficients to exactly zero (feature selection).

#### Elastic Net
Combination of L1 and L2 regularization.

## 6. Practical Applications

### 6.1 Principal Component Analysis (PCA)
- Dimensionality reduction using eigenvectors
- Find principal components that explain maximum variance
- Applications: Image compression, feature extraction

### 6.2 Gradient Descent in Machine Learning
- Optimize loss functions in neural networks
- Batch, stochastic, and mini-batch variants
- Learning rate scheduling and adaptive methods

### 6.3 A/B Testing
- Hypothesis testing for product changes
- Statistical significance vs practical significance
- Sample size calculations

### 6.4 Time Series Analysis
- Autocorrelation and stationarity
- ARIMA models
- Forecasting techniques

## 7. Common Pitfalls and Best Practices

### 7.1 Mathematical Errors
- Matrix multiplication order
- Derivative calculations
- Integration limits

### 7.2 Statistical Fallacies
- Correlation vs causation
- Simpson's paradox
- p-hacking (multiple testing without correction)

### 7.3 Computational Issues
- Numerical stability
- Floating-point precision
- Matrix conditioning

### 7.4 Best Practices
- Always check assumptions
- Use appropriate sample sizes
- Validate results on holdout data
- Document all steps and decisions

## 8. Tools and Libraries

### Python Libraries
- **NumPy**: Numerical computing and linear algebra
- **SciPy**: Scientific computing and statistics
- **StatsModels**: Statistical modeling and testing
- **SymPy**: Symbolic mathematics

### R Packages
- **base**: Core statistical functions
- **MASS**: Modern applied statistics
- **car**: Companion to applied regression
- **lme4**: Linear mixed-effects models

### Online Resources
- Khan Academy (Calculus and Statistics)
- 3Blue1Brown (Linear Algebra visualizations)
- StatQuest with Josh Starmer
- Brilliant.org (Interactive mathematics)

## 9. Assessment

### Quiz Questions
1. What is the difference between a vector and a scalar?
2. Explain the concept of eigenvalues and eigenvectors.
3. What is the Central Limit Theorem and why is it important?
4. Describe the steps in hypothesis testing.
5. What are the assumptions of linear regression?
6. Explain the bias-variance tradeoff in the context of regularization.

### Practical Exercises
1. Implement matrix multiplication from scratch
2. Calculate derivatives of polynomial functions
3. Simulate random variables from different distributions
4. Perform hypothesis testing on sample data
5. Build and evaluate a regression model

## 10. Further Reading

### Books
- "Introduction to Linear Algebra" by Gilbert Strang
- "Calculus" by James Stewart
- "All of Statistics" by Larry Wasserman
- "Elements of Statistical Learning" by Hastie et al.

### Online Courses
- Coursera: Mathematics for Machine Learning
- edX: Data Analysis and Statistical Inference
- Khan Academy: Probability and Statistics

### Research Papers
- "The Elements of Statistical Learning" (free PDF)
- "An Introduction to Statistical Learning" (free PDF)
- Research papers on specific algorithms

## Next Steps

Congratulations on completing the mathematical foundations! These concepts form the backbone of data science and machine learning. In the next module, we'll apply these mathematical concepts using programming tools.

**Ready to continue?** Proceed to [Module 3: Programming Foundations](../03_programming_foundations/)
