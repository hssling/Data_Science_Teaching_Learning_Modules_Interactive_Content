"""
Module 2: Mathematics and Statistics Fundamentals - Code Examples
=================================================================

This file contains practical examples demonstrating mathematical and statistical
concepts essential for data science using Python libraries like NumPy, SciPy,
and StatsModels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== Module 2: Mathematics and Statistics Fundamentals Examples ===\n")

# =============================================================================
# 1. LINEAR ALGEBRA EXAMPLES
# =============================================================================

print("1. Linear Algebra Examples")
print("=" * 30)

# Example 1.1: Vector Operations
print("1.1 Vector Operations")

# Create vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print(f"Vector v1: {v1}")
print(f"Vector v2: {v2}")

# Vector addition
print(f"v1 + v2: {v1 + v2}")

# Scalar multiplication
print(f"3 * v1: {3 * v1}")

# Dot product
print(f"Dot product v1 · v2: {np.dot(v1, v2)}")

# Vector magnitude (norm)
print(f"Magnitude of v1: {np.linalg.norm(v1):.3f}")

# Unit vector
v1_unit = v1 / np.linalg.norm(v1)
print(f"Unit vector of v1: {v1_unit}")
print(f"Magnitude of unit vector: {np.linalg.norm(v1_unit):.3f}")

# Example 1.2: Matrix Operations
print("\n1.2 Matrix Operations")

# Create matrices
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]])

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)

# Matrix addition
print("\nA + B:")
print(A + B)

# Matrix multiplication
print("\nA × B:")
print(np.dot(A, B))

# Transpose
print("\nTranspose of A:")
print(A.T)

# Identity matrix
I = np.eye(3)
print("\n3x3 Identity Matrix:")
print(I)

# Matrix inverse (for square matrices)
A_square = np.array([[1, 2],
                     [3, 4]])
print(f"\nOriginal matrix:\n{A_square}")
print(f"Inverse:\n{np.linalg.inv(A_square)}")
print(f"Verification (A × A⁻¹):\n{np.dot(A_square, np.linalg.inv(A_square))}")

# Example 1.3: Eigenvalues and Eigenvectors
print("\n1.3 Eigenvalues and Eigenvectors")

# Create a square matrix
M = np.array([[4, 2],
              [1, 3]])

print(f"Matrix M:\n{M}")

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(M)

print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Verify: M × v = λ × v
for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    Mv = np.dot(M, eigenvec)
    lambda_v = eigenval * eigenvec
    print(f"\nEigenvalue {i+1}: {eigenval:.3f}")
    print(f"M × v{i+1}: {Mv}")
    print(f"λ × v{i+1}: {lambda_v}")
    print(f"Difference: {Mv - lambda_v}")

# Example 1.4: Singular Value Decomposition (SVD)
print("\n1.4 Singular Value Decomposition (SVD)")

# Create a matrix
X = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

print(f"Original matrix X:\n{X}")

# Perform SVD
U, s, Vt = np.linalg.svd(X)

print(f"\nU matrix (left singular vectors):\n{U}")
print(f"Singular values: {s}")
print(f"V^T matrix (right singular vectors):\n{Vt}")

# Reconstruct matrix
Sigma = np.zeros((X.shape[0], X.shape[1]))
Sigma[:min(X.shape), :min(X.shape)] = np.diag(s)
X_reconstructed = U @ Sigma @ Vt

print(f"\nReconstructed matrix:\n{X_reconstructed.astype(int)}")
print(f"Reconstruction error: {np.max(np.abs(X - X_reconstructed)):.2e}")

# =============================================================================
# 2. CALCULUS EXAMPLES
# =============================================================================

print("\n\n2. Calculus Examples")
print("=" * 25)

# Example 2.1: Numerical Differentiation
print("2.1 Numerical Differentiation")

def numerical_derivative(f, x, h=1e-5):
    """Calculate numerical derivative using central difference"""
    return (f(x + h) - f(x - h)) / (2 * h)

# Define functions
def f1(x):
    return x**2 + 3*x + 1  # f'(x) = 2x + 3

def f2(x):
    return np.sin(x)  # f'(x) = cos(x)

def f3(x):
    return np.exp(x)  # f'(x) = exp(x)

# Test points
x_test = np.pi/4

print("Function: f(x) = x² + 3x + 1")
print(f"Analytical derivative at x = {x_test:.3f}: {2*x_test + 3:.6f}")
print(f"Numerical derivative: {numerical_derivative(f1, x_test):.6f}")

print("\nFunction: f(x) = sin(x)")
print(f"Analytical derivative at x = {x_test:.3f}: {np.cos(x_test):.6f}")
print(f"Numerical derivative: {numerical_derivative(f2, x_test):.6f}")

print("\nFunction: f(x) = e^x")
print(f"Analytical derivative at x = {x_test:.3f}: {np.exp(x_test):.6f}")
print(f"Numerical derivative: {numerical_derivative(f3, x_test):.6f}")

# Example 2.2: Gradient Descent
print("\n2.2 Gradient Descent Optimization")

def gradient_descent(gradient_func, start_point, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
    """Simple gradient descent implementation"""
    x = start_point
    history = [x]

    for i in range(max_iter):
        grad = gradient_func(x)
        x_new = x - learning_rate * grad
        history.append(x_new)

        if abs(x_new - x) < tolerance:
            break
        x = x_new

    return x, history

# Function: f(x) = x² - 4x + 4 (minimum at x = 2)
def f_quad(x):
    return x**2 - 4*x + 4

def grad_quad(x):
    return 2*x - 4  # Derivative of x² - 4x + 4

# Run gradient descent
minimum, history = gradient_descent(grad_quad, start_point=0.0, learning_rate=0.1)

print(f"Function: f(x) = x² - 4x + 4")
print(f"Analytical minimum: x = 2, f(2) = 0")
print(f"Gradient descent result: x = {minimum:.6f}, f(x) = {f_quad(minimum):.6f}")
print(f"Number of iterations: {len(history)-1}")

# Visualize gradient descent
x_vals = np.linspace(-1, 5, 100)
y_vals = f_quad(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, 'b-', label='f(x) = x² - 4x + 4')
plt.plot(history, [f_quad(x) for x in history], 'ro-', alpha=0.7, label='Gradient descent path')
plt.plot(minimum, f_quad(minimum), 'g*', markersize=15, label='Final minimum')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent Optimization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('gradient_descent_demo.png', dpi=300, bbox_inches='tight')
plt.show()
print("Gradient descent visualization saved as 'gradient_descent_demo.png'")

# =============================================================================
# 3. PROBABILITY EXAMPLES
# =============================================================================

print("\n\n3. Probability Examples")
print("=" * 25)

# Example 3.1: Common Probability Distributions
print("3.1 Common Probability Distributions")

# Set random seed for reproducibility
np.random.seed(42)

# Normal distribution
mu, sigma = 0, 1
normal_data = np.random.normal(mu, sigma, 10000)

# Binomial distribution (n=10, p=0.3)
n, p = 10, 0.3
binomial_data = np.random.binomial(n, p, 10000)

# Poisson distribution (λ=3)
lambda_poisson = 3
poisson_data = np.random.poisson(lambda_poisson, 10000)

# Plot distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Normal distribution
axes[0].hist(normal_data, bins=50, alpha=0.7, density=True, color='blue')
x_normal = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
axes[0].plot(x_normal, stats.norm.pdf(x_normal, mu, sigma), 'r-', linewidth=2)
axes[0].set_title('Normal Distribution (μ=0, σ=1)')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Density')

# Binomial distribution
axes[1].hist(binomial_data, bins=np.arange(12)-0.5, alpha=0.7, density=True, color='green')
x_binom = np.arange(0, n+1)
axes[1].plot(x_binom, stats.binom.pmf(x_binom, n, p), 'r-', linewidth=2, marker='o')
axes[1].set_title('Binomial Distribution (n=10, p=0.3)')
axes[1].set_xlabel('Number of successes')
axes[1].set_ylabel('Probability')

# Poisson distribution
axes[2].hist(poisson_data, bins=np.arange(15)-0.5, alpha=0.7, density=True, color='orange')
x_poisson = np.arange(0, 15)
axes[2].plot(x_poisson, stats.poisson.pmf(x_poisson, lambda_poisson), 'r-', linewidth=2, marker='o')
axes[2].set_title('Poisson Distribution (λ=3)')
axes[2].set_xlabel('Number of events')
axes[2].set_ylabel('Probability')

plt.tight_layout()
plt.savefig('probability_distributions.png', dpi=300, bbox_inches='tight')
plt.show()
print("Probability distributions plot saved as 'probability_distributions.png'")

# Example 3.2: Expected Value and Variance
print("\n3.2 Expected Value and Variance")

# Discrete random variable example
outcomes = np.array([1, 2, 3, 4, 5, 6])  # Die faces
probabilities = np.array([1/6] * 6)  # Fair die

expected_value = np.sum(outcomes * probabilities)
variance = np.sum((outcomes - expected_value)**2 * probabilities)
std_dev = np.sqrt(variance)

print("Fair 6-sided die:")
print(f"Expected value (mean): {expected_value:.3f}")
print(f"Variance: {variance:.3f}")
print(f"Standard deviation: {std_dev:.3f}")

# Continuous random variable (normal distribution)
print("
Normal distribution (μ=5, σ=2):")
mu, sigma = 5, 2
print(f"Expected value: {mu}")
print(f"Variance: {sigma**2}")
print(f"Standard deviation: {sigma}")

# Example 3.3: Bayes' Theorem
print("\n3.3 Bayes' Theorem")

# Medical test example
# P(Disease) = 0.01 (1% of population has disease)
# P(Positive|Disease) = 0.99 (test sensitivity)
# P(Positive|No Disease) = 0.05 (false positive rate)

P_disease = 0.01
P_positive_given_disease = 0.99
P_positive_given_no_disease = 0.05
P_no_disease = 1 - P_disease

# P(Positive) = P(Positive|Disease) × P(Disease) + P(Positive|No Disease) × P(No Disease)
P_positive = (P_positive_given_disease * P_disease) + (P_positive_given_no_disease * P_no_disease)

# P(Disease|Positive) = P(Positive|Disease) × P(Disease) / P(Positive)
P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

print("Medical Test Example:")
print(f"P(Disease): {P_disease:.3f}")
print(f"P(Positive|Disease): {P_positive_given_disease:.3f}")
print(f"P(Positive|No Disease): {P_positive_given_no_disease:.3f}")
print(f"P(Positive): {P_positive:.3f}")
print(f"P(Disease|Positive): {P_disease_given_positive:.3f}")
print(f"Posterior probability: {P_disease_given_positive:.1%}")

# =============================================================================
# 4. STATISTICAL INFERENCE EXAMPLES
# =============================================================================

print("\n\n4. Statistical Inference Examples")
print("=" * 35)

# Example 4.1: Central Limit Theorem Demonstration
print("4.1 Central Limit Theorem Demonstration")

# Generate data from exponential distribution (not normal)
np.random.seed(42)
population_size = 100000
sample_sizes = [5, 30, 100]
num_samples = 1000

# Exponential distribution (skewed)
population = np.random.exponential(scale=2, size=population_size)
population_mean = np.mean(population)
population_std = np.std(population)

print(f"Population: Exponential distribution")
print(f"Population mean: {population_mean:.3f}")
print(f"Population std: {population_std:.3f}")

# Sample means for different sample sizes
sample_means = {}
for n in sample_sizes:
    means = [np.mean(np.random.choice(population, size=n)) for _ in range(num_samples)]
    sample_means[n] = means

# Plot sampling distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, n in enumerate(sample_sizes):
    axes[i].hist(sample_means[n], bins=30, alpha=0.7, density=True, color='skyblue')
    axes[i].axvline(population_mean, color='red', linestyle='--', linewidth=2, label=f'Population mean: {population_mean:.3f}')

    # Add normal curve approximation
    mean_of_means = np.mean(sample_means[n])
    std_of_means = np.std(sample_means[n])
    x_normal = np.linspace(mean_of_means - 3*std_of_means, mean_of_means + 3*std_of_means, 100)
    axes[i].plot(x_normal, stats.norm.pdf(x_normal, mean_of_means, std_of_means),
                'orange', linewidth=2, label='Normal approximation')

    axes[i].set_title(f'Sample Size n = {n}')
    axes[i].set_xlabel('Sample Mean')
    axes[i].set_ylabel('Density')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('central_limit_theorem.png', dpi=300, bbox_inches='tight')
plt.show()
print("Central Limit Theorem demonstration saved as 'central_limit_theorem.png'")

# Example 4.2: Hypothesis Testing
print("\n4.2 Hypothesis Testing - One Sample t-test")

# Generate sample data
np.random.seed(42)
sample = np.random.normal(loc=105, scale=15, size=50)  # Sample from population with mean 105
population_mean_claim = 100  # Claimed population mean

print(f"Sample size: {len(sample)}")
print(f"Sample mean: {np.mean(sample):.3f}")
print(f"Sample standard deviation: {np.std(sample, ddof=1):.3f}")
print(f"Claimed population mean: {population_mean_claim}")

# Perform one-sample t-test
t_statistic, p_value = stats.ttest_1samp(sample, population_mean_claim)

print("
One-sample t-test results:")
print(f"t-statistic: {t_statistic:.3f}")
print(f"p-value: {p_value:.3f}")

alpha = 0.05
if p_value < alpha:
    print(f"Reject H₀ at α = {alpha} (evidence suggests population mean ≠ {population_mean_claim})")
else:
    print(f"Fail to reject H₀ at α = {alpha} (no evidence that population mean ≠ {population_mean_claim})")

# Example 4.3: Confidence Intervals
print("\n4.3 Confidence Intervals")

# Calculate 95% confidence interval for the mean
confidence_level = 0.95
degrees_of_freedom = len(sample) - 1
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)
standard_error = sample_std / np.sqrt(len(sample))

# t-critical value
t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

margin_of_error = t_critical * standard_error
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

print(f"Sample mean: {sample_mean:.3f}")
print(f"Standard error: {standard_error:.3f}")
print(f"t-critical value (95%): {t_critical:.3f}")
print(f"Margin of error: {margin_of_error:.3f}")
print(f"95% Confidence Interval: ({confidence_interval[0]:.3f}, {confidence_interval[1]:.3f})")

# =============================================================================
# 5. REGRESSION ANALYSIS EXAMPLES
# =============================================================================

print("\n\n5. Regression Analysis Examples")
print("=" * 32)

# Example 5.1: Simple Linear Regression
print("5.1 Simple Linear Regression")

# Generate sample data
np.random.seed(42)
n_points = 100
X = np.random.uniform(0, 10, n_points)
# True relationship: y = 2 + 3*x + noise
y = 2 + 3*X + np.random.normal(0, 2, n_points)

# Add intercept term
X_with_intercept = sm.add_constant(X)

# Fit OLS model
ols_model = sm.OLS(y, X_with_intercept).fit()

print("Simple Linear Regression Results:")
print(ols_model.summary())

# Get coefficients
intercept, slope = ols_model.params
print("
Manual calculation verification:")
print(f"Intercept: {intercept:.3f} (true: 2.000)")
print(f"Slope: {slope:.3f} (true: 3.000)")
print(f"R²: {ols_model.rsquared:.3f}")

# Example 5.2: Multiple Linear Regression
print("\n5.2 Multiple Linear Regression")

# Generate multiple features
np.random.seed(42)
n_samples = 200
X1 = np.random.normal(10, 3, n_samples)
X2 = np.random.normal(5, 2, n_samples)
X3 = np.random.normal(8, 1.5, n_samples)

# True relationship: y = 1 + 2*X1 - 1.5*X2 + 0.5*X3 + noise
y_multi = 1 + 2*X1 - 1.5*X2 + 0.5*X3 + np.random.normal(0, 1, n_samples)

# Create feature matrix
X_multi = np.column_stack([X1, X2, X3])
X_multi_with_intercept = sm.add_constant(X_multi)

# Fit multiple regression
multi_model = sm.OLS(y_multi, X_multi_with_intercept).fit()

print("Multiple Linear Regression Results:")
print(multi_model.summary())

# Example 5.3: Regularization Comparison
print("\n5.3 Regularization Comparison (Ridge vs Lasso)")

# Create dataset with multicollinearity
np.random.seed(42)
n_features = 20
n_samples = 100

# Generate correlated features
X_corr = np.random.randn(n_samples, n_features)
# Add correlation between first few features
for i in range(1, 5):
    X_corr[:, i] = X_corr[:, 0] + 0.5 * np.random.randn(n_samples)

# True coefficients (sparse - only first 5 are non-zero)
true_coef = np.zeros(n_features)
true_coef[:5] = [2, -1.5, 1, -0.5, 0.8]
y_reg = X_corr @ true_coef + 0.1 * np.random.randn(n_samples)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_corr, y_reg, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit models
ridge = Ridge(alpha=0.1)
lasso = Lasso(alpha=0.01)
ols = LinearRegression()

ridge.fit(X_train_scaled, y_train)
lasso.fit(X_train_scaled, y_train)
ols.fit(X_train_scaled, y_train)

# Predictions
ridge_pred = ridge.predict(X_test_scaled)
lasso_pred = lasso.predict(X_test_scaled)
ols_pred = ols.predict(X_test_scaled)

# Calculate R² scores
ridge_r2 = r2_score(y_test, ridge_pred)
lasso_r2 = r2_score(y_test, lasso_pred)
ols_r2 = r2_score(y_test, ols_pred)

print("Regularization Comparison:")
print(f"OLS R²: {ols_r2:.3f}")
print(f"Ridge R²: {ridge_r2:.3f}")
print(f"Lasso R²: {lasso_r2:.3f}")

print("
Coefficient Analysis:")
print("True coefficients (first 10):", true_coef[:10])
print("OLS coefficients (first 10):", ols.coef_[:10])
print("Ridge coefficients (first 10):", ridge.coef_[:10])
print("Lasso coefficients (first 10):", lasso.coef_[:10])

# Count non-zero coefficients
ols_nonzero = np.sum(np.abs(ols.coef_) > 0.01)
ridge_nonzero = np.sum(np.abs(ridge.coef_) > 0.01)
lasso_nonzero = np.sum(np.abs(lasso.coef_) > 0.01)

print("
Feature Selection:")
print(f"OLS non-zero coefficients: {ols_nonzero}/{n_features}")
print(f"Ridge non-zero coefficients: {ridge_nonzero}/{n_features}")
print(f"Lasso non-zero coefficients: {lasso_nonzero}/{n_features}")

# =============================================================================
# 6. PRACTICAL APPLICATIONS
# =============================================================================

print("\n\n6. Practical Applications")
print("=" * 25)

# Example 6.1: A/B Testing Simulation
print("6.1 A/B Testing Simulation")

def ab_test_simulation(n_users=10000, conversion_rate_a=0.12, conversion_rate_b=0.14, alpha=0.05):
    """Simulate A/B test for conversion rates"""

    # Generate conversion data
    conversions_a = np.random.binomial(1, conversion_rate_a, n_users//2)
    conversions_b = np.random.binomial(1, conversion_rate_b, n_users//2)

    # Calculate conversion rates
    rate_a = np.mean(conversions_a)
    rate_b = np.mean(conversions_b)

    # Perform two-sample proportion test
    count = np.array([np.sum(conversions_a), np.sum(conversions_b)])
    nobs = np.array([len(conversions_a), len(conversions_b)])

    z_stat, p_value = stats.proportions_ztest(count, nobs)

    print(f"A/B Test Results:")
    print(f"Group A conversion rate: {rate_a:.3f} (n={nobs[0]})")
    print(f"Group B conversion rate: {rate_b:.3f} (n={nobs[1]})")
    print(f"Absolute difference: {rate_b - rate_a:.3f}")
    print(f"Relative improvement: {((rate_b - rate_a) / rate_a * 100):.1f}%")
    print(f"z-statistic: {z_stat:.3f}")
    print(f"p-value: {p_value:.3f}")

    if p_value < alpha:
        print(f"Statistically significant at α = {alpha} ✓")
        if rate_b > rate_a:
            print("Group B performs better than Group A")
        else:
            print("Group A performs better than Group B")
    else:
        print(f"Not statistically significant at α = {alpha}")

    return rate_a, rate_b, p_value

# Run A/B test simulation
np.random.seed(42)
ab_test_simulation()

print("\n=== End of Module 2 Examples ===")
print("Generated visualization files:")
print("- gradient_descent_demo.png")
print("- probability_distributions.png")
print("- central_limit_theorem.png")

print("\nKey Mathematical Concepts Demonstrated:")
print("- Vector and matrix operations")
print("- Eigenvalue decomposition")
print("- Numerical differentiation and optimization")
print("- Probability distributions and Bayes' theorem")
print("- Statistical inference and hypothesis testing")
print("- Linear regression and regularization")
print("- A/B testing applications")
