#!/usr/bin/env python3
"""
Module 2: Mathematics and Statistics Fundamentals - Exercises
Practical exercises to reinforce mathematical and statistical concepts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("📊 MODULE 2: MATHEMATICS AND STATISTICS FUNDAMENTALS - EXERCISES")
print("=" * 70)

class StatisticsExercises:
    """Interactive exercises for Module 2"""

    def __init__(self):
        self.completed_exercises = 0
        self.total_exercises = 8

    def exercise_1_descriptive_statistics(self):
        """Exercise 1: Descriptive Statistics"""
        print("\n" + "="*60)
        print("📈 EXERCISE 1: DESCRIPTIVE STATISTICS")
        print("="*60)

        print("Calculate descriptive statistics for the following dataset:")
        data = [12, 15, 18, 22, 25, 28, 30, 35, 40, 45]
        print(f"Dataset: {data}")

        # Student calculations
        mean = np.mean(data)
        median = np.median(data)
        mode_result = stats.mode(data, keepdims=True)
        variance = np.var(data, ddof=1)  # Sample variance
        std_dev = np.std(data, ddof=1)  # Sample standard deviation
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        range_val = max(data) - min(data)

        print("
📊 Descriptive Statistics Results:")
        print(f"Mean: {mean:.2f}")
        print(f"Median: {median:.2f}")
        print(f"Mode: {mode_result.mode[0]}")
        print(f"Variance: {variance:.2f}")
        print(f"Standard Deviation: {std_dev:.2f}")
        print(f"Q1 (25th percentile): {q1:.2f}")
        print(f"Q3 (75th percentile): {q3:.2f}")
        print(f"IQR: {iqr:.2f}")
        print(f"Range: {range_val:.2f}")

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Box plot
        axes[0].boxplot(data)
        axes[0].set_title('Box Plot')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)

        # Histogram
        axes[1].hist(data, bins=8, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1].axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.1f}')
        axes[1].axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.1f}')
        axes[1].set_title('Histogram with Mean and Median')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('exercise1_descriptive_stats.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("💡 Key Insights:")
        print("- The data is positively skewed (mean > median)")
        print("- Most values fall between Q1 and Q3 (IQR)")
        print("- There are no obvious outliers in this dataset")

        self.completed_exercises += 1
        input("\nPress Enter to continue to the next exercise...")

    def exercise_2_probability_distributions(self):
        """Exercise 2: Probability Distributions"""
        print("\n" + "="*60)
        print("🎲 EXERCISE 2: PROBABILITY DISTRIBUTIONS")
        print("="*60)

        np.random.seed(42)

        # Generate samples from different distributions
        normal_data = np.random.normal(loc=50, scale=10, size=1000)
        exponential_data = np.random.exponential(scale=2, size=1000)
        binomial_data = np.random.binomial(n=10, p=0.3, size=1000)

        print("Generated samples from three different distributions:")
        print("- Normal distribution (μ=50, σ=10)")
        print("- Exponential distribution (λ=0.5)")
        print("- Binomial distribution (n=10, p=0.3)")

        # Calculate statistics
        distributions = {
            'Normal': normal_data,
            'Exponential': exponential_data,
            'Binomial': binomial_data
        }

        print("
📊 Distribution Statistics:")
        print("-" * 50)
        for name, data in distributions.items():
            mean = np.mean(data)
            std = np.std(data)
            print(f"{name:<15} Mean: {mean:.3f}, Std: {std:.3f}")

        # Visualize distributions
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, (name, data) in enumerate(distributions.items()):
            axes[i].hist(data, bins=30, alpha=0.7, density=True, color=['skyblue', 'lightcoral', 'lightgreen'][i])

            # Add theoretical curves
            if name == 'Normal':
                x = np.linspace(min(data), max(data), 100)
                axes[i].plot(x, stats.norm.pdf(x, 50, 10), 'r-', linewidth=2, label='Theoretical')
            elif name == 'Exponential':
                x = np.linspace(0, max(data), 100)
                axes[i].plot(x, stats.expon.pdf(x, scale=2), 'r-', linewidth=2, label='Theoretical')
            elif name == 'Binomial':
                x = np.arange(0, 11)
                axes[i].bar(x, stats.binom.pmf(x, 10, 0.3), alpha=0.5, color='orange', label='Theoretical')

            axes[i].set_title(f'{name} Distribution')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Density' if name != 'Binomial' else 'Probability')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('exercise2_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("💡 Key Insights:")
        print("- Normal: Symmetric, bell-shaped curve")
        print("- Exponential: Skewed right, memoryless property")
        print("- Binomial: Discrete, counts successes in trials")

        self.completed_exercises += 1
        input("\nPress Enter to continue to the next exercise...")

    def exercise_3_central_limit_theorem(self):
        """Exercise 3: Central Limit Theorem Demonstration"""
        print("\n" + "="*60)
        print("📏 EXERCISE 3: CENTRAL LIMIT THEOREM")
        print("="*60)

        np.random.seed(42)

        # Population: Exponential distribution (skewed)
        population_size = 10000
        population = np.random.exponential(scale=2, size=population_size)

        print(f"Population: Exponential distribution with {population_size} samples")
        print(".3f")
        print(".3f")

        # Sample sizes to test
        sample_sizes = [5, 10, 30, 50, 100]
        n_samples = 1000

        print(f"\nTesting sample sizes: {sample_sizes}")
        print(f"Number of samples per size: {n_samples}")

        # Calculate sample means for each sample size
        sample_means = {}

        for n in sample_sizes:
            means = []
            for _ in range(n_samples):
                sample = np.random.choice(population, size=n, replace=True)
                means.append(np.mean(sample))
            sample_means[n] = np.array(means)

        # Visualize the results
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        for i, n in enumerate(sample_sizes):
            # Histogram of sample means
            axes[i].hist(sample_means[n], bins=30, alpha=0.7, density=True,
                        color='skyblue', edgecolor='black')

            # Add normal curve approximation
            mean_of_means = np.mean(sample_means[n])
            std_of_means = np.std(sample_means[n])
            x = np.linspace(mean_of_means - 3*std_of_means, mean_of_means + 3*std_of_means, 100)
            axes[i].plot(x, stats.norm.pdf(x, mean_of_means, std_of_means),
                        'r-', linewidth=2, label='Normal approximation')

            axes[i].axvline(np.mean(population), color='green', linestyle='--',
                           linewidth=2, label=f'Population mean: {np.mean(population):.2f}')
            axes[i].axvline(mean_of_means, color='orange', linestyle='--',
                           linewidth=2, label=f'Sample mean: {mean_of_means:.2f}')

            axes[i].set_title(f'Sample Size n = {n}')
            axes[i].set_xlabel('Sample Mean')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        # Summary statistics
        axes[5].axis('off')
        summary_text = "CENTRAL LIMIT THEOREM SUMMARY\n\n"
        summary_text += f"Population Mean: {np.mean(population):.3f}\n"
        summary_text += f"Population Std: {np.std(population):.3f}\n\n"

        for n in sample_sizes:
            mean_of_means = np.mean(sample_means[n])
            std_of_means = np.std(sample_means[n])
            expected_std = np.std(population) / np.sqrt(n)

            summary_text += f"n={n:3d}: Mean={mean_of_means:.3f}, Std={std_of_means:.3f} (Expected: {expected_std:.3f})\n"

        axes[5].text(0.1, 0.5, summary_text, transform=axes[5].transAxes,
                    fontsize=10, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.5))

        plt.tight_layout()
        plt.savefig('exercise3_clt.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("💡 Key Insights:")
        print("- Sample means become more normally distributed as sample size increases")
        print("- The mean of sample means approaches the population mean")
        print("- The standard error decreases as sample size increases (SE = σ/√n)")

        self.completed_exercises += 1
        input("\nPress Enter to continue to the next exercise...")

    def exercise_4_hypothesis_testing(self):
        """Exercise 4: Hypothesis Testing"""
        print("\n" + "="*60)
        print("🧪 EXERCISE 4: HYPOTHESIS TESTING")
        print("="*60)

        np.random.seed(42)

        # Scenario: Testing if a new drug increases response time
        # H₀: μ ≤ 10 seconds (no improvement)
        # H₁: μ > 10 seconds (improvement)

        print("Scenario: A new drug is claimed to increase reaction time.")
        print("We test on 30 patients. Population mean reaction time is 10 seconds.")
        print()
        print("Hypotheses:")
        print("H₀: μ ≤ 10 seconds (null hypothesis)")
        print("H₁: μ > 10 seconds (alternative hypothesis)")
        print("α = 0.05 (significance level)")

        # Generate sample data (drug increases reaction time by 1.5 seconds on average)
        true_mean = 11.5  # Drug effect
        sample_size = 30
        sample_std = 2.0

        sample = np.random.normal(true_mean, sample_std, sample_size)

        print("
📊 Sample Statistics:")
        print(f"Sample size: {sample_size}")
        print(".3f")
        print(".3f")

        # Perform one-sample t-test
        hypothesized_mean = 10.0
        t_statistic, p_value = stats.ttest_1samp(sample, hypothesized_mean)

        print("
🧪 Hypothesis Test Results:")
        print(".3f")
        print(".6f")

        # Decision
        alpha = 0.05
        if p_value < alpha:
            decision = "Reject H₀"
            conclusion = "The drug significantly increases reaction time"
        else:
            decision = "Fail to reject H₀"
            conclusion = "No significant evidence that the drug increases reaction time"

        print(f"Significance level (α): {alpha}")
        print(f"Decision: {decision}")
        print(f"Conclusion: {conclusion}")

        # Power analysis visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Sampling distribution visualization
        x = np.linspace(8, 14, 1000)
        null_dist = stats.norm.pdf(x, hypothesized_mean, sample_std/np.sqrt(sample_size))
        sample_dist = stats.norm.pdf(x, true_mean, sample_std/np.sqrt(sample_size))

        axes[0].plot(x, null_dist, 'b-', linewidth=2, label='Null Distribution (H₀)')
        axes[0].plot(x, sample_dist, 'r-', linewidth=2, label='True Distribution (H₁)')
        axes[0].axvline(np.mean(sample), color='green', linestyle='--', linewidth=2,
                       label=f'Sample Mean: {np.mean(sample):.2f}')
        axes[0].axvline(hypothesized_mean + stats.norm.ppf(1-alpha) * (sample_std/np.sqrt(sample_size)),
                       color='orange', linestyle='--', linewidth=2, label='Critical Value')

        axes[0].fill_between(x, null_dist, where=(x > hypothesized_mean + stats.norm.ppf(1-alpha) * (sample_std/np.sqrt(sample_size))),
                           alpha=0.3, color='blue', label='Rejection Region')
        axes[0].set_title('Hypothesis Test Visualization')
        axes[0].set_xlabel('Sample Mean')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # P-value interpretation
        axes[1].plot(x, null_dist, 'b-', linewidth=2, label='Null Distribution')
        axes[1].axvline(np.mean(sample), color='red', linestyle='--', linewidth=3,
                       label=f'Observed Mean: {np.mean(sample):.2f}')

        # Shade p-value region
        if t_statistic > 0:  # Right-tailed test
            axes[1].fill_between(x, null_dist, where=(x >= np.mean(sample)),
                               alpha=0.5, color='red', label=f'p-value = {p_value:.4f}')
        else:  # Left-tailed test
            axes[1].fill_between(x, null_dist, where=(x <= np.mean(sample)),
                               alpha=0.5, color='red', label=f'p-value = {p_value:.4f}')

        axes[1].set_title('P-Value Visualization')
        axes[1].set_xlabel('Sample Mean')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('exercise4_hypothesis_testing.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("💡 Key Concepts:")
        print("- Type I error (α): False positive - rejecting H₀ when true")
        print("- Type II error (β): False negative - failing to reject H₀ when false")
        print("- Power (1-β): Probability of correctly rejecting H₀ when false")
        print("- p-value: Probability of observing data as extreme as ours assuming H₀ is true")

        self.completed_exercises += 1
        input("\nPress Enter to continue to the next exercise...")

    def exercise_5_confidence_intervals(self):
        """Exercise 5: Confidence Intervals"""
        print("\n" + "="*60)
        print("📏 EXERCISE 5: CONFIDENCE INTERVALS")
        print("="*60)

        np.random.seed(42)

        # Scenario: Estimating population mean height
        print("Scenario: Estimating the average height of students in a university.")
        print("We survey 50 students and want to estimate the population mean height.")

        # Generate sample data
        true_population_mean = 170  # cm
        population_std = 8  # cm
        sample_size = 50

        sample = np.random.normal(true_population_mean, population_std, sample_size)

        print("
📊 Sample Data:")
        print(f"Sample size: {sample_size}")
        print(".2f")
        print(".2f")

        # Calculate 95% confidence interval
        confidence_level = 0.95
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)  # Sample standard deviation
        standard_error = sample_std / np.sqrt(sample_size)

        # t-critical value for 95% confidence with df = n-1
        t_critical = stats.t.ppf((1 + confidence_level) / 2, sample_size - 1)

        margin_of_error = t_critical * standard_error
        ci_lower = sample_mean - margin_of_error
        ci_upper = sample_mean + margin_of_error

        print("
🎯 95% Confidence Interval Calculation:")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")

        # Check if true population mean is in the interval
        contains_true_mean = ci_lower <= true_population_mean <= ci_upper
        print(f"\nTrue population mean ({true_population_mean} cm) is {'within' if contains_true_mean else 'outside'} the confidence interval")

        # Multiple confidence intervals demonstration
        print("
🔄 Multiple Samples Demonstration:")
        print("Generating 20 different samples to show how confidence intervals vary...")

        n_simulations = 20
        cis = []

        for i in range(n_simulations):
            sim_sample = np.random.normal(true_population_mean, population_std, sample_size)
            sim_mean = np.mean(sim_sample)
            sim_std = np.std(sim_sample, ddof=1)
            sim_se = sim_std / np.sqrt(sample_size)
            sim_t_crit = stats.t.ppf((1 + confidence_level) / 2, sample_size - 1)
            sim_margin = sim_t_crit * sim_se
            cis.append((sim_mean - sim_margin, sim_mean + sim_margin))

        # Visualization
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot confidence intervals
        for i, (lower, upper) in enumerate(cis):
            color = 'green' if lower <= true_population_mean <= upper else 'red'
            ax.plot([lower, upper], [i, i], color=color, linewidth=3, marker='o', markersize=4)
            ax.plot([true_population_mean, true_population_mean], [i-0.2, i+0.2],
                   color='blue', linewidth=2, alpha=0.7)

        ax.axvline(true_population_mean, color='blue', linestyle='--', linewidth=2,
                  label=f'True Population Mean: {true_population_mean} cm')
        ax.set_xlabel('Height (cm)')
        ax.set_ylabel('Sample Number')
        ax.set_title('95% Confidence Intervals from 20 Different Samples')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add text annotation
        green_count = sum(1 for lower, upper in cis if lower <= true_population_mean <= upper)
        ax.text(0.02, 0.98, f'{green_count}/{n_simulations} intervals contain the true mean',
               transform=ax.transAxes, fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.savefig('exercise5_confidence_intervals.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("💡 Key Insights:")
        print("- 95% confidence intervals should contain the true parameter 95% of the time")
        print("- Wider intervals indicate more uncertainty (smaller samples, higher variability)")
        print("- Confidence intervals get narrower as sample size increases")
        print("- The interval is random, not the parameter - the parameter is fixed")

        self.completed_exercises += 1
        input("\nPress Enter to continue to the next exercise...")

    def exercise_6_correlation_analysis(self):
        """Exercise 6: Correlation Analysis"""
        print("\n" + "="*60)
        print("📈 EXERCISE 6: CORRELATION ANALYSIS")
        print("="*60)

        np.random.seed(42)

        # Generate correlated data
        n_points = 100

        # Strong positive correlation
        x1 = np.random.normal(50, 10, n_points)
        y1 = 2 * x1 + np.random.normal(0, 5, n_points)  # y = 2x + noise

        # Moderate negative correlation
        x2 = np.random.normal(0, 1, n_points)
        y2 = -0.7 * x2 + np.random.normal(0, 0.5, n_points)  # y = -0.7x + noise

        # No correlation
        x3 = np.random.normal(25, 5, n_points)
        y3 = np.random.normal(75, 8, n_points)  # Independent variables

        datasets = {
            'Strong Positive': (x1, y1),
            'Moderate Negative': (x2, y2),
            'No Correlation': (x3, y3)
        }

        print("Analyzing correlation in three different datasets:")

        # Calculate and display correlations
        results = {}
        for name, (x, y) in datasets.items():
            pearson_r, pearson_p = stats.pearsonr(x, y)
            spearman_r, spearman_p = stats.spearmanr(x, y)

            results[name] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p
            }

            print(f"\n{name} Correlation:")
            print(".3f")
            print(".3f")
            print(".3f")
            print(".3f")

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, (name, (x, y)) in enumerate(datasets.items()):
            # Scatter plot
            axes[i].scatter(x, y, alpha=0.6, color=['skyblue', 'lightcoral', 'lightgreen'][i])

            # Add regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            x_line = np.linspace(min(x), max(x), 100)
            y_line = slope * x_line + intercept
            axes[i].plot(x_line, y_line, 'r-', linewidth=2, label=f'r = {r_value:.3f}')

            axes[i].set_title(f'{name} Correlation\nr = {results[name]["pearson_r"]:.3f}')
            axes[i].set_xlabel('X variable')
            axes[i].set_ylabel('Y variable')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('exercise6_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Correlation matrix example
        print("
📊 Correlation Matrix Example:")
        print("Creating a correlation matrix for multiple variables...")

        # Generate multivariate data
        data = np.random.multivariate_normal(
            mean=[50, 60, 70, 80],
            cov=[[100, 60, 40, 20],
                 [60, 100, 50, 30],
                 [40, 50, 100, 40],
                 [20, 30, 40, 100]],
            size=200
        )

        df = pd.DataFrame(data, columns=['Math_Score', 'Science_Score', 'English_Score', 'History_Score'])

        correlation_matrix = df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix: Student Subject Scores')
        plt.tight_layout()
        plt.savefig('exercise6_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("💡 Key Concepts:")
        print("- Pearson correlation: Measures linear relationship between continuous variables")
        print("- Spearman correlation: Measures monotonic relationship (rank-based)")
        print("- Correlation ≠ Causation: Association doesn't imply cause-and-effect")
        print("- Correlation matrices help identify relationships between multiple variables")

        self.completed_exercises += 1
        input("\nPress Enter to continue to the next exercise...")

    def exercise_7_linear_regression(self):
        """Exercise 7: Linear Regression"""
        print("\n" + "="*60)
        print("📈 EXERCISE 7: LINEAR REGRESSION")
        print("="*60)

        np.random.seed(42)

        # Generate data with known relationship
        n_points = 100
        true_slope = 2.5
        true_intercept = 10
        noise_level = 3

        x = np.random.uniform(0, 20, n_points)
        y = true_slope * x + true_intercept + np.random.normal(0, noise_level, n_points)

        print("Generated data with known relationship: y = 2.5x + 10 + noise")
        print(f"Sample size: {n_points} points")

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        print("
📊 Linear Regression Results:")
        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")

        # Calculate predictions and residuals
        y_pred = slope * x + intercept
        residuals = y - y_pred
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)

        print("
📈 Model Performance:")
        print(".3f")
        print(".3f")

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Scatter plot with regression line
        axes[0,0].scatter(x, y, alpha=0.6, color='skyblue', label='Data points')
        axes[0,0].plot(x, y_pred, 'r-', linewidth=2, label=f'Fitted line: y = {slope:.2f}x + {intercept:.2f}')
        axes[0,0].plot(x, true_slope * x + true_intercept, 'g--', linewidth=2,
                      label=f'True line: y = {true_slope}x + {true_intercept}')
        axes[0,0].set_title('Linear Regression Fit')
        axes[0,0].set_xlabel('X')
        axes[0,0].set_ylabel('Y')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # Residuals plot
        axes[0,1].scatter(y_pred, residuals, alpha=0.6, color='lightcoral')
        axes[0,1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0,1].set_title('Residuals vs Fitted Values')
        axes[0,1].set_xlabel('Fitted Values')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].grid(True, alpha=0.3)

        # Q-Q plot for residuals normality
        stats.probplot(residuals, dist="norm", plot=axes[1,0])
        axes[1,0].set_title('Q-Q Plot of Residuals')
        axes[1,0].grid(True, alpha=0.3)

        # Residuals histogram
        axes[1,1].hist(residuals, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1,1].axvline(np.mean(residuals), color='red', linestyle='--', linewidth=2,
                         label=f'Mean: {np.mean(residuals):.3f}')
        axes[1,1].set_title('Residuals Distribution')
        axes[1,1].set_xlabel('Residual Value')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('exercise7_linear_regression.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("💡 Key Diagnostics:")
        print("- Residuals should be randomly scattered around zero (homoscedasticity)")
        print("- Residuals should be normally distributed")
        print("- No patterns in residuals vs fitted values")
        print("- R² indicates proportion of variance explained by the model")

        self.completed_exercises += 1
        input("\nPress Enter to continue to the next exercise...")

    def exercise_8_practical_application(self):
        """Exercise 8: Practical Statistics Application"""
        print("\n" + "="*60)
        print("🎯 EXERCISE 8: PRACTICAL STATISTICS APPLICATION")
        print("="*60)

        print("Real-world scenario: A/B testing for website conversion rates")
        print("Company wants to test if a new website design increases conversion rate.")

        np.random.seed(42)

        # A/B test setup
        n_visitors_control = 5000
        n_visitors_variant = 5000

        # Control group: 12% conversion rate
        # Variant group: 14% conversion rate (2% absolute improvement)
        control_rate = 0.12
        variant_rate = 0.14

        control_conversions = np.random.binomial(1, control_rate, n_visitors_control)
        variant_conversions = np.random.binomial(1, variant_rate, n_visitors_variant)

        control_conv_rate = np.mean(control_conversions)
        variant_conv_rate = np.mean(variant_conversions)

        print("
📊 A/B Test Results:")
        print(f"Control group: {n_visitors_control} visitors, {control_conv_rate:.3f} conversion rate")
        print(f"Variant group: {n_visitors_variant} visitors, {variant_conv_rate:.3f} conversion rate")
        print(".3f")
        print(".3f")

        # Statistical significance test (two-proportion z-test)
        from statsmodels.stats.proportion import proportions_ztest

        count = np.array([np.sum(control_conversions), np.sum(variant_conversions)])
        nobs = np.array([n_visitors_control, n_visitors_variant])

        z_stat, p_value = proportions_ztest(count, nobs, alternative='smaller')

        print("
🧪 Statistical Test Results:")
        print(".3f")
        print(".6f")

        alpha = 0.05
        if p_value < alpha:
            result = "STATISTICALLY SIGNIFICANT"
            recommendation = "🚀 Launch the new design - it significantly improves conversion!"
        else:
            result = "NOT statistically significant"
            recommendation = "⏳ Continue testing or consider other design changes"

        print(f"Significance level: α = {alpha}")
        print(f"Result: {result}")
        print(f"Recommendation: {recommendation}")

        # Confidence intervals for conversion rates
        from statsmodels.stats.proportion import proportion_confint

        control_ci = proportion_confint(np.sum(control_conversions), n_visitors_control, alpha=0.05, method='wilson')
        variant_ci = proportion_confint(np.sum(variant_conversions), n_visitors_variant, alpha=0.05, method='wilson')

        print("
📏 95% Confidence Intervals:")
        print(".3f")
        print(".3f")

        # Power analysis
        from statsmodels.stats.power import NormalIndPower
        from statsmodels.stats.proportion import proportion_effectsize

        effect_size = proportion_effectsize(control_rate, variant_rate)
        power_analysis = NormalIndPower()
        power = power_analysis.power(effect_size, n_visitors_control, alpha=0.05)

        print("
⚡ Power Analysis:")
        print(".3f")
        print(".3f")

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Conversion rates comparison
        groups = ['Control', 'Variant']
        rates = [control_conv_rate, variant_conv_rate]
        cis = [control_ci, variant_ci]

        axes[0,0].bar(groups, rates, color=['skyblue', 'lightcoral'], alpha=0.7)
        axes[0,0].errorbar(groups, rates,
                          yerr=[(rates[0]-cis[0][0], cis[0][1]-rates[0]),
                                (rates[1]-cis[1][0], cis[1][1]-rates[1])],
                          fmt='none', color='black', capsize=5)
        axes[0,0].set_title('Conversion Rates with 95% Confidence Intervals')
        axes[0,0].set_ylabel('Conversion Rate')
        axes[0,0].grid(True, alpha=0.3)

        # Statistical significance visualization
        x_vals = np.linspace(-4, 4, 1000)
        null_dist = stats.norm.pdf(x_vals, 0, 1)

        axes[0,1].plot(x_vals, null_dist, 'b-', linewidth=2, label='Null Distribution')
        axes[0,1].axvline(z_stat, color='red', linestyle='--', linewidth=3,
                         label=f'Observed z = {z_stat:.2f}')
        axes[0,1].axvline(stats.norm.ppf(1-alpha), color='orange', linestyle='--', linewidth=2,
                         label=f'Critical value: {stats.norm.ppf(1-alpha):.2f}')

        # Shade rejection region
        axes[0,1].fill_between(x_vals, null_dist,
                              where=(x_vals > stats.norm.ppf(1-alpha)),
                              alpha=0.3, color='blue', label='Rejection Region')

        axes[0,1].set_title('Statistical Significance Test')
        axes[0,1].set_xlabel('z-statistic')
        axes[0,1].set_ylabel('Density')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # Sample size vs power analysis
        sample_sizes = np.arange(1000, 10001, 1000)
        powers = [power_analysis.power(effect_size, n, alpha=0.05) for n in sample_sizes]

        axes[1,0].plot(sample_sizes, powers, 'g-', linewidth=2, marker='o')
        axes[1,0].axhline(0.8, color='red', linestyle='--', linewidth=2, label='Target Power (80%)')
        axes[1,0].set_title('Statistical Power vs Sample Size')
        axes[1,0].set_xlabel('Sample Size (per group)')
        axes[1,0].set_ylabel('Statistical Power')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # Practical significance (lift) analysis
        lifts = np.linspace(0.01, 0.10, 50)  # 1% to 10% absolute lift
        powers_lift = [power_analysis.power(proportion_effectsize(control_rate, control_rate + lift),
                                          n_visitors_control, alpha=0.05) for lift in lifts]

        axes[1,1].plot(lifts * 100, powers_lift, 'purple', linewidth=2)
        axes[1,1].axhline(0.8, color='red', linestyle='--', linewidth=2, label='Target Power (80%)')
        axes[1,1].axvline((variant_rate - control_rate) * 100, color='green', linestyle='--',
                         linewidth=2, label=f'Observed lift: {(variant_rate - control_rate)*100:.1f}%')
        axes[1,1].set_title('Statistical Power vs Effect Size')
        axes[1,1].set_xlabel('Absolute Lift (%)')
        axes[1,1].set_ylabel('Statistical Power')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('exercise8_ab_testing.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("💡 A/B Testing Best Practices:")
        print("- Define clear hypothesis and success metrics before testing")
        print("- Ensure adequate sample size for statistical power")
        print("- Consider both statistical and practical significance")
        print("- Account for multiple testing if running many experiments")
        print("- Monitor for external factors that might affect results")

        self.completed_exercises += 1

    def show_progress(self):
        """Show completion progress"""
        print("
📊 EXERCISE COMPLETION SUMMARY"        print("=" * 40)
        print(f"Completed: {self.completed_exercises}/{self.total_exercises} exercises")
        print(".1f")

        if self.completed_exercises == self.total_exercises:
            print("🎉 Congratulations! You have completed all exercises!")
            print("You now have hands-on experience with:")
            print("✓ Descriptive statistics and data visualization")
            print("✓ Probability distributions and their properties")
            print("✓ Central Limit Theorem and sampling distributions")
            print("✓ Hypothesis testing and p-values")
            print("✓ Confidence intervals and their interpretation")
            print("✓ Correlation analysis and its limitations")
            print("✓ Linear regression and model diagnostics")
            print("✓ A/B testing and experimental design")

    def run_all_exercises(self):
        """Run all exercises in sequence"""
        exercises = [
            self.exercise_1_descriptive_statistics,
            self.exercise_2_probability_distributions,
            self.exercise_3_central_limit_theorem,
            self.exercise_4_hypothesis_testing,
            self.exercise_5_confidence_intervals,
            self.exercise_6_correlation_analysis,
            self.exercise_7_linear_regression,
            self.exercise_8_practical_application
        ]

        for exercise in exercises:
            exercise()

        self.show_progress()

if __name__ == "__main__":
    exercises = StatisticsExercises()
    exercises.run_all_exercises()
