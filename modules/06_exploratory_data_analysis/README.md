# Module 6: Exploratory Data Analysis (EDA)

## Overview
Exploratory Data Analysis (EDA) is the process of analyzing and visualizing data to understand its main characteristics, uncover patterns, and identify relationships between variables. This module teaches systematic approaches to explore datasets, create meaningful visualizations, and extract actionable insights that inform subsequent modeling decisions.

## Learning Objectives
By the end of this module, you will be able to:
- Perform systematic univariate and multivariate analysis
- Create comprehensive EDA reports with visualizations
- Identify data distributions, outliers, and anomalies
- Understand relationships between variables through correlation analysis
- Apply statistical tests to validate hypotheses
- Create automated EDA pipelines for rapid data understanding
- Communicate findings effectively through data storytelling

## 1. Introduction to EDA

### 1.1 What is Exploratory Data Analysis?

EDA is an approach to analyzing datasets to:
- **Summarize main characteristics** of the data
- **Discover patterns and relationships** between variables
- **Identify anomalies and outliers** that need attention
- **Test assumptions** about the data
- **Generate hypotheses** for further investigation
- **Inform feature engineering** and modeling decisions

### 1.2 EDA vs Confirmatory Data Analysis

#### Exploratory Data Analysis (EDA)
- **Purpose**: Discover patterns, generate hypotheses
- **Approach**: Flexible, open-ended exploration
- **Methods**: Visualization, summary statistics, pattern discovery
- **Outcome**: Insights, hypotheses, data understanding

#### Confirmatory Data Analysis (CDA)
- **Purpose**: Test specific hypotheses
- **Approach**: Structured, hypothesis-driven
- **Methods**: Statistical tests, significance testing
- **Outcome**: Validation of hypotheses, statistical evidence

### 1.3 EDA Workflow

1. **Data Collection**: Gather relevant data sources
2. **Data Cleaning**: Handle missing values, outliers, inconsistencies
3. **Univariate Analysis**: Understand individual variables
4. **Bivariate Analysis**: Explore relationships between pairs of variables
5. **Multivariate Analysis**: Understand complex interactions
6. **Hypothesis Generation**: Formulate questions and hypotheses
7. **Insight Communication**: Present findings and recommendations

## 2. Univariate Analysis

### 2.1 Analyzing Single Variables

#### Numerical Variables
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_numerical_variable(df: pd.DataFrame, column: str):
    """Comprehensive analysis of a numerical variable"""

    data = df[column].dropna()

    print(f"\n=== Analysis of {column} ===")
    print(f"Data Type: {df[column].dtype}")
    print(f"Missing Values: {df[column].isnull().sum()} ({df[column].isnull().sum()/len(df)*100:.1f}%)")
    print(f"Total Observations: {len(data)}")

    # Basic statistics
    print("
Basic Statistics:")
    print(f"Mean: {data.mean():.3f}")
    print(f"Median: {data.median():.3f}")
    print(f"Mode: {data.mode().iloc[0] if len(data.mode()) > 0 else 'N/A'}")
    print(f"Standard Deviation: {data.std():.3f}")
    print(f"Variance: {data.var():.3f}")
    print(f"Range: {data.max() - data.min():.3f}")
    print(f"Interquartile Range: {data.quantile(0.75) - data.quantile(0.25):.3f}")

    # Distribution shape
    skewness = data.skew()
    kurtosis = data.kurtosis()
    print("
Distribution Shape:")
    print(f"Skewness: {skewness:.3f} ({'Right-skewed' if skewness > 0.5 else 'Left-skewed' if skewness < -0.5 else 'Approximately symmetric'})")
    print(f"Kurtosis: {kurtosis:.3f} ({'Heavy-tailed' if kurtosis > 0.5 else 'Light-tailed' if kurtosis < -0.5 else 'Normal-like'})")

    # Percentiles
    print("
Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"{p}th percentile: {data.quantile(p/100):.3f}")

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Univariate Analysis: {column}', fontsize=16, fontweight='bold')

    # Histogram with KDE
    sns.histplot(data=data, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution (Histogram + KDE)')
    axes[0, 0].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
    axes[0, 0].axvline(data.median(), color='green', linestyle='--', label=f'Median: {data.median():.2f}')
    axes[0, 0].legend()

    # Box plot
    sns.boxplot(y=data, ax=axes[0, 1])
    axes[0, 1].set_title('Box Plot')

    # Q-Q plot
    stats.probplot(data, dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title('Q-Q Plot (Normality Test)')

    # Cumulative distribution
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data))
    axes[1, 0].plot(sorted_data, yvals)
    axes[1, 0].set_title('Cumulative Distribution')
    axes[1, 0].set_xlabel(column)
    axes[1, 0].set_ylabel('Cumulative Probability')

    # Violin plot
    sns.violinplot(y=data, ax=axes[1, 1])
    axes[1, 1].set_title('Violin Plot')

    # Strip plot (sample)
    sample_size = min(1000, len(data))
    sample_data = data.sample(sample_size, random_state=42)
    axes[1, 2].scatter(range(len(sample_data)), sample_data, alpha=0.6, s=10)
    axes[1, 2].set_title('Strip Plot (Sample)')
    axes[1, 2].set_xlabel('Index')
    axes[1, 2].set_ylabel(column)

    plt.tight_layout()
    plt.savefig(f'eda_{column}_univariate.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'skewness': skewness,
        'kurtosis': kurtosis,
        'missing_pct': df[column].isnull().sum() / len(df) * 100
    }

# Usage
# Assuming we have a DataFrame with numerical columns
# results = analyze_numerical_variable(df, 'price')
```

#### Categorical Variables
```python
def analyze_categorical_variable(df: pd.DataFrame, column: str):
    """Comprehensive analysis of a categorical variable"""

    data = df[column].dropna()

    print(f"\n=== Analysis of {column} ===")
    print(f"Data Type: {df[column].dtype}")
    print(f"Missing Values: {df[column].isnull().sum()} ({df[column].isnull().sum()/len(df)*100:.1f}%)")
    print(f"Total Observations: {len(data)}")
    print(f"Unique Categories: {data.nunique()}")

    # Frequency distribution
    value_counts = data.value_counts()
    value_percentages = data.value_counts(normalize=True) * 100

    print("
Frequency Distribution:")
    for category, count in value_counts.items():
        percentage = value_percentages[category]
        print(f"{category}: {count} ({percentage:.1f}%)")

    # Mode analysis
    mode_value = data.mode().iloc[0] if len(data.mode()) > 0 else None
    mode_count = value_counts.iloc[0] if len(value_counts) > 0 else 0
    mode_percentage = value_percentages.iloc[0] if len(value_percentages) > 0 else 0

    print("
Mode Analysis:")
    print(f"Primary Mode: {mode_value}")
    print(f"Mode Frequency: {mode_count} ({mode_percentage:.1f}%)")

    # Entropy (diversity measure)
    proportions = value_percentages / 100
    entropy = -np.sum(proportions * np.log2(proportions + 1e-10))  # Add small value to avoid log(0)
    max_entropy = np.log2(data.nunique())

    print("
Diversity Measures:")
    print(f"Entropy: {entropy:.3f} bits")
    print(f"Maximum Possible Entropy: {max_entropy:.3f} bits")
    print(f"Relative Diversity: {entropy/max_entropy*100:.1f}%")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Categorical Analysis: {column}', fontsize=16, fontweight='bold')

    # Bar chart
    top_n = min(20, len(value_counts))  # Show top 20 categories
    value_counts.head(top_n).plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Frequency Distribution')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Pie chart (only for few categories)
    if data.nunique() <= 10:
        value_counts.head(10).plot(kind='pie', ax=axes[0, 1], autopct='%1.1f%%')
        axes[0, 1].set_title('Proportion Distribution')
        axes[0, 1].set_ylabel('')
    else:
        # Alternative: Horizontal bar chart for many categories
        value_counts.head(15).plot(kind='barh', ax=axes[0, 1])
        axes[0, 1].set_title('Top 15 Categories')
        axes[0, 1].set_xlabel('Count')

    # Cumulative frequency
    cumsum = value_counts.cumsum()
    cumsum_pct = (cumsum / cumsum.iloc[-1] * 100)

    axes[1, 0].plot(range(1, len(cumsum) + 1), cumsum_pct.values)
    axes[1, 0].set_title('Cumulative Frequency (%)')
    axes[1, 0].set_xlabel('Number of Categories')
    axes[1, 0].set_ylabel('Cumulative Percentage')
    axes[1, 0].axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
    axes[1, 0].legend()

    # Category length analysis
    if data.dtype == 'object':
        lengths = data.astype(str).str.len()
        axes[1, 1].hist(lengths, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Category Name Length Distribution')
        axes[1, 1].set_xlabel('Length')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(lengths.mean(), color='red', linestyle='--', label=f'Mean: {lengths.mean():.1f}')
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'eda_{column}_categorical.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'unique_categories': data.nunique(),
        'most_frequent': mode_value,
        'most_frequent_count': mode_count,
        'most_frequent_pct': mode_percentage,
        'entropy': entropy,
        'max_entropy': max_entropy,
        'missing_pct': df[column].isnull().sum() / len(df) * 100
    }

# Usage
# results = analyze_categorical_variable(df, 'category')
```

## 3. Bivariate Analysis

### 3.1 Numerical vs Numerical Variables

#### Scatter Plots and Correlation
```python
def analyze_numerical_bivariate(df: pd.DataFrame, var1: str, var2: str):
    """Analyze relationship between two numerical variables"""

    data = df[[var1, var2]].dropna()

    print(f"\n=== Bivariate Analysis: {var1} vs {var2} ===")
    print(f"Sample Size: {len(data)}")

    # Correlation analysis
    pearson_corr, pearson_p = stats.pearsonr(data[var1], data[var2])
    spearman_corr, spearman_p = stats.spearmanr(data[var1], data[var2])

    print("
Correlation Analysis:")
    print(f"Pearson Correlation: {pearson_corr:.3f} (p-value: {pearson_p:.3f})")
    print(f"Spearman Correlation: {spearman_corr:.3f} (p-value: {spearman_p:.3f})")

    # Strength interpretation
    def interpret_correlation(corr):
        abs_corr = abs(corr)
        if abs_corr < 0.3:
            return "Weak"
        elif abs_corr < 0.7:
            return "Moderate"
        else:
            return "Strong"

    print(f"Correlation Strength: {interpret_correlation(pearson_corr)}")
    print(f"Relationship Direction: {'Positive' if pearson_corr > 0 else 'Negative'}")

    # Statistical significance
    alpha = 0.05
    print(f"Statistical Significance (α={alpha}):")
    print(f"Pearson: {'Significant' if pearson_p < alpha else 'Not significant'}")
    print(f"Spearman: {'Significant' if spearman_p < alpha else 'Not significant'}")

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Bivariate Analysis: {var1} vs {var2}', fontsize=16, fontweight='bold')

    # Scatter plot
    axes[0, 0].scatter(data[var1], data[var2], alpha=0.6, s=30)
    axes[0, 0].set_title('Scatter Plot')
    axes[0, 0].set_xlabel(var1)
    axes[0, 0].set_ylabel(var2)

    # Add regression line
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(data[var1], data[var2])
        x_line = np.linspace(data[var1].min(), data[var1].max(), 100)
        y_line = slope * x_line + intercept
        axes[0, 0].plot(x_line, y_line, color='red', linewidth=2, label=f'R² = {r_value**2:.3f}')
        axes[0, 0].legend()
    except:
        pass

    # Hexbin plot (for large datasets)
    axes[0, 1].hexbin(data[var1], data[var2], gridsize=20, cmap='Blues')
    axes[0, 1].set_title('Hexbin Plot')
    axes[0, 1].set_xlabel(var1)
    axes[0, 1].set_ylabel(var2)

    # 2D histogram
    hist = axes[0, 2].hist2d(data[var1], data[var2], bins=20, cmap='viridis')
    axes[0, 2].set_title('2D Histogram')
    axes[0, 2].set_xlabel(var1)
    axes[0, 2].set_ylabel(var2)
    plt.colorbar(hist[3], ax=axes[0, 2])

    # Residual plot (if regression was performed)
    if 'slope' in locals():
        predicted = slope * data[var1] + intercept
        residuals = data[var2] - predicted

        axes[1, 0].scatter(predicted, residuals, alpha=0.6, s=30)
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        axes[1, 0].set_title('Residual Plot')
        axes[1, 0].set_xlabel('Predicted Values')
        axes[1, 0].set_ylabel('Residuals')

    # Distribution of each variable
    sns.histplot(data[var1], ax=axes[1, 1], kde=True, alpha=0.7)
    axes[1, 1].set_title(f'{var1} Distribution')
    axes[1, 1].set_xlabel(var1)

    sns.histplot(data[var2], ax=axes[1, 2], kde=True, alpha=0.7)
    axes[1, 2].set_title(f'{var2} Distribution')
    axes[1, 2].set_xlabel(var2)

    plt.tight_layout()
    plt.savefig(f'eda_{var1}_vs_{var2}_bivariate.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'sample_size': len(data)
    }

# Usage
# results = analyze_numerical_bivariate(df, 'price', 'rating')
```

#### Categorical vs Categorical Variables
```python
def analyze_categorical_bivariate(df: pd.DataFrame, var1: str, var2: str):
    """Analyze relationship between two categorical variables"""

    data = df[[var1, var2]].dropna()

    print(f"\n=== Bivariate Analysis: {var1} vs {var2} ===")
    print(f"Sample Size: {len(data)}")

    # Contingency table
    contingency_table = pd.crosstab(data[var1], data[var2], margins=True)
    print("
Contingency Table:")
    print(contingency_table)

    # Chi-square test
    if data[var1].nunique() > 1 and data[var2].nunique() > 1:
        chi2, p_value, dof, expected = stats.chi2_contingency(
            pd.crosstab(data[var1], data[var2])
        )

        print("
Chi-Square Test:")
        print(f"Chi-Square Statistic: {chi2:.3f}")
        print(f"P-value: {p_value:.3f}")
        print(f"Degrees of Freedom: {dof}")

        alpha = 0.05
        print(f"Statistical Significance (α={alpha}): {'Significant' if p_value < alpha else 'Not significant'}")

        # Cramer's V (measure of association)
        n = len(data)
        min_dim = min(contingency_table.shape) - 1  # Exclude margins
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        print(f"Cramer's V: {cramers_v:.3f} ({'Strong' if cramers_v > 0.5 else 'Moderate' if cramers_v > 0.3 else 'Weak'} association)")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Categorical Bivariate Analysis: {var1} vs {var2}', fontsize=16, fontweight='bold')

    # Stacked bar chart
    contingency_pct = pd.crosstab(data[var1], data[var2], normalize='index') * 100
    contingency_pct.plot(kind='bar', stacked=True, ax=axes[0, 0])
    axes[0, 0].set_title('Stacked Bar Chart (Row Percentages)')
    axes[0, 0].set_ylabel('Percentage')
    axes[0, 0].legend(title=var2, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Heatmap
    sns.heatmap(contingency_table.iloc[:-1, :-1], annot=True, fmt='d', cmap='YlGnBu', ax=axes[0, 1])
    axes[0, 1].set_title('Contingency Table Heatmap')
    axes[0, 1].set_xlabel(var2)
    axes[0, 1].set_ylabel(var1)

    # Mosaic plot (simplified version)
    # Calculate proportions
    props = pd.crosstab(data[var1], data[var2], normalize='all')

    # Create a simplified mosaic plot
    cumsum = props.cumsum(axis=1)
    left = cumsum - props

    colors = plt.cm.Set3(np.linspace(0, 1, len(props.columns)))

    for i, (idx, row) in enumerate(props.iterrows()):
        for j, (col, val) in enumerate(row.items()):
            axes[1, 0].barh(i, val, left=left.loc[idx, col], color=colors[j], alpha=0.7)

    axes[1, 0].set_title('Mosaic Plot (Simplified)')
    axes[1, 0].set_yticks(range(len(props.index)))
    axes[1, 0].set_yticklabels(props.index)
    axes[1, 0].set_xlabel('Proportion')
    axes[1, 0].legend(props.columns, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Individual distributions
    var1_counts = data[var1].value_counts()
    var2_counts = data[var2].value_counts()

    axes[1, 1].bar(range(len(var1_counts)), var1_counts.values, alpha=0.7, label=var1, color='skyblue')
    axes[1, 1].set_title('Individual Distributions')
    axes[1, 1].set_xlabel('Categories')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_xticks([])  # Remove x ticks for clarity

    # Add second distribution on same plot
    ax2 = axes[1, 1].twinx()
    ax2.bar(range(len(var2_counts)), var2_counts.values, alpha=0.7, label=var2, color='orange', width=0.4)
    ax2.set_ylabel('Count', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Combined legend
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig(f'eda_{var1}_vs_{var2}_categorical.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'contingency_table': contingency_table,
        'chi2_stat': chi2 if 'chi2' in locals() else None,
        'p_value': p_value if 'p_value' in locals() else None,
        'cramers_v': cramers_v if 'cramers_v' in locals() else None
    }

# Usage
# results = analyze_categorical_bivariate(df, 'category', 'region')
```

#### Numerical vs Categorical Variables
```python
def analyze_mixed_bivariate(df: pd.DataFrame, numerical_var: str, categorical_var: str):
    """Analyze relationship between numerical and categorical variables"""

    data = df[[numerical_var, categorical_var]].dropna()

    print(f"\n=== Mixed Bivariate Analysis: {numerical_var} vs {categorical_var} ===")
    print(f"Sample Size: {len(data)}")

    # Group statistics
    group_stats = data.groupby(categorical_var)[numerical_var].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(3)

    print("
Group Statistics:")
    print(group_stats)

    # ANOVA test (if more than 2 groups)
    groups = [group for name, group in data.groupby(categorical_var)[numerical_var]]
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)

        print("
ANOVA Test:")
        print(f"F-statistic: {f_stat:.3f}")
        print(f"P-value: {p_value:.3f}")

        alpha = 0.05
        print(f"Statistical Significance (α={alpha}): {'Significant' if p_value < alpha else 'Not significant'}")

        if p_value < alpha:
            # Post-hoc test (Tukey HSD) if significant
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            tukey_results = pairwise_tukeyhsd(data[numerical_var], data[categorical_var])
            print("
Tukey HSD Post-hoc Test:")
            print(tukey_results)

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Mixed Bivariate Analysis: {numerical_var} vs {categorical_var}', fontsize=16, fontweight='bold')

    # Box plot
    sns.boxplot(data=data, x=categorical_var, y=numerical_var, ax=axes[0, 0])
    axes[0, 0].set_title('Box Plot')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Violin plot
    sns.violinplot(data=data, x=categorical_var, y=numerical_var, ax=axes[0, 1])
    axes[0, 1].set_title('Violin Plot')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Strip plot
    sns.stripplot(data=data, x=categorical_var, y=numerical_var, ax=axes[0, 2], alpha=0.6, jitter=True)
    axes[0, 2].set_title('Strip Plot')
    axes[0, 2].tick_params(axis='x', rotation=45)

    # Bar plot of means
    means = data.groupby(categorical_var)[numerical_var].mean()
    stds = data.groupby(categorical_var)[numerical_var].std()

    axes[1, 0].bar(range(len(means)), means.values, yerr=stds.values, capsize=5, alpha=0.7)
    axes[1, 0].set_title('Mean with Error Bars')
    axes[1, 0].set_xticks(range(len(means)))
    axes[1, 0].set_xticklabels(means.index, rotation=45)
    axes[1, 0].set_ylabel(numerical_var)

    # Swarm plot
    sns.swarmplot(data=data, x=categorical_var, y=numerical_var, ax=axes[1, 1], alpha=0.7)
    axes[1, 1].set_title('Swarm Plot')
    axes[1, 1].tick_params(axis='x', rotation=45)

    # Point plot (mean and confidence intervals)
    sns.pointplot(data=data, x=categorical_var, y=numerical_var, ax=axes[1, 2], errorbar='ci')
    axes[1, 2].set_title('Point Plot (Mean ± CI)')
    axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'eda_{numerical_var}_vs_{categorical_var}_mixed.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'group_stats': group_stats,
        'f_stat': f_stat if 'f_stat' in locals() else None,
        'p_value': p_value if 'p_value' in locals() else None,
        'tukey_results': tukey_results if 'tukey_results' in locals() else None
    }

# Usage
# results = analyze_mixed_bivariate(df, 'price', 'category')
```

## 4. Multivariate Analysis

### 4.1 Correlation Analysis
```python
def create_correlation_analysis(df: pd.DataFrame, method: str = 'pearson'):
    """Create comprehensive correlation analysis"""

    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    if len(numerical_cols) < 2:
        print("Need at least 2 numerical columns for correlation analysis")
        return None

    print(f"\n=== Correlation Analysis ({method}) ===")
    print(f"Variables: {list(numerical_cols)}")

    # Calculate correlation matrix
    if method == 'pearson':
        corr_matrix = df[numerical_cols].corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = df[numerical_cols].corr(method='spearman')
    elif method == 'kendall':
        corr_matrix = df[numerical_cols].corr(method='kendall')
    else:
        raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")

    # Display correlation matrix
    print("
Correlation Matrix:")
    print(corr_matrix.round(3))

    # Find strongest correlations
    # Get upper triangle of correlation matrix
    upper_triangle = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))

    # Find strongest positive and negative correlations
    strongest_positive = upper_triangle.max().max()
    strongest_negative = upper_triangle.min().min()

    pos_indices = np.where(upper_triangle == strongest_positive)
    neg_indices = np.where(upper_triangle == strongest_negative)

    if len(pos_indices[0]) > 0:
        pos_var1, pos_var2 = numerical_cols[pos_indices[0][0]], numerical_cols[pos_indices[1][0]]
        print(f"\nStrongest Positive Correlation: {pos_var1} vs {pos_var2} = {strongest_positive:.3f}")

    if len(neg_indices[0]) > 0:
        neg_var1, neg_var2 = numerical_cols[neg_indices[0][0]], numerical_cols[neg_indices[1][0]]
        print(f"Strongest Negative Correlation: {neg_var1} vs {neg_var2} = {strongest_negative:.3f}")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Correlation Analysis ({method.capitalize()})', fontsize=16, fontweight='bold')

    # Heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, ax=axes[0, 0])
    axes[0, 0].set_title('Correlation Heatmap')

    # Distribution of correlations
    correlations = upper_triangle.values.flatten()
    correlations = correlations[~np.isnan(correlations)]

    axes[0, 1].hist(correlations, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Distribution of Correlation Coefficients')
    axes[0, 1].set_xlabel('Correlation Coefficient')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(correlations), color='red', linestyle='--', label=f'Mean: {np.mean(correlations):.3f}')
    axes[0, 1].legend()

    # Scatter plot matrix (pairplot) - simplified version
    # Show only top correlated pairs
    if len(numerical_cols) > 2:
        # Find top 3 correlated pairs
        corr_pairs = []
        for i in range(len(numerical_cols)):
            for j in range(i+1, len(numerical_cols)):
                corr_pairs.append((numerical_cols[i], numerical_cols[j], abs(corr_matrix.iloc[i, j])))

        corr_pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = corr_pairs[:3] if len(corr_pairs) >= 3 else corr_pairs

        # Create subplot for each top pair
        for idx, (var1, var2, corr) in enumerate(top_pairs):
            if idx < 3:  # Only show top 3
                row = 1
                col = idx
                if col < 3:  # Ensure we don't exceed subplot dimensions
                    axes[row, col].scatter(df[var1], df[var2], alpha=0.6, s=20)
                    axes[row, col].set_title('.3f')
                    axes[row, col].set_xlabel(var1)
                    axes[row, col].set_ylabel(var2)

    plt.tight_layout()
    plt.savefig(f'correlation_analysis_{method}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'correlation_matrix': corr_matrix,
        'method': method,
        'strongest_positive': strongest_positive if 'strongest_positive' in locals() else None,
        'strongest_negative': strongest_negative if 'strongest_negative' in locals() else None
    }

# Usage
# corr_results = create_correlation_analysis(df, method='pearson')
```

### 4.2 Principal Component Analysis (PCA)
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def perform_pca_analysis(df: pd.DataFrame, n_components: int = None, variance_threshold: float = 0.95):
    """Perform PCA analysis for dimensionality reduction"""

    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    if len(numerical_cols) < 2:
        print("Need at least 2 numerical columns for PCA")
        return None

    # Prepare data
    X = df[numerical_cols].dropna()

    if len(X) == 0:
        print("No valid data for PCA after removing missing values")
        return None

    print(f"\n=== PCA Analysis ===")
    print(f"Original dimensions: {X.shape}")
    print(f"Variables: {list(numerical_cols)}")

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    if n_components is None:
        # Determine optimal number of components
        pca_full = PCA()
        pca_full.fit(X_scaled)

        # Find number of components for desired variance
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

        print(f"Components needed for {variance_threshold*100}% variance: {n_components}")

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    print("
Explained Variance:")
    for i, (var, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
        print(f"PC{i+1}: {var:.3f} ({cum_var:.3f} cumulative)")

    # Component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=numerical_cols
    )

    print("
Principal Component Loadings:")
    print(loadings.round(3))

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Principal Component Analysis', fontsize=16, fontweight='bold')

    # Scree plot
    axes[0, 0].plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-', alpha=0.7)
    axes[0, 0].set_title('Scree Plot')
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Explained Variance Ratio')
    axes[0, 0].grid(True, alpha=0.3)

    # Cumulative variance
    axes[0, 1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-', alpha=0.7)
    axes[0, 1].axhline(y=variance_threshold, color='green', linestyle='--', alpha=0.7, label=f'{variance_threshold*100}% threshold')
    axes[0, 1].set_title('Cumulative Explained Variance')
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('Cumulative Variance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Component loadings heatmap
    sns.heatmap(loadings, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[1, 0])
    axes[1, 0].set_title('Component Loadings')

    # PCA scatter plot (first 2 components)
    if X_pca.shape[1] >= 2:
        axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=30)
        axes[1, 1].set_title('PCA Scatter Plot (PC1 vs PC2)')
        axes[1, 1].set_xlabel('Principal Component 1')
        axes[1, 1].set_ylabel('Principal Component 2')
        axes[1, 1].grid(True, alpha=0.3)

        # Add explained variance to axis labels
        pc1_var = explained_variance_ratio[0] * 100
        pc2_var = explained_variance_ratio[1] * 100
        axes[1, 1].set_xlabel('.1f')
        axes[1, 1].set_ylabel('.1f')

    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'pca_model': pca,
        'scaler': scaler,
        'X_pca': X_pca,
        'loadings': loadings,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'n_components': n_components
    }

# Usage
# pca_results = perform_pca_analysis(df, variance_threshold=0.90)
```

## 5. Automated EDA Pipeline

### 5.1 Comprehensive EDA Class
```python
class AutomatedEDA:
    """Automated Exploratory Data Analysis pipeline"""

    def __init__(self, df: pd.DataFrame, target_column: str = None):
        self.df = df.copy()
        self.target_column = target_column
        self.eda_results = {}

    def run_complete_eda(self, output_dir: str = 'eda_outputs'):
        """Run complete EDA analysis"""

        import os
        os.makedirs(output_dir, exist_ok=True)

        print("🚀 Starting Automated EDA Pipeline")
        print("=" * 50)

        # 1. Dataset overview
        self._dataset_overview()

        # 2. Univariate analysis
        self._univariate_analysis()

        # 3. Bivariate analysis
        self._bivariate_analysis()

        # 4. Multivariate analysis
        self._multivariate_analysis()

        # 5. Generate summary report
        self._generate_summary_report(output_dir)

        print("
✅ EDA Pipeline Complete!"        print(f"📊 Results saved to: {output_dir}/")
        print(f"📋 Summary report: {output_dir}/eda_summary_report.txt")

        return self.eda_results

    def _dataset_overview(self):
        """Generate dataset overview"""
        print("\n📊 Dataset Overview")

        overview = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'duplicate_rows': self.df.duplicated().sum()
        }

        # Missing data summary
        missing_summary = self.df.isnull().sum()
        missing_percent = (missing_summary / len(self.df)) * 100

        overview['missing_summary'] = {
            'total_missing': missing_summary.sum(),
            'columns_with_missing': (missing_summary > 0).sum(),
            'missing_by_column': missing_summary.to_dict(),
            'missing_percent_by_column': missing_percent.to_dict()
        }

        self.eda_results['overview'] = overview

        print(f"Shape: {overview['shape']}")
        print(f"Columns: {len(overview['columns'])}")
        print(f"Memory Usage: {overview['memory_usage'] / 1024 / 1024:.2f} MB")
        print(f"Duplicate Rows: {overview['duplicate_rows']}")
        print(f"Missing Values: {overview['missing_summary']['total_missing']}")

    def _univariate_analysis(self):
        """Perform univariate analysis"""
        print("\n📈 Univariate Analysis")

        univariate_results = {}

        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64']:
                # Numerical analysis
                stats = {
                    'type': 'numerical',
                    'count': self.df[col].count(),
                    'missing': self.df[col].isnull().sum(),
                    'mean': self.df[col].mean(),
                    'median': self.df[col].median(),
                    'std': self.df[col].std(),
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'skewness': self.df[col].skew(),
                    'kurtosis': self.df[col].kurtosis()
                }
                univariate_results[col] = stats

            else:
                # Categorical analysis
                value_counts = self.df[col].value_counts()
                stats = {
                    'type': 'categorical',
                    'count': self.df[col].count(),
                    'missing': self.df[col].isnull().sum(),
                    'unique': self.df[col].nunique(),
                    'mode': self.df[col].mode().iloc[0] if len(self.df[col].mode()) > 0 else None,
                    'mode_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'top_categories': value_counts.head(5).to_dict()
                }
                univariate_results[col] = stats

        self.eda_results['univariate'] = univariate_results
        print(f"Analyzed {len(univariate_results)} variables")

    def _bivariate_analysis(self):
        """Perform bivariate analysis"""
        print("\n🔗 Bivariate Analysis")

        bivariate_results = {}

        # Numerical vs target (if target exists)
        if self.target_column and self.target_column in self.df.columns:
            if self.df[self.target_column].dtype in ['int64', 'float64']:
                # Numerical target
                correlations = self.df.select_dtypes(include=[np.number]).corr()[self.target_column].drop(self.target_column)
                bivariate_results['target_correlations'] = correlations.to_dict()
            else:
                # Categorical target
                group_means = self.df.groupby(self.target_column).mean(numeric_only=True)
                bivariate_results['target_group_means'] = group_means.to_dict()

        # General correlations (numerical variables)
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            corr_matrix = self.df[numerical_cols].corr()
            # Get strongest correlations
            corr_pairs = []
            for i in range(len(numerical_cols)):
                for j in range(i+1, len(numerical_cols)):
                    corr_pairs.append((numerical_cols[i], numerical_cols[j], abs(corr_matrix.iloc[i, j])))

            corr_pairs.sort(key=lambda x: x[2], reverse=True)
            bivariate_results['strongest_correlations'] = corr_pairs[:10]  # Top 10

        self.eda_results['bivariate'] = bivariate_results
        print("Completed bivariate relationship analysis")

    def _multivariate_analysis(self):
        """Perform multivariate analysis"""
        print("\n🌐 Multivariate Analysis")

        multivariate_results = {}

        # PCA analysis
        try:
            pca_results = perform_pca_analysis(self.df)
            if pca_results:
                multivariate_results['pca'] = {
                    'n_components': pca_results['n_components'],
                    'explained_variance': pca_results['explained_variance_ratio'].tolist(),
                    'cumulative_variance': pca_results['cumulative_variance'].tolist()
                }
        except Exception as e:
            print(f"PCA analysis failed: {e}")

        # Correlation analysis
        try:
            corr_results = create_correlation_analysis(self.df, method='pearson')
            if corr_results:
                multivariate_results['correlation'] = {
                    'method': corr_results['method'],
                    'strongest_positive': corr_results.get('strongest_positive'),
                    'strongest_negative': corr_results.get('strongest_negative')
                }
        except Exception as e:
            print(f"Correlation analysis failed: {e}")

        self.eda_results['multivariate'] = multivariate_results
        print("Completed multivariate analysis")

    def _generate_summary_report(self, output_dir: str):
        """Generate comprehensive summary report"""
        report_path = os.path.join(output_dir, 'eda_summary_report.txt')

        with open(report_path, 'w') as f:
            f.write("EXPLORATORY DATA ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Dataset overview
            overview = self.eda_results.get('overview', {})
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Shape: {overview.get('shape', 'N/A')}\n")
            f.write(f"Columns: {len(overview.get('columns', []))}\n")
            f.write(f"Memory Usage: {overview.get('memory_usage', 0) / 1024 / 1024:.2f} MB\n")
            f.write(f"Duplicate Rows: {overview.get('duplicate_rows', 0)}\n")

            missing = overview.get('missing_summary', {})
            f.write(f"Total Missing Values: {missing.get('total_missing', 0)}\
