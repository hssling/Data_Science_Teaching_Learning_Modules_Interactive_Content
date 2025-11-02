# Module 5: Data Cleaning and Preprocessing

## Overview
Data cleaning and preprocessing are critical steps in the data science pipeline, often consuming 70-80% of a data scientist's time. This module provides comprehensive techniques for handling missing data, detecting and treating outliers, standardizing formats, and preparing data for analysis and modeling. You'll learn both automated and manual approaches to ensure data quality and reliability.

## Learning Objectives
By the end of this module, you will be able to:
- Identify and handle different types of missing data
- Detect and treat outliers using statistical and machine learning methods
- Standardize and normalize data for consistent analysis
- Handle categorical variables through encoding techniques
- Implement feature scaling and transformation methods
- Create automated data cleaning pipelines
- Validate data integrity and quality
- Handle imbalanced datasets and sampling techniques

## 1. Understanding Data Quality Issues

### 1.1 Types of Data Quality Problems

#### Missing Data
- **Completely Random Missing (MCAR)**: Missingness unrelated to any observed/unobserved data
- **Missing at Random (MAR)**: Missingness related to observed data but not the missing value itself
- **Missing Not at Random (MNAR)**: Missingness related to the unobserved missing value

#### Data Inconsistencies
- **Format inconsistencies**: Different date formats, phone number formats
- **Unit inconsistencies**: Mixing metric and imperial units
- **Categorical inconsistencies**: Typos, abbreviations, case variations

#### Invalid Data
- **Out-of-range values**: Ages of 200 years, negative prices
- **Impossible combinations**: Married single people, pregnant males
- **Data type mismatches**: Text in numeric fields

### 1.2 Data Quality Assessment Framework

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class DataQualityAssessor:
    """Comprehensive data quality assessment framework"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.quality_report = {}

    def assess_completeness(self) -> Dict[str, Dict]:
        """Assess data completeness across all columns"""
        completeness = {}

        for col in self.df.columns:
            total_count = len(self.df)
            non_null_count = self.df[col].notna().sum()
            null_count = self.df[col].isna().sum()

            completeness[col] = {
                'total_rows': total_count,
                'non_null_count': non_null_count,
                'null_count': null_count,
                'completeness_rate': non_null_count / total_count * 100,
                'null_percentage': null_count / total_count * 100
            }

        self.quality_report['completeness'] = completeness
        return completeness

    def assess_uniqueness(self) -> Dict[str, Dict]:
        """Assess uniqueness and duplicate data"""
        uniqueness = {}

        for col in self.df.columns:
            total_count = len(self.df)
            unique_count = self.df[col].nunique()
            duplicate_count = total_count - unique_count

            uniqueness[col] = {
                'total_values': total_count,
                'unique_values': unique_count,
                'duplicate_values': duplicate_count,
                'uniqueness_rate': unique_count / total_count * 100
            }

        # Check for duplicate rows
        duplicate_rows = self.df.duplicated().sum()
        uniqueness['duplicate_rows'] = {
            'count': duplicate_rows,
            'percentage': duplicate_rows / len(self.df) * 100
        }

        self.quality_report['uniqueness'] = uniqueness
        return uniqueness

    def assess_validity(self, validation_rules: Dict[str, callable] = None) -> Dict[str, Dict]:
        """Assess data validity against business rules"""
        validity = {}

        # Default validation rules
        default_rules = {
            'email': lambda x: pd.isna(x) or ('@' in str(x) and '.' in str(x)),
            'phone': lambda x: pd.isna(x) or (str(x).replace('-', '').replace('(', '').replace(')', '').replace(' ', '').isdigit() and len(str(x).replace('-', '').replace('(', '').replace(')', '').replace(' ', '')) >= 10),
            'age': lambda x: pd.isna(x) or (isinstance(x, (int, float)) and 0 <= x <= 120),
            'price': lambda x: pd.isna(x) or (isinstance(x, (int, float)) and x >= 0),
            'quantity': lambda x: pd.isna(x) or (isinstance(x, (int, float)) and x >= 0)
        }

        # Merge with custom rules
        if validation_rules:
            default_rules.update(validation_rules)

        for col in self.df.columns:
            if col in default_rules:
                rule_func = default_rules[col]
                valid_count = self.df[col].apply(rule_func).sum()
                total_count = len(self.df)

                validity[col] = {
                    'valid_count': valid_count,
                    'invalid_count': total_count - valid_count,
                    'validity_rate': valid_count / total_count * 100
                }

        self.quality_report['validity'] = validity
        return validity

    def assess_consistency(self) -> Dict[str, Dict]:
        """Assess data consistency and logical relationships"""
        consistency = {}

        # Check date consistency
        date_cols = [col for col in self.df.columns if 'date' in col.lower()]
        if len(date_cols) >= 2:
            try:
                date1 = pd.to_datetime(self.df[date_cols[0]], errors='coerce')
                date2 = pd.to_datetime(self.df[date_cols[1]], errors='coerce')

                valid_dates = date1.notna() & date2.notna()
                if valid_dates.sum() > 0:
                    logical_order = (date2 >= date1)[valid_dates]
                    consistency['date_order'] = {
                        'consistent_count': logical_order.sum(),
                        'inconsistent_count': len(logical_order) - logical_order.sum(),
                        'consistency_rate': logical_order.sum() / len(logical_order) * 100
                    }
            except:
                pass

        # Check numeric range consistency
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound))
            consistency[f'{col}_outliers'] = {
                'outlier_count': outliers.sum(),
                'non_outlier_count': len(self.df) - outliers.sum(),
                'outlier_percentage': outliers.sum() / len(self.df) * 100
            }

        self.quality_report['consistency'] = consistency
        return consistency

    def generate_quality_report(self) -> Dict:
        """Generate comprehensive data quality report"""
        self.assess_completeness()
        self.assess_uniqueness()
        self.assess_validity()
        self.assess_consistency()

        return self.quality_report

    def get_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)"""
        if not self.quality_report:
            self.generate_quality_report()

        scores = []

        # Completeness score (weighted by importance)
        completeness_scores = []
        for col_data in self.quality_report.get('completeness', {}).values():
            if isinstance(col_data, dict) and 'completeness_rate' in col_data:
                completeness_scores.append(col_data['completeness_rate'])

        if completeness_scores:
            scores.append(np.mean(completeness_scores) * 0.4)  # 40% weight

        # Validity score
        validity_scores = []
        for col_data in self.quality_report.get('validity', {}).values():
            if isinstance(col_data, dict) and 'validity_rate' in col_data:
                validity_scores.append(col_data['validity_rate'])

        if validity_scores:
            scores.append(np.mean(validity_scores) * 0.4)  # 40% weight

        # Consistency score
        consistency_scores = []
        for col_data in self.quality_report.get('consistency', {}).values():
            if isinstance(col_data, dict) and 'consistency_rate' in col_data:
                consistency_scores.append(col_data['consistency_rate'])

        if consistency_scores:
            scores.append(np.mean(consistency_scores) * 0.2)  # 20% weight

        return np.mean(scores) if scores else 0.0

# Usage example
# Create sample data with quality issues
np.random.seed(42)
data = {
    'customer_id': range(1, 1001),
    'name': ['Customer_' + str(i) for i in range(1, 1001)],
    'email': ['customer' + str(i) + '@example.com' for i in range(1, 1001)],
    'age': np.random.normal(35, 10, 1000),
    'price': np.random.uniform(10, 1000, 1000),
    'quantity': np.random.randint(1, 50, 1000),
    'signup_date': pd.date_range('2020-01-01', periods=1000, freq='D'),
    'last_purchase': pd.date_range('2023-01-01', periods=1000, freq='D')
}

df = pd.DataFrame(data)

# Introduce quality issues
df.loc[np.random.choice(df.index, 50, replace=False), 'email'] = None  # Missing emails
df.loc[np.random.choice(df.index, 30, replace=False), 'age'] = None    # Missing ages
df.loc[100, 'price'] = -500  # Negative price
df.loc[200, 'age'] = 150     # Impossible age
df.loc[300, 'email'] = 'invalid-email'  # Invalid email

# Assess data quality
assessor = DataQualityAssessor(df)
quality_report = assessor.generate_quality_report()
quality_score = assessor.get_quality_score()

print(f"Overall Data Quality Score: {quality_score:.2f}%")
print("\nCompleteness Report:")
for col, metrics in quality_report['completeness'].items():
    if isinstance(metrics, dict):
        print(f"{col}: {metrics['completeness_rate']:.1f}% complete")
```

## 2. Handling Missing Data

### 2.1 Understanding Missing Data Patterns

#### Visualizing Missing Data
```python
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_missing_data(df: pd.DataFrame):
    """Create comprehensive missing data visualizations"""

    # Missing data matrix
    plt.figure(figsize=(12, 8))
    msno.matrix(df, sparkline=False)
    plt.title('Missing Data Matrix', fontsize=16, fontweight='bold')
    plt.savefig('missing_data_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Missing data heatmap
    plt.figure(figsize=(10, 8))
    msno.heatmap(df)
    plt.title('Missing Data Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.savefig('missing_data_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Missing data bar chart
    plt.figure(figsize=(12, 6))
    msno.bar(df)
    plt.title('Missing Data by Column', fontsize=16, fontweight='bold')
    plt.savefig('missing_data_bar.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Missing data statistics
    missing_stats = df.isnull().sum()
    missing_percent = (missing_stats / len(df)) * 100

    print("Missing Data Summary:")
    print("=" * 50)
    for col in df.columns:
        if missing_stats[col] > 0:
            print(f"{col}: {missing_stats[col]} missing ({missing_percent[col]:.2f}%)")

# Usage
visualize_missing_data(df)
```

### 2.2 Missing Data Imputation Techniques

#### Statistical Imputation Methods
```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

class MissingDataImputer:
    """Comprehensive missing data imputation framework"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.imputation_methods = {}

    def impute_mean_median_mode(self, strategy: str = 'mean') -> pd.DataFrame:
        """Impute missing values using mean, median, or mode"""
        df_imputed = self.df.copy()

        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_imputed.select_dtypes(include=['object']).columns

        # Numeric columns
        if len(numeric_cols) > 0:
            if strategy in ['mean', 'median']:
                imputer = SimpleImputer(strategy=strategy)
                df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
                self.imputation_methods['numeric'] = f'{strategy}_imputation'

        # Categorical columns
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            df_imputed[categorical_cols] = imputer.fit_transform(df_imputed[categorical_cols])
            self.imputation_methods['categorical'] = 'mode_imputation'

        return df_imputed

    def impute_knn(self, n_neighbors: int = 5) -> pd.DataFrame:
        """Impute missing values using K-Nearest Neighbors"""
        df_imputed = self.df.copy()

        # KNN imputer works only with numeric data
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
            self.imputation_methods['knn'] = f'knn_imputation_k{n_neighbors}'

        return df_imputed

    def impute_iterative(self, estimator=None, max_iter: int = 10) -> pd.DataFrame:
        """Impute missing values using iterative imputation (MICE)"""
        df_imputed = self.df.copy()

        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            if estimator is None:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)

            imputer = IterativeImputer(
                estimator=estimator,
                max_iter=max_iter,
                random_state=42
            )
            df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
            self.imputation_methods['iterative'] = f'iterative_imputation_{max_iter}iter'

        return df_imputed

    def impute_forward_backward_fill(self, method: str = 'forward') -> pd.DataFrame:
        """Impute missing values using forward or backward fill"""
        df_imputed = self.df.copy()

        if method == 'forward':
            df_imputed = df_imputed.fillna(method='ffill')
            self.imputation_methods['temporal'] = 'forward_fill'
        elif method == 'backward':
            df_imputed = df_imputed.fillna(method='bfill')
            self.imputation_methods['temporal'] = 'backward_fill'

        return df_imputed

    def impute_interpolation(self, method: str = 'linear') -> pd.DataFrame:
        """Impute missing values using interpolation"""
        df_imputed = self.df.copy()

        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            df_imputed[col] = df_imputed[col].interpolate(method=method)

        self.imputation_methods['interpolation'] = f'{method}_interpolation'
        return df_imputed

    def compare_imputation_methods(self) -> Dict[str, pd.DataFrame]:
        """Compare different imputation methods"""
        methods = {}

        # Mean imputation
        methods['mean'] = self.impute_mean_median_mode('mean')

        # Median imputation
        methods['median'] = self.impute_mean_median_mode('median')

        # KNN imputation
        methods['knn'] = self.impute_knn(n_neighbors=5)

        # Iterative imputation
        methods['iterative'] = self.impute_iterative()

        return methods

# Usage example
imputer = MissingDataImputer(df)

# Compare different imputation methods
imputation_results = imputer.compare_imputation_methods()

print("Imputation Methods Comparison:")
print("=" * 40)
for method_name, imputed_df in imputation_results.items():
    remaining_missing = imputed_df.isnull().sum().sum()
    print(f"{method_name}: {remaining_missing} missing values remaining")

# Choose best method based on your analysis
final_df = imputation_results['knn']  # Example choice
```

## 3. Outlier Detection and Treatment

### 3.1 Statistical Outlier Detection Methods

#### Z-Score Method
```python
from scipy import stats

def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, List[int]]:
    """Detect outliers using Z-score method"""
    outliers = {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))

        # Find outliers
        outlier_indices = np.where(z_scores > threshold)[0].tolist()
        outliers[col] = outlier_indices

    return outliers

# Usage
zscore_outliers = detect_outliers_zscore(df, threshold=3.0)
print("Z-Score Outliers:")
for col, indices in zscore_outliers.items():
    print(f"{col}: {len(indices)} outliers")
```

#### IQR Method
```python
def detect_outliers_iqr(df: pd.DataFrame, multiplier: float = 1.5) -> Dict[str, List[int]]:
    """Detect outliers using IQR method"""
    outliers = {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        # Calculate Q1, Q3, IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        # Find outliers
        outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        outliers[col] = outlier_indices

    return outliers

# Usage
iqr_outliers = detect_outliers_iqr(df, multiplier=1.5)
print("IQR Outliers:")
for col, indices in iqr_outliers.items():
    print(f"{col}: {len(indices)} outliers")
```

#### Isolation Forest Method
```python
from sklearn.ensemble import IsolationForest

def detect_outliers_isolation_forest(df: pd.DataFrame, contamination: float = 0.1) -> Dict[str, List[int]]:
    """Detect outliers using Isolation Forest"""
    outliers = {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        # Prepare data (handle missing values)
        col_data = df[col].fillna(df[col].median()).values.reshape(-1, 1)

        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(col_data)

        # Find outliers (-1 indicates outlier)
        outlier_indices = df[outlier_labels == -1].index.tolist()
        outliers[col] = outlier_indices

    return outliers

# Usage
iso_outliers = detect_outliers_isolation_forest(df, contamination=0.1)
print("Isolation Forest Outliers:")
for col, indices in iso_outliers.items():
    print(f"{col}: {len(indices)} outliers")
```

### 3.2 Outlier Treatment Strategies

#### Capping/Winsorizing
```python
def cap_outliers(df: pd.DataFrame, method: str = 'iqr', multiplier: float = 1.5) -> pd.DataFrame:
    """Cap outliers using specified method"""
    df_capped = df.copy()

    numeric_cols = df_capped.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df_capped[col].quantile(0.25)
            Q3 = df_capped[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

        elif method == 'percentile':
            lower_bound = df_capped[col].quantile(0.05)  # 5th percentile
            upper_bound = df_capped[col].quantile(0.95)  # 95th percentile

        # Cap values
        df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)

    return df_capped

# Usage
df_capped = cap_outliers(df, method='iqr', multiplier=1.5)
print("Outliers capped using IQR method")
```

#### Outlier Removal
```python
def remove_outliers(df: pd.DataFrame, outlier_indices: Dict[str, List[int]]) -> pd.DataFrame:
    """Remove detected outliers from DataFrame"""
    # Combine all outlier indices
    all_outlier_indices = set()
    for indices in outlier_indices.values():
        all_outlier_indices.update(indices)

    # Remove outliers
    df_clean = df.drop(list(all_outlier_indices))

    print(f"Removed {len(all_outlier_indices)} outlier rows")
    print(f"Remaining rows: {len(df_clean)}")

    return df_clean

# Usage
all_outliers = detect_outliers_iqr(df)  # Or combine multiple methods
df_no_outliers = remove_outliers(df, all_outliers)
```

#### Transformation Methods
```python
def transform_outliers(df: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    """Transform data to handle outliers"""
    df_transformed = df.copy()

    numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if method == 'log':
            # Log transformation (add small constant to handle zeros)
            df_transformed[col] = np.log(df_transformed[col] - df_transformed[col].min() + 1)

        elif method == 'sqrt':
            # Square root transformation
            df_transformed[col] = np.sqrt(np.abs(df_transformed[col]))

        elif method == 'boxcox':
            # Box-Cox transformation
            from scipy.stats import boxcox
            transformed_data, _ = boxcox(df_transformed[col] - df_transformed[col].min() + 1)
            df_transformed[col] = transformed_data

    return df_transformed

# Usage
df_transformed = transform_outliers(df, method='log')
print("Applied log transformation to handle outliers")
```

## 4. Data Standardization and Normalization

### 4.1 Feature Scaling Techniques

#### StandardScaler (Z-score normalization)
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def standardize_features(df: pd.DataFrame, columns: List[str] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardize features using Z-score normalization"""
    df_scaled = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])

    return df_scaled, scaler

# Usage
df_standardized, scaler = standardize_features(df)
print("Features standardized (mean=0, std=1)")
print(f"Feature means after scaling: {scaler.mean_}")
print(f"Feature stds after scaling: {scaler.scale_}")
```

#### MinMaxScaler (Normalization)
```python
def normalize_features(df: pd.DataFrame, columns: List[str] = None,
                      feature_range: Tuple = (0, 1)) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Normalize features to specified range"""
    df_normalized = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    scaler = MinMaxScaler(feature_range=feature_range)
    df_normalized[columns] = scaler.fit_transform(df_normalized[columns])

    return df_normalized, scaler

# Usage
df_normalized, minmax_scaler = normalize_features(df, feature_range=(0, 1))
print("Features normalized to [0, 1] range")
```

#### RobustScaler (Robust to outliers)
```python
def robust_scale_features(df: pd.DataFrame, columns: List[str] = None) -> Tuple[pd.DataFrame, RobustScaler]:
    """Scale features using robust statistics (median and IQR)"""
    df_robust = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    scaler = RobustScaler()
    df_robust[columns] = scaler.fit_transform(df_robust[columns])

    return df_robust, scaler

# Usage
df_robust, robust_scaler = robust_scale_features(df)
print("Features scaled using robust statistics (median and IQR)")
```

### 4.2 Comparing Scaling Methods
```python
def compare_scaling_methods(df: pd.DataFrame, columns: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Compare different scaling methods"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    methods = {}

    # Standard scaling
    methods['standard'], _ = standardize_features(df, columns)

    # Min-Max scaling
    methods['minmax'], _ = normalize_features(df, columns)

    # Robust scaling
    methods['robust'], _ = robust_scale_features(df, columns)

    return methods

# Usage
scaling_comparison = compare_scaling_methods(df)

print("Scaling Methods Comparison:")
print("=" * 40)
for method_name, scaled_df in scaling_comparison.items():
    print(f"\n{method_name.upper()} Scaling:")
    for col in scaled_df.select_dtypes(include=[np.number]).columns[:3]:  # First 3 columns
        print(".3f")
```

## 5. Categorical Variable Encoding

### 5.1 Label Encoding
```python
from sklearn.preprocessing import LabelEncoder

def label_encode_categorical(df: pd.DataFrame, columns: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """Apply label encoding to categorical columns"""
    df_encoded = df.copy()
    encoders = {}

    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()

    for col in columns:
        encoder = LabelEncoder()
        df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
        encoders[col] = encoder

    return df_encoded, encoders

# Usage
df_label_encoded, label_encoders = label_encode_categorical(df)
print("Applied label encoding to categorical variables")
```

### 5.2 One-Hot Encoding
```python
from sklearn.preprocessing import OneHotEncoder

def one_hot_encode_categorical(df: pd.DataFrame, columns: List[str] = None,
                              drop_first: bool = False) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """Apply one-hot encoding to categorical columns"""
    df_encoded = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()

    # Create one-hot encoder
    encoder = OneHotEncoder(sparse=False, drop='first' if drop_first else None)
    encoded_data = encoder.fit_transform(df_encoded[columns])

    # Create column names
    feature_names = encoder.get_feature_names_out(columns)

    # Create DataFrame with encoded data
    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)

    # Drop original columns and add encoded columns
    df_encoded = df_encoded.drop(columns, axis=1)
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

    return df_encoded, encoder

# Usage
df_onehot_encoded, onehot_encoder = one_hot_encode_categorical(df, drop_first=True)
print("Applied one-hot encoding to categorical variables")
print(f"Original shape: {df.shape}")
print(f"Encoded shape: {df_onehot_encoded.shape}")
```

### 5.3 Ordinal Encoding
```python
from sklearn.preprocessing import OrdinalEncoder

def ordinal_encode_categorical(df: pd.DataFrame, columns: List[str] = None,
                              categories: Dict[str, List] = None) -> Tuple[pd.DataFrame, OrdinalEncoder]:
    """Apply ordinal encoding with custom category orders"""
    df_encoded = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()

    # Create ordinal encoder
    encoder = OrdinalEncoder(categories=[categories[col] for col in columns] if categories else 'auto')
    df_encoded[columns] = encoder.fit_transform(df_encoded[columns])

    return df_encoded, encoder

# Usage with custom order
custom_categories = {
    'education': ['High School', 'Bachelor', 'Master', 'PhD'],
    'experience': ['Entry', 'Mid', 'Senior', 'Expert']
}

df_ordinal_encoded, ordinal_encoder = ordinal_encode_categorical(df, categories=custom_categories)
print("Applied ordinal encoding with custom category orders")
```

### 5.4 Target Encoding
```python
def target_encode_categorical(df: pd.DataFrame, categorical_cols: List[str],
                            target_col: str, smoothing: float = 1.0) -> pd.DataFrame:
    """Apply target encoding to categorical variables"""
    df_encoded = df.copy()

    for col in categorical_cols:
        # Calculate mean target value for each category
        category_means = df.groupby(col)[target_col].mean()
        global_mean = df[target_col].mean()

        # Apply smoothing
        category_counts = df[col].value_counts()
        smoothed_means = (category_counts * category_means + smoothing * global_mean) / (category_counts + smoothing)

        # Map encoded values
        df_encoded[col] = df_encoded[col].map(smoothed_means)

    return df_encoded

# Usage (assuming we have a target column)
if 'target' in df.columns:
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_target_encoded = target_encode_categorical(df, categorical_cols, 'target')
    print("Applied target encoding to categorical variables")
```

## 6. Feature Engineering

### 6.1 Creating New Features
```python
def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from existing data"""
    df_featured = df.copy()

    # Date-based features
    if 'date' in df_featured.columns:
        df_featured['date'] = pd.to_datetime(df_featured['date'])
        df_featured['year'] = df_featured['date'].dt.year
        df_featured['month'] = df_featured['date'].dt.month
        df_featured['day'] = df_featured['date'].dt.day
        df_featured['day_of_week'] = df_featured['date'].dt.dayofweek
        df_featured['quarter'] = df_featured['date'].dt.quarter
        df_featured['is_weekend'] = df_featured['date'].dt.dayofweek.isin([5, 6]).astype(int)

    # Numeric feature interactions
    numeric_cols = df_featured.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        # Ratios and products
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                # Avoid division by zero
                df_featured[f'{col1}_div_{col2}'] = df_featured[col1] / (df_featured[col2] + 1e-6)
                df_featured[f'{col1}_times_{col2}'] = df_featured[col1] * df_featured[col2]

    # Categorical feature combinations
    categorical_cols = df_featured.select_dtypes(include=['object']).columns
    if len(categorical_cols) >= 2:
        for i in range(len(categorical_cols)):
            for j in range(i+1, len(categorical_cols)):
                col1, col2 = categorical_cols[i], categorical_cols[j]
                df_featured[f'{col1}_{col2}_combined'] = df_featured[col1] + '_' + df_featured[col2]

    # Statistical features
    if len(numeric_cols) > 0:
        # Rolling statistics (if time series)
        if 'date' in df_featured.columns:
            df_featured = df_featured.sort_values('date')
            for col in numeric_cols:
                df_featured[f'{col}_rolling_mean_7'] = df_featured[col].rolling(window=7).mean()
                df_featured[f'{col}_rolling_std_7'] = df_featured[col].rolling(window=7).std()

        # Group-based statistics
        for col in numeric_cols:
            df_featured[f'{col}_zscore'] = (df_featured[col] - df_featured[col].mean()) / df_featured[col].std()

    return df_featured

# Usage
df_with_features = create_derived_features(df)
print(f"Created additional features. New shape: {df_with_features.shape}")
```

### 6.2 Feature Selection Techniques
```python
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

def select_features_correlation(df: pd.DataFrame, target_col: str, k: int = 10) -> List[str]:
    """Select features based on correlation with target"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop(target_col) if target_col in numeric_cols else numeric_cols

    correlations = df[numeric_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)

    return correlations.head(k).index.tolist()

def select_features_univariate(df: pd.DataFrame, target_col: str, k: int = 10,
                              method: str = 'f_regression') -> List[str]:
    """Select features using univariate statistical tests"""
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Handle categorical variables
    X_numeric = X.select_dtypes(include=[np.number])

    if method == 'f_regression':
        selector = SelectKBest(score_func=f_regression, k=k)
    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k=k)

    selector.fit(X_numeric, y)

    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = X_numeric.columns[selected_indices].tolist()

    return selected_features

def select_features_importance(df: pd.DataFrame, target_col: str, k: int = 10) -> List[str]:
    """Select features based on model importance"""
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Handle categorical variables (simple approach)
    X_processed = pd.get_dummies(X, drop_first=True)

    # Train random forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_processed, y)

    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_processed.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    return feature_importance.head(k)['feature'].tolist()

# Usage
if 'target' in df.columns:
    # Correlation-based selection
    corr_features = select_features_correlation(df, 'target', k=5)
    print(f"Top 5 features by correlation: {corr_features}")
