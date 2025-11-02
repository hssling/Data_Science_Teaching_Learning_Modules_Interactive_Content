# Module 13: Projects and Case Studies

## Overview
This module provides hands-on projects and real-world case studies that demonstrate the application of data science concepts across various industries. You'll work on comprehensive projects that integrate multiple skills learned throughout the curriculum, from data collection to model deployment. Each project includes detailed requirements, implementation guidance, and evaluation criteria.

## Learning Objectives
By the end of this module, you will be able to:
- Apply data science methodologies to real-world problems
- Design and implement end-to-end data science solutions
- Work with diverse datasets and business domains
- Present findings and recommendations to stakeholders
- Evaluate project success and iterate on solutions
- Understand industry-specific data science applications

## 1. Project 1: Customer Churn Prediction

### 1.1 Business Problem
A telecommunications company wants to predict which customers are likely to churn (cancel their service) so they can take proactive retention actions. The goal is to identify high-risk customers and develop targeted retention strategies.

### 1.2 Dataset Description
- **Source**: Telco Customer Churn dataset (Kaggle)
- **Size**: ~7,000 customers, 21 features
- **Target Variable**: Churn (Yes/No)
- **Features**: Demographics, service usage, billing information, customer satisfaction

### 1.3 Project Requirements

#### Phase 1: Data Understanding and Preparation
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load dataset
df = pd.read_csv('telco_customer_churn.csv')

# Initial data exploration
print("Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Data types and basic statistics
print("\nData Types:")
print(df.dtypes)

print("\nTarget Distribution:")
print(df['Churn'].value_counts(normalize=True))

# Visualize churn distribution
plt.figure(figsize=(8, 6))
df['Churn'].value_counts().plot(kind='bar')
plt.title('Customer Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.savefig('churn_distribution.png')
plt.show()
```

#### Phase 2: Exploratory Data Analysis
```python
# Demographic analysis
demographic_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, col in enumerate(demographic_cols):
    row, col_idx = i // 2, i % 2
    churn_by_demo = df.groupby([col, 'Churn']).size().unstack()
    churn_by_demo.plot(kind='bar', stacked=True, ax=axes[row, col_idx])
    axes[row, col_idx].set_title(f'Churn by {col}')
    axes[row, col_idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('demographic_analysis.png')
plt.show()

# Service usage analysis
service_cols = ['PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']

# Calculate churn rates by service
churn_rates = {}
for col in service_cols:
    churn_rate = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean())
    churn_rates[col] = churn_rate

# Visualize service churn rates
service_churn_df = pd.DataFrame(churn_rates)
service_churn_df.plot(kind='bar', figsize=(12, 6))
plt.title('Churn Rates by Service Type')
plt.ylabel('Churn Rate')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('service_churn_analysis.png')
plt.show()

# Contract and billing analysis
contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean())
payment_churn = df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').mean())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

contract_churn.plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Churn Rate by Contract Type')
ax1.set_ylabel('Churn Rate')
ax1.tick_params(axis='x', rotation=45)

payment_churn.plot(kind='bar', ax=ax2, color='lightcoral')
ax2.set_title('Churn Rate by Payment Method')
ax2.set_ylabel('Churn Rate')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('contract_payment_analysis.png')
plt.show()
```

#### Phase 3: Feature Engineering
```python
# Data preprocessing function
def preprocess_churn_data(df):
    """Preprocess the telco churn dataset"""

    # Make a copy
    df_processed = df.copy()

    # Handle TotalCharges (convert to numeric, handle missing)
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median(), inplace=True)

    # Convert SeniorCitizen to string
    df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].astype(str)

    # Convert Churn to binary
    df_processed['Churn'] = (df_processed['Churn'] == 'Yes').astype(int)

    # Feature engineering
    # 1. Tenure groups
    df_processed['tenure_group'] = pd.cut(df_processed['tenure'],
                                        bins=[0, 12, 24, 36, 48, 60, 72],
                                        labels=['0-1yr', '1-2yr', '2-3yr', '3-4yr', '4-5yr', '5-6yr'])

    # 2. Monthly charges groups
    df_processed['monthly_charges_group'] = pd.cut(df_processed['MonthlyCharges'],
                                                 bins=[0, 30, 50, 70, 90, 120],
                                                 labels=['0-30', '30-50', '50-70', '70-90', '90+'])

    # 3. Service count (number of services)
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies']

    # Convert 'Yes'/'No' to 1/0 for service columns
    for col in service_cols:
        if col in df_processed.columns:
            df_processed[col] = (df_processed[col] == 'Yes').astype(int)

    df_processed['total_services'] = df_processed[service_cols].sum(axis=1)

    # 4. Customer value score
    df_processed['customer_value_score'] = (
        df_processed['tenure'] * 0.3 +
        df_processed['MonthlyCharges'] * 0.4 +
        df_processed['TotalCharges'] * 0.3
    )

    # Encode categorical variables
    categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                       'PhoneService', 'MultipleLines', 'InternetService',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaperlessBilling', 'PaymentMethod',
                       'tenure_group', 'monthly_charges_group']

    # Label encoding for binary categories
    binary_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PaperlessBilling']
    for col in binary_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])

    # One-hot encoding for multi-class categories
    multi_class_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies', 'Contract',
                       'PaymentMethod', 'tenure_group', 'monthly_charges_group']

    df_encoded = pd.get_dummies(df_processed, columns=multi_class_cols, drop_first=True)

    # Drop customer ID
    if 'customerID' in df_encoded.columns:
        df_encoded = df_encoded.drop('customerID', axis=1)

    return df_encoded

# Apply preprocessing
df_processed = preprocess_churn_data(df)
print(f"Processed dataset shape: {df_processed.shape}")
print(f"Feature columns: {len(df_processed.columns) - 1}")  # Excluding target
```

#### Phase 4: Model Development
```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

# Split features and target
X = df_processed.drop('Churn', axis=1)
y = df_processed['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training function
def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Train and evaluate a model"""

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }

    print(f"\n{model_name} Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    return model, metrics

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
}

model_results = {}
trained_models = {}

for model_name, model in models.items():
    trained_model, metrics = train_and_evaluate_model(
        model, model_name, X_train_scaled, X_test_scaled, y_train, y_test
    )
    model_results[model_name] = metrics
    trained_models[model_name] = trained_model

# Compare model performance
results_df = pd.DataFrame(model_results).T
print("\nModel Comparison:")
print(results_df.round(4))

# Visualize model comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
for i, metric in enumerate(metrics_to_plot):
    row, col = i // 2, i % 2
    results_df[metric].plot(kind='bar', ax=axes[row, col])
    axes[row, col].set_title(f'{metric.capitalize()} Comparison')
    axes[row, col].tick_params(axis='x', rotation=45)
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()
```

#### Phase 5: Model Interpretation and Business Insights
```python
# Feature importance analysis
def analyze_feature_importance(model, model_name, feature_names):
    """Analyze feature importance for tree-based models"""

    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Plot top 20 features
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'{model_name} - Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_feature_importance.png')
        plt.show()

        return importance_df
    else:
        print(f"{model_name} does not support feature importance analysis")
        return None

# Analyze feature importance for best model
best_model_name = results_df['f1_score'].idxmax()
best_model = trained_models[best_model_name]

feature_importance = analyze_feature_importance(best_model, best_model_name, X.columns)

# Business insights
def generate_business_insights(model_results, feature_importance):
    """Generate actionable business insights"""

    insights = []

    # Model performance insights
    best_model = model_results.loc[model_results['f1_score'].idxmax()]
    insights.append(f"Best performing model achieves {best_model['f1_score']:.1f} F1-score")
    insights.append(f"Model can identify {best_model['recall']:.1f} of customers who will churn")

    # Feature importance insights
    if feature_importance is not None:
        top_features = feature_importance.head(5)['feature'].tolist()

        feature_interpretations = {
            'Contract_Month-to-month': "Month-to-month contracts have highest churn risk",
            'tenure': "Newer customers are more likely to churn",
            'MonthlyCharges': "Higher monthly charges correlate with churn",
            'total_services': "Customers with fewer services are more likely to churn",
            'InternetService_Fiber optic': "Fiber optic customers show higher churn rates"
        }

        for feature in top_features:
            if feature in feature_interpretations:
                insights.append(feature_interpretations[feature])

    # Retention strategy recommendations
    insights.extend([
        "Implement targeted retention campaigns for month-to-month customers",
        "Offer incentives for customers in first 12 months",
        "Consider loyalty programs for high-value customers",
        "Monitor customers with multiple service complaints",
        "Develop personalized retention offers based on churn probability"
    ])

    return insights

# Generate and display insights
business_insights = generate_business_insights(results_df, feature_importance)

print("\nBusiness Insights and Recommendations:")
print("=" * 50)
for i, insight in enumerate(business_insights, 1):
    print(f"{i}. {insight}")
```

### 1.4 Project Evaluation Criteria

#### Technical Evaluation
- **Data preprocessing quality**: 20%
- **Feature engineering creativity**: 15%
- **Model selection and tuning**: 20%
- **Model performance metrics**: 15%
- **Code quality and documentation**: 15%
- **Visualization quality**: 15%

#### Business Evaluation
- **Problem understanding**: 20%
- **Actionable insights**: 30%
- **Business recommendations**: 25%
- **Presentation clarity**: 15%
- **Impact assessment**: 10%

## 2. Project 2: Fraud Detection System

### 2.1 Business Problem
A financial institution needs to detect fraudulent credit card transactions in real-time to prevent financial losses and protect customers. The system must balance fraud detection accuracy with minimizing false positives that inconvenience legitimate customers.

### 2.2 Dataset Description
- **Source**: Credit Card Fraud Detection dataset (Kaggle)
- **Size**: 284,807 transactions, 31 features
- **Target Variable**: Class (0 = Normal, 1 = Fraud)
- **Features**: Time, Amount, V1-V28 (PCA-transformed features)
- **Challenge**: Highly imbalanced dataset (0.172% fraud rate)

### 2.3 Implementation Approach

#### Handling Class Imbalance
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter

# Analyze class distribution
print("Class Distribution:")
print(df['Class'].value_counts())
print(f"Fraud rate: {df['Class'].value_counts()[1] / len(df) * 100:.4f}%")

# Split features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define resampling strategies
over = SMOTE(sampling_strategy=0.1)  # Oversample minority to 10% of majority
under = RandomUnderSampler(sampling_strategy=0.5)  # Undersample majority to 2:1 ratio

# Create pipeline
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

# Apply resampling
X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)

print("Original training set:")
print(f"Normal: {Counter(y_train)[0]}, Fraud: {Counter(y_train)[1]}")

print("Resampled training set:")
print(f"Normal: {Counter(y_resampled)[0]}, Fraud: {Counter(y_resampled)[1]}")
```

#### Anomaly Detection Models
```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Prepare data for anomaly detection
X_normal = X_train[y_train == 0]  # Only normal transactions for training
X_test_scaled = StandardScaler().fit_transform(X_test)

# Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.001, random_state=42)
iso_forest.fit(X_normal)

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.001, novelty=True)
lof.fit(X_normal)

# One-Class SVM
oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.001)
oc_svm.fit(X_normal)

# Evaluate anomaly detection models
models = {
    'Isolation Forest': iso_forest,
    'Local Outlier Factor': lof,
    'One-Class SVM': oc_svm
}

for model_name, model in models.items():
    # Get anomaly scores
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X_test_scaled)
    else:
        scores = model.score_samples(X_test_scaled)

    # Convert to binary predictions (anomaly = fraud)
    threshold = np.percentile(scores, 1)  # Top 1% as anomalies
    predictions = (scores < threshold).astype(int)

    # Calculate metrics
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"\n{model_name}:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
```

#### Real-time Scoring System
```python
import joblib
from datetime import datetime

class FraudDetectionSystem:
    """Real-time fraud detection system"""

    def __init__(self, model, scaler, threshold: float = 0.5):
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.feature_names = None

    def preprocess_transaction(self, transaction_data: dict) -> np.ndarray:
        """Preprocess transaction data for prediction"""

        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])

        # Handle missing features
        if self.feature_names is not None:
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0  # Default value

        # Select and order features
        if self.feature_names is not None:
            df = df[self.feature_names]

        # Scale features
        features_scaled = self.scaler.transform(df)

        return features_scaled

    def predict_fraud(self, transaction_data: dict) -> dict:
        """Predict fraud probability for a transaction"""

        # Preprocess
        features = self.preprocess_transaction(transaction_data)

        # Get prediction probability
        fraud_probability = self.model.predict_proba(features)[0, 1]

        # Make binary decision
        is_fraud = fraud_probability >= self.threshold

        # Risk assessment
        if fraud_probability < 0.1:
            risk_level = "Low"
        elif fraud_probability < 0.5:
            risk_level = "Medium"
        else:
            risk_level = "High"

        result = {
            'transaction_id': transaction_data.get('transaction_id', 'unknown'),
            'fraud_probability': float(fraud_probability),
            'is_fraud': bool(is_fraud),
            'risk_level': risk_level,
            'timestamp': datetime.now().isoformat(),
            'recommendation': self._get_recommendation(is_fraud, fraud_probability)
        }

        return result

    def _get_recommendation(self, is_fraud: bool, probability: float) -> str:
        """Generate recommendation based on prediction"""

        if is_fraud:
            if probability > 0.8:
                return "Block transaction immediately and investigate account"
            else:
                return "Flag for manual review and additional verification"
        else:
            if probability < 0.1:
                return "Approve transaction"
            else:
                return "Additional verification may be required"

    def save_system(self, filepath: str):
        """Save the fraud detection system"""

        system_data = {
            'model': self.model,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'feature_names': self.feature_names
        }

        joblib.dump(system_data, filepath)
        print(f"Fraud detection system saved to {filepath}")

    @classmethod
    def load_system(cls, filepath: str):
        """Load a saved fraud detection system"""

        system_data = joblib.load(filepath)

        system = cls(
            model=system_data['model'],
            scaler=system_data['scaler'],
            threshold=system_data['threshold']
        )
        system.feature_names = system_data['feature_names']

        return system

# Example usage
# Initialize system
fraud_system = FraudDetectionSystem(best_model, scaler, threshold=0.3)
fraud_system.feature_names = X_train.columns.tolist()

# Test transaction
test_transaction = {
    'transaction_id': 'txn_12345',
    'Time': 100000,
    'Amount': 500.0,
    'V1': -1.5, 'V2': 0.5, 'V3': 1.2,  # ... other V features
}

result = fraud_system.predict_fraud(test_transaction)
print("Fraud Detection Result:")
for key, value in result.items():
    print(f"{key}: {value}")

# Save system
fraud_system.save_system('fraud_detection_system.pkl')
```

## 3. Project 3: Recommendation Engine

### 3.1 Business Problem
An e-commerce platform wants to build a personalized product recommendation system to increase customer engagement, conversion rates, and average order value. The system should provide relevant product suggestions based on user behavior and preferences.

### 3.2 Dataset Description
- **Source**: Online Retail dataset (UCI Machine Learning Repository)
- **Size**: ~541,909 transactions, 8 features
- **Features**: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

### 3.3 Collaborative Filtering Implementation

#### User-Item Matrix Construction
```python
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data
df = pd.read_csv('online_retail.csv')

# Data cleaning
df = df.dropna(subset=['CustomerID', 'Description'])
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

# Create customer-product matrix
customer_product_matrix = df.pivot_table(
    index='CustomerID',
    columns='StockCode',
    values='Quantity',
    aggfunc='sum',
    fill_value=0
)

# Convert to sparse matrix for efficiency
sparse_matrix = csr_matrix(customer_product_matrix.values)

print(f"Customer-Product Matrix Shape: {customer_product_matrix.shape}")
print(f"Sparsity: {(sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1])) * 100:.2f}%")
```

#### User-Based Collaborative Filtering
```python
def user_based_recommendations(customer_id, customer_product_matrix, n_recommendations=5):
    """Generate user-based collaborative filtering recommendations"""

    # Calculate user similarity
    user_similarity = cosine_similarity(customer_product_matrix)

    # Get target user index
    try:
        user_idx = customer_product_matrix.index.get_loc(customer_id)
    except KeyError:
        return []

    # Get similarity scores for target user
    user_sim_scores = user_similarity[user_idx]

    # Find most similar users (excluding self)
    similar_users_indices = user_sim_scores.argsort()[::-1][1:]

    # Get products purchased by target user
    target_user_products = set(customer_product_matrix.loc[customer_id][customer_product_matrix.loc[customer_id] > 0].index)

    # Collect recommendations from similar users
    recommendations = {}
    for similar_user_idx in similar_users_indices[:10]:  # Top 10 similar users
        similar_user_id = customer_product_matrix.index[similar_user_idx]
        similarity_score = user_sim_scores[similar_user_idx]

        # Get products purchased by similar user
        similar_user_products = customer_product_matrix.loc[similar_user_id]
        new_products = similar_user_products[similar_user_products > 0].index.difference(target_user_products)

        # Add to recommendations with weighted score
        for product in new_products:
            if product not in recommendations:
                recommendations[product] = 0
            recommendations[product] += similarity_score

    # Sort and return top recommendations
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return sorted_recommendations[:n_recommendations]

# Example usage
customer_id = 12350  # Example customer
recommendations = user_based_recommendations(customer_id, customer_product_matrix)

print(f"Recommendations for Customer {customer_id}:")
for product_code, score in recommendations:
    product_name = df[df['StockCode'] == product_code]['Description'].iloc[0]
    print(f"  {product_code}: {product_name} (Score: {score:.3f})")
```

#### Item-Based Collaborative Filtering
```python
def item_based_recommendations(product_code, customer_product_matrix, n_recommendations=5):
    """Generate item-based collaborative filtering recommendations"""

    # Calculate item similarity
    item_similarity = cosine_similarity(customer_product_matrix.T)

    # Get target product index
    try:
        product_idx = customer_product_matrix.columns.get_loc(product_code)
    except KeyError:
        return []

    # Get similarity scores for target product
    product_sim_scores = item_similarity[product_idx]

    # Find most similar products (excluding self)
    similar_products_indices = product_sim_scores.argsort()[::-1][1:]

    # Get top similar products with scores
    recommendations = []
    for idx in similar_products_indices[:n_recommendations]:
        similar_product_code = customer_product_matrix.columns[idx]
        similarity_score = product_sim_scores[idx]

        recommendations.append((similar_product_code, similarity_score))

    return recommendations

# Example usage
product_code = '85123A'  # Example product
recommendations = item_based_recommendations(product_code, customer_product_matrix)

product_name = df[df['StockCode'] == product_code]['Description'].iloc[0]
print(f"Products similar to {product_code} ({product_name}):")
for similar_product, score in recommendations:
    similar_name = df[df['StockCode'] == similar_product]['Description'].iloc[0]
    print(f"  {similar_product}: {similar_name} (Similarity: {score:.3f})")
```

#### Matrix Factorization with SVD
```python
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

def matrix_factorization_recommendations(customer_product_matrix, n_factors=50, n_recommendations=5):
    """Generate recommendations using matrix factorization"""

    # Normalize the matrix (center by user mean)
    user_means = customer_product_matrix.mean(axis=1)
    normalized_matrix = customer_product_matrix.sub(user_means, axis=0)

    # Apply SVD
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    user_factors = svd.fit_transform(normalized_matrix)
    item_factors = svd.components_.T

    # Reconstruct the matrix
    reconstructed_matrix = user_factors @ item_factors.T

    # Add back user means
    reconstructed_matrix = reconstructed_matrix.add(user_means, axis=0)

    recommendations = {}

    for customer_id in customer_product_matrix.index:
        # Get customer's row from reconstructed matrix
        customer_idx = customer_product_matrix.index.get_loc(customer_id)
        customer_predictions = reconstructed_matrix[customer_idx]

        # Get products customer hasn't purchased
        customer_purchases = customer_product_matrix.loc[customer_id]
        unpurchased_products = customer_purchases[customer_purchases == 0].index

        # Get predicted ratings for unpurchased products
        predicted_ratings = customer_predictions[unpurchased_products]

        # Get top recommendations
        top_product_indices = predicted_ratings.argsort()[::-1][:n_recommendations]
        top_products = unpurchased_products[top_product_indices]
        top_scores = predicted_ratings[top_product_indices]

        recommendations[customer_id] = list(zip(top_products, top_scores))

    return recommendations

# Generate recommendations for all customers
mf_recommendations = matrix_factorization_recommendations(customer_product_matrix)

# Example for one customer
customer_id = 12350
if customer_id in mf_recommendations:
    print(f"Matrix Factorization recommendations for Customer {customer_id}:")
    for product_code, score in mf_recommendations[customer_id]:
        product_name = df[df['StockCode'] == product_code]['Description'].iloc[0]
        print(f"  {product_code}: {product_name} (Predicted Rating: {score:.3f})")
```

## 4. Project 4: Time Series Forecasting

### 4.1 Business Problem
A retail company needs accurate demand forecasting for inventory management and supply chain optimization. The system should predict product demand for the next 3-6 months to minimize stockouts and overstock situations.

### 4.2 Dataset Description
- **Source**: Store Item Demand Forecasting Challenge (Kaggle)
- **Size**: 10 stores × 50 items × daily sales for 5 years
- **Target Variable**: Sales quantity per item per store per day
- **Features**: Date, Store, Item, Sales

### 4.3 Time Series Analysis and Forecasting

#### Time Series Decomposition
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load time series data
df = pd.read_csv('train.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Aggregate sales by date for overall analysis
daily_sales = df.groupby('date')['sales'].sum()

# Time series decomposition
decomposition = seasonal_decompose(daily_sales, model='additive', period=365)

fig, axes = plt.subplots(4, 1, figsize=(15, 12))

decomposition.observed.plot(ax=axes[0])
axes[0].set_title('Observed')

decomposition.trend.plot(ax=axes[1])
axes[1].set_title('Trend')

decomposition.seasonal.plot(ax=axes[2])
axes[2].set_title('Seasonal')

decomposition.resid.plot(ax=axes[3])
axes[3].set_title('Residual')

plt.tight_layout()
plt.savefig('time_series_decomposition.png')
plt.show()

# Stationarity test
def test_stationarity(timeseries):
    """Test for stationarity using Augmented Dickey-Fuller test"""

    # Rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()

    # Plot rolling statistics
    plt.figure(figsize=(12, 6))
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolling_mean, color='red', label='Rolling Mean')
    plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.savefig('stationarity_test.png')
    plt.show()

    # Augmented Dickey-Fuller test
    adf_test = adfuller(timeseries, autolag='AIC')

    print('Augmented Dickey-Fuller Test Results:')
    print(f'ADF Statistic: {adf_test[0]:.4f}')
    print(f'p-value: {adf_test[1]:.4f}')
    print(f'Critical Values:')
    for key, value in adf_test[4].items():
        print(f'  {key}: {value:.4f}')

    if adf_test[1] < 0.05:
        print("Series is stationary (reject null hypothesis)")
    else:
        print("Series is non-stationary (fail to reject null hypothesis)")

    return adf_test[1] < 0.05

# Test stationarity
is_stationary = test_stationarity(daily_sales)
```

#### ARIMA/SARIMA Modeling
```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Prepare data for modeling
train_size = int(len(daily_sales) * 0.8)
train_data = daily_sales[:train_size]
test_data = daily_sales[train_size:]

# ARIMA model
def fit_arima_model(train_data, order=(5,1,0)):
    """Fit ARIMA model"""

    model = ARIMA(train_data, order=order)
    model_fit = model.fit()

    return model_fit

# SARIMA model (with seasonality)
def fit_sarima_model(train_data, order=(1,1,1), seasonal_order=(1,1,1,7)):
    """Fit SARIMA model"""

    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    return model_fit

# Fit models
arima_model = fit_arima_model(train_data)
sarima_model = fit_sarima_model(train_data)

# Make predictions
arima_predictions = arima_model.forecast(steps=len(test_data))
sarima_predictions = sarima_model.forecast(steps=len(test_data))

# Evaluate models
def evaluate_forecasts(actual, predicted, model_name):
    """Evaluate forecasting model performance"""

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)

    print(f"\n{model_name} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")

    return {'mae': mae, 'mse': mse, 'rmse': rmse}

arima_metrics = evaluate_forecasts(test_data, arima_predictions, "ARIMA")
sarima_metrics = evaluate_forecasts(test_data, sarima_predictions, "SARIMA")

# Visualize predictions
plt.figure(figsize=(15, 8))

plt.plot(train_data.index[-30:], train_data.values[-30:], label='Training Data', color='blue')
plt.plot(test_data.index, test_data.values, label='Actual Test Data', color='green')
plt.plot(test_data.index, arima_predictions, label='ARIMA Predictions', color='red', linestyle='--')
plt.plot(test_data.index, sarima_predictions, label='SARIMA Predictions', color='orange', linestyle='--')

plt.title('Time Series Forecasting: ARIMA vs SARIMA')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('forecasting_comparison.png')
plt.show()
```

#### Machine Learning for Time Series
```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Feature engineering for time series
def create_time_series_features(df, target_column='sales', lags=7):
    """Create features for time series forecasting"""

    df_featured = df.copy()

    # Lag features
    for lag in range(1, lags + 1):
        df_featured[f'sales_lag_{lag}'] = df_featured[target_column].shift(lag)

    # Rolling statistics
    df_featured['rolling_mean_7'] = df_featured[target_column].rolling(window=7).mean()
    df_featured['rolling_std_7'] = df_featured[target_column].rolling(window=7).std()
    df_featured['rolling_mean_30'] = df_featured[target_column].rolling(window=30).mean()

    # Date features
    df_featured['day_of_week'] = df_featured.index.dayofweek
    df_featured['month'] = df_featured.index.month
    df_featured['quarter'] = df_featured.index.quarter
    df_featured['year'] = df_featured.index.year
    df_featured['day_of_year'] = df_featured.index.dayofyear

    # Seasonal features
    df_featured['is_weekend'] = df_featured['day_of_week'].isin([5, 6]).astype(int)
    df_featured['is_month_end'] = (df_featured.index.day > 25).astype(int)

    # Remove rows with NaN (due to lag features)
    df_featured = df_featured.dropna()

    return df_featured

# Create features
df_featured = create_time_series_features(pd.DataFrame(daily_sales))

# Prepare data for ML
feature_cols = [col for col in df_featured.columns if col != 'sales']
X = df_featured[feature_cols]
y = df_featured['sales']

# Time series split for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train ML models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
}

model_scores = {}

for model_name, model in models.items():
    cv_scores = []

    for train_idx, test_idx in tscv.split(X_scaled):
        X_train_cv, X_test_cv = X_scaled[train_idx], X_scaled[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train_cv, y_train_cv)
        y_pred_cv = model.predict(X_test_cv)

        rmse = np.sqrt(mean_squared_error(y_test_cv, y_pred_cv))
        cv_scores.append(rmse)

    model_scores[model_name] = {
        'mean_rmse': np.mean(cv_scores),
        'std_rmse': np.std(cv_scores)
    }

    print(f"{model_name} CV RMSE: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores):.2f})")

# Train best model on full training data
best_model_name = min(model_scores.keys(), key=lambda x: model_scores[x]['mean_rmse'])
best_model = models[best_model_name]

# Final training
train_size = int(len(df_featured) * 0.8)
X_train_final = X_scaled[:train_size]
y_train_final = y.iloc[:train_size]
X_test_final = X_scaled[train_size:]
y_test_final = y.iloc[train_size:]

best_model.fit(X_train_final, y_train_final)
y_pred_final = best_model.predict(X_test_final)

final_rmse = np.sqrt(mean_squared_error(y_test_final, y_pred_final))
print(f"\nFinal {best_model_name} RMSE: {final_rmse:.2f}")

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15])
    plt.title(f'{best_model_name} Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance_ts.png')
    plt.show()
```

## 5. Project Deliverables and Assessment

### 5.1 Required Deliverables

#### Technical Deliverables
1. **Complete codebase** with modular functions and classes
2. **Data preprocessing pipeline** with automated cleaning and feature engineering
3. **Model training and evaluation scripts** with cross-validation
4. **Model deployment code** for production systems
5. **Documentation** with setup instructions and API references
6. **Unit tests** for critical functions

#### Business Deliverables
1. **Executive summary** with key findings and business impact
2. **Technical report** with methodology and model details
3. **Visualization dashboard** for model monitoring
4. **Recommendations document** with actionable insights
5. **Presentation slides** for stakeholder communication

### 5.2 Assessment Rubric

#### Project Complexity (20%)
- **Problem scope and difficulty**: 5%
- **Data complexity and volume**: 5%
- **Technical implementation challenges**: 5%
- **Business impact potential**: 5%

#### Technical Excellence (40%)
- **Data preprocessing quality**: 8%
- **Feature engineering creativity**: 7%
- **Model selection and validation**: 8%
- **Code quality and documentation**: 7%
- **Performance optimization**: 5%
- **Error handling and robustness**: 5%

#### Business Value (25%)
- **Problem understanding**: 5%
- **Solution effectiveness**: 7%
- **Actionable insights**: 6%
- **Business recommendations**: 4%
- **Impact quantification**: 3%

#### Communication (15%)
- **Report clarity and structure**: 4%
- **Visualization quality**: 4%
- **Presentation effectiveness**: 4%
- **Stakeholder communication**: 3%

## 6. Industry Case Studies

### 6.1 Netflix Recommendation System
**Challenge**: Provide personalized content recommendations from 20,000+ titles
**Solution**: Hybrid collaborative filtering + content-based filtering
**Impact**: 75% of watched content comes from recommendations
**Key Technologies**: Matrix factorization, deep learning, A/B testing

### 6.2 Uber Dynamic Pricing
**Challenge**: Optimize pricing based on real-time supply/demand
**Solution**: Time series forecasting + reinforcement learning
**Impact**: 20% increase in driver utilization
**Key Technologies**: Streaming analytics, ML pipelines, real-time processing

### 6.3 Airbnb Host Recommendations
**Challenge**: Help hosts optimize listings for better bookings
**Solution**: NLP analysis + predictive modeling
**Impact**: 10% increase in booking rates
**Key Technologies**: Text mining, computer vision, recommendation algorithms

### 6.4 Spotify Music Discovery
**Challenge**: Create personalized playlists and discovery features
**Solution**: Audio feature extraction + collaborative filtering
**Impact**: 30% increase in user engagement
**Key Technologies**: Audio signal processing, embeddings, graph algorithms

## Next Steps

Congratulations on completing comprehensive data science projects! You now have hands-on experience with real-world applications across multiple domains. In the final module, we'll explore career development strategies and professional growth in data science.

**Ready to continue?** Proceed to [Module 14: Career Development](../14_career_development/)
