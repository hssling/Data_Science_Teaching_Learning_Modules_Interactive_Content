#!/usr/bin/env python3
"""
Predictive Analytics Project: Customer Churn Prediction
=======================================================

A complete end-to-end data science project demonstrating:
- Data collection and preprocessing
- Exploratory data analysis
- Feature engineering
- Model building and evaluation
- Model deployment considerations

Dataset: Telco Customer Churn dataset
Goal: Predict which customers are likely to churn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🔮 PREDICTIVE ANALYTICS PROJECT: CUSTOMER CHURN PREDICTION")
print("=" * 70)

class ChurnPredictionProject:
    """Complete customer churn prediction project"""

    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.preprocessor = None

    def load_data(self):
        """Load and examine the dataset"""
        print("\n📊 1. DATA LOADING AND INITIAL EXPLORATION")
        print("-" * 50)

        # For this example, we'll create a synthetic dataset
        # In a real project, you would load from a CSV file
        np.random.seed(42)
        n_samples = 5000

        # Generate synthetic customer data
        data = {
            'customerID': [f'CUST_{i:04d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'tenure': np.random.poisson(30, n_samples),  # Months with company
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.6, 0.2, 0.2]),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
            'MonthlyCharges': np.random.uniform(20, 120, n_samples),
            'TotalCharges': np.random.uniform(100, 8000, n_samples),
        }

        # Create target variable (Churn) with realistic correlations
        churn_probabilities = []

        for i in range(n_samples):
            base_prob = 0.2  # Base churn rate

            # Higher churn for month-to-month contracts
            if data['Contract'][i] == 'Month-to-month':
                base_prob += 0.3
            elif data['Contract'][i] == 'One year':
                base_prob += 0.1

            # Higher churn for electronic check payments
            if data['PaymentMethod'][i] == 'Electronic check':
                base_prob += 0.15

            # Lower tenure reduces churn probability
            tenure_factor = max(0, (60 - data['tenure'][i]) / 60)
            base_prob += tenure_factor * 0.2

            # Senior citizens more likely to churn
            if data['SeniorCitizen'][i] == 1:
                base_prob += 0.1

            # Higher monthly charges increase churn
            charge_factor = (data['MonthlyCharges'][i] - 20) / 100
            base_prob += charge_factor * 0.1

            # Add some randomness
            final_prob = min(0.8, max(0.05, base_prob + np.random.normal(0, 0.1)))
            churn_probabilities.append(final_prob)

        data['Churn'] = np.random.binomial(1, churn_probabilities, n_samples)
        data['Churn'] = data['Churn'].map({0: 'No', 1: 'Yes'})

        self.data = pd.DataFrame(data)

        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        print(f"\nChurn distribution:")
        print(self.data['Churn'].value_counts(normalize=True))

        return self.data

    def exploratory_data_analysis(self):
        """Perform comprehensive exploratory data analysis"""
        print("\n📈 2. EXPLORATORY DATA ANALYSIS")
        print("-" * 40)

        # Basic statistics
        print("Basic Statistics:")
        print(self.data.describe())

        # Churn rate by categorical variables
        categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                          'PhoneService', 'InternetService', 'Contract',
                          'PaperlessBilling', 'PaymentMethod']

        print("\nChurn Rates by Categorical Variables:")
        print("-" * 50)

        for col in categorical_cols:
            churn_rate = self.data.groupby(col)['Churn'].value_counts(normalize=True).unstack()
            print(f"\n{col}:")
            print(churn_rate['Yes'].round(3))

        # Visualize key relationships
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Churn by contract type
        contract_churn = self.data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
        contract_churn['Yes'].plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Churn Rate by Contract Type')
        axes[0,0].set_ylabel('Churn Rate')
        axes[0,0].tick_params(axis='x', rotation=45)

        # Churn by payment method
        payment_churn = self.data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack()
        payment_churn['Yes'].plot(kind='bar', ax=axes[0,1], color='lightcoral')
        axes[0,1].set_title('Churn Rate by Payment Method')
        axes[0,1].set_ylabel('Churn Rate')
        axes[0,1].tick_params(axis='x', rotation=45)

        # Tenure distribution by churn
        self.data[self.data['Churn'] == 'No']['tenure'].hist(alpha=0.7, bins=30, ax=axes[0,2], label='No Churn')
        self.data[self.data['Churn'] == 'Yes']['tenure'].hist(alpha=0.7, bins=30, ax=axes[0,2], label='Churn')
        axes[0,2].set_title('Tenure Distribution by Churn Status')
        axes[0,2].set_xlabel('Tenure (months)')
        axes[0,2].legend()

        # Monthly charges distribution by churn
        self.data[self.data['Churn'] == 'No']['MonthlyCharges'].hist(alpha=0.7, bins=30, ax=axes[1,0], label='No Churn')
        self.data[self.data['Churn'] == 'Yes']['MonthlyCharges'].hist(alpha=0.7, bins=30, ax=axes[1,0], label='Churn')
        axes[1,0].set_title('Monthly Charges Distribution by Churn Status')
        axes[1,0].set_xlabel('Monthly Charges ($)')
        axes[1,0].legend()

        # Correlation heatmap for numerical variables
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
        correlation_matrix = self.data[numerical_cols + [self.data['Churn'].map({'No': 0, 'Yes': 1})]].corr()

        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[1,1])
        axes[1,1].set_title('Correlation Matrix')

        # Churn rate by tenure groups
        self.data['tenure_group'] = pd.cut(self.data['tenure'],
                                         bins=[0, 12, 24, 36, 48, 60, 100],
                                         labels=['0-12', '12-24', '24-36', '36-48', '48-60', '60+'])

        tenure_churn = self.data.groupby('tenure_group')['Churn'].value_counts(normalize=True).unstack()
        tenure_churn['Yes'].plot(kind='line', marker='o', ax=axes[1,2], color='green')
        axes[1,2].set_title('Churn Rate by Tenure Group')
        axes[1,2].set_xlabel('Tenure Group (months)')
        axes[1,2].set_ylabel('Churn Rate')

        plt.tight_layout()
        plt.savefig('churn_eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("EDA visualizations saved as 'churn_eda_analysis.png'")

    def feature_engineering(self):
        """Create new features and prepare data for modeling"""
        print("\n🔧 3. FEATURE ENGINEERING")
        print("-" * 30)

        # Create a copy of the data
        df = self.data.copy()

        # Convert TotalCharges to numeric (handle empty strings)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'], inplace=True)

        # Create new features
        df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)  # Avoid division by zero
        df['ChargesDifference'] = df['MonthlyCharges'] - df['AvgMonthlySpend']
        df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 100],
                                 labels=['New', 'Medium', 'Long', 'Very_Long'])

        # Service usage score (count of additional services)
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies']

        df['ServiceCount'] = df[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
        df['HasMultipleServices'] = (df['ServiceCount'] > 2).astype(int)

        # Contract and payment risk score
        contract_risk = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
        payment_risk = {'Electronic check': 3, 'Mailed check': 2,
                       'Bank transfer (automatic)': 1, 'Credit card (automatic)': 1}

        df['ContractRisk'] = df['Contract'].map(contract_risk)
        df['PaymentRisk'] = df['PaymentMethod'].map(payment_risk)
        df['TotalRiskScore'] = df['ContractRisk'] + df['PaymentRisk']

        # Encode target variable
        df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

        print("New features created:")
        print("- AvgMonthlySpend: Average monthly spending")
        print("- ChargesDifference: Difference between current and average charges")
        print("- TenureGroup: Categorical tenure groups")
        print("- ServiceCount: Number of additional services")
        print("- HasMultipleServices: Binary indicator for multiple services")
        print("- ContractRisk: Risk score based on contract type")
        print("- PaymentRisk: Risk score based on payment method")
        print("- TotalRiskScore: Combined risk score")

        # Prepare features for modeling
        # Identify column types
        categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                          'MultipleLines', 'InternetService', 'OnlineSecurity',
                          'OnlineBackup', 'DeviceProtection', 'TechSupport',
                          'StreamingTV', 'StreamingMovies', 'Contract',
                          'PaperlessBilling', 'PaymentMethod', 'TenureGroup']

        numerical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
                         'AvgMonthlySpend', 'ChargesDifference', 'ServiceCount',
                         'ContractRisk', 'PaymentRisk', 'TotalRiskScore']

        # Create preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        # Prepare X and y
        feature_cols = numerical_cols + categorical_cols
        X = df[feature_cols]
        y = df['Churn']

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nTraining set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Churn rate in training set: {self.y_train.mean():.3f}")
        print(f"Churn rate in test set: {self.y_test.mean():.3f}")

        return X, y

    def model_building(self):
        """Build and compare multiple machine learning models"""
        print("\n🤖 4. MODEL BUILDING AND COMPARISON")
        print("-" * 40)

        # Define models to compare
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
        }

        # Create pipelines with preprocessing
        pipelines = {}
        for name, model in models.items():
            pipelines[name] = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])

        # Train and evaluate models
        results = {}
        for name, pipeline in pipelines.items():
            print(f"\nTraining {name}...")

            # Cross-validation scores
            cv_scores = cross_val_score(pipeline, self.X_train, self.y_train,
                                      cv=5, scoring='roc_auc')

            # Fit on full training data
            pipeline.fit(self.X_train, self.y_train)

            # Predictions on test set
            y_pred = pipeline.predict(self.X_test)
            y_pred_proba = pipeline.predict_proba(self.X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)

            results[name] = {
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'pipeline': pipeline,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

            print(f"  CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            print(f"  Test ROC AUC: {roc_auc:.3f}")
            print(f"  Test Accuracy: {accuracy:.3f}")
            print(f"  Test F1-Score: {f1:.3f}")

        self.models = results

        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
        self.best_model = results[best_model_name]

        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   ROC AUC: {self.best_model['roc_auc']:.3f}")
        print(f"   Accuracy: {self.best_model['accuracy']:.3f}")
        print(f"   F1-Score: {self.best_model['f1']:.3f}")

        return results

    def model_evaluation(self):
        """Detailed evaluation of the best model"""
        print("\n📊 5. MODEL EVALUATION AND INTERPRETATION")
        print("-" * 50)

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.best_model['predictions'])
        print("Confusion Matrix:")
        print(cm)

        # Classification Report
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.best_model['predictions'],
                                  target_names=['No Churn', 'Churn']))

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(self.y_test, self.best_model['probabilities'])

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {self.best_model["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Churn Prediction Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('churn_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Feature Importance (for tree-based models)
        if hasattr(self.best_model['pipeline'].named_steps['classifier'], 'feature_importances_'):
            # Get feature names after preprocessing
            feature_names = self._get_feature_names()
            importances = self.best_model['pipeline'].named_steps['classifier'].feature_importances_

            # Create feature importance dataframe
            feat_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            # Plot top 20 features
            plt.figure(figsize=(12, 8))
            top_features = feat_importance.head(20)
            sns.barplot(x='importance', y='feature', data=top_features)
            plt.title('Top 20 Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('churn_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()

            print("\nTop 10 Most Important Features:")
            for i, (_, row) in enumerate(feat_importance.head(10).iterrows()):
                print(f"{i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")

    def _get_feature_names(self):
        """Get feature names after preprocessing"""
        # Get numerical feature names
        numerical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
                         'AvgMonthlySpend', 'ChargesDifference', 'ServiceCount',
                         'ContractRisk', 'PaymentRisk', 'TotalRiskScore']

        # Get categorical feature names after one-hot encoding
        categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                          'MultipleLines', 'InternetService', 'OnlineSecurity',
                          'OnlineBackup', 'DeviceProtection', 'TechSupport',
                          'StreamingTV', 'StreamingMovies', 'Contract',
                          'PaperlessBilling', 'PaymentMethod', 'TenureGroup']

        # Fit preprocessor to get feature names
        self.preprocessor.fit(self.X_train)
        feature_names = (numerical_cols +
                        list(self.preprocessor.named_transformers_['cat']
                            .named_steps['onehot'].get_feature_names_out(categorical_cols)))

        return feature_names

    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning on the best model"""
        print("\n⚙️ 6. HYPERPARAMETER TUNING")
        print("-" * 30)

        # Define parameter grid for Random Forest (assuming it's the best model)
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2']
        }

        # Create pipeline for tuning
        rf_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        # Grid search with cross-validation
        print("Performing grid search with 5-fold cross-validation...")
        grid_search = GridSearchCV(
            rf_pipeline,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.X_train, self.y_train)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")

        # Evaluate best model on test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]

        test_roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        test_f1 = f1_score(self.y_test, y_pred)

        print(f"\nTuned Model Test Performance:")
        print(f"ROC AUC: {test_roc_auc:.3f}")
        print(f"Accuracy: {test_accuracy:.3f}")
        print(f"F1-Score: {test_f1:.3f}")

        return grid_search.best_estimator_

    def model_deployment_considerations(self):
        """Discuss model deployment and production considerations"""
        print("\n🚀 7. MODEL DEPLOYMENT CONSIDERATIONS")
        print("-" * 45)

        print("Production Deployment Checklist:")
        print("✓ Model Serialization:")
        print("  - Save trained model using joblib or pickle")
        print("  - Version control for models")
        print("  - Model registry for tracking versions")

        print("\n✓ API Development:")
        print("  - REST API using Flask/FastAPI")
        print("  - Input validation and error handling")
        print("  - Response formatting")

        print("\n✓ Monitoring & Maintenance:")
        print("  - Performance monitoring (accuracy, latency)")
        print("  - Data drift detection")
        print("  - Model retraining pipeline")
        print("  - Alert system for model degradation")

        print("\n✓ Scalability:")
        print("  - Containerization with Docker")
        print("  - Orchestration with Kubernetes")
        print("  - Load balancing and auto-scaling")

        print("\n✓ Security & Compliance:")
        print("  - Data privacy and GDPR compliance")
        print("  - Input sanitization")
        print("  - Rate limiting and authentication")

        # Example model serialization
        import joblib

        print("\n💾 Example: Model Serialization")
        print("-" * 30)

        # Save the best model
        joblib.dump(self.best_model['pipeline'], 'churn_prediction_model.pkl')
        print("Model saved as 'churn_prediction_model.pkl'")

        # Example prediction function
        def predict_churn(customer_data):
            """
            Example prediction function for deployment

            Parameters:
            customer_data (dict): Customer features

            Returns:
            dict: Prediction results
            """
            # Load model (in production, load once at startup)
            model = joblib.load('churn_prediction_model.pkl')

            # Convert to DataFrame
            df = pd.DataFrame([customer_data])

            # Make prediction
            prediction = model.predict(df)[0]
            probability = model.predict_proba(df)[0][1]

            return {
                'churn_prediction': 'Yes' if prediction == 1 else 'No',
                'churn_probability': probability,
                'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low'
            }

        # Example usage
        sample_customer = {
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 12,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 75.5,
            'TotalCharges': 900.0
        }

        result = predict_churn(sample_customer)
        print(f"\nSample Prediction for new customer:")
        print(f"Churn Prediction: {result['churn_prediction']}")
        print(f"Churn Probability: {result['churn_probability']:.3f}")
        print(f"Risk Level: {result['risk_level']}")

    def run_complete_project(self):
        """Run the complete churn prediction project"""
        print("Starting Complete Customer Churn Prediction Project")
        print("=" * 60)

        # Execute all steps
        self.load_data()
        self.exploratory_data_analysis()
        self.feature_engineering()
        self.model_building()
        self.model_evaluation()
        self.hyperparameter_tuning()
        self.model_deployment_considerations()

        print("\n" + "=" * 60)
        print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Generated files:")
        print("- churn_eda_analysis.png")
        print("- churn_roc_curve.png")
        print("- churn_feature_importance.png")
        print("- churn_prediction_model.pkl")
        print("\nKey Achievements:")
        print("✓ Complete data science workflow implemented")
        print("✓ Multiple ML models compared and evaluated")
        print("✓ Feature engineering and preprocessing pipeline")
        print("✓ Model interpretation and deployment considerations")
        print("✓ Production-ready code with proper documentation")

if __name__ == "__main__":
    # Run the complete project
    project = ChurnPredictionProject()
    project.run_complete_project()
