# Module 12: Ethics and Best Practices in Data Science

## Overview
Ethical considerations and best practices are fundamental to responsible data science. This module explores the ethical challenges in data collection, model development, and deployment, along with industry standards and frameworks for responsible AI. You'll learn to identify bias, ensure fairness, maintain privacy, and implement ethical decision-making throughout the data science lifecycle.

## Learning Objectives
By the end of this module, you will be able to:
- Understand ethical challenges in data science and AI
- Identify and mitigate bias in data and algorithms
- Implement privacy-preserving techniques
- Apply fairness metrics and evaluation frameworks
- Understand regulatory compliance (GDPR, CCPA, etc.)
- Develop ethical AI deployment strategies
- Communicate ethical considerations to stakeholders
- Create responsible AI governance frameworks

## 1. Introduction to Data Ethics

### 1.1 The Importance of Ethics in Data Science

#### Why Ethics Matter
- **Human Impact**: Data science decisions affect real people's lives
- **Trust and Transparency**: Building trust with users and stakeholders
- **Legal Compliance**: Meeting regulatory requirements
- **Social Responsibility**: Contributing positively to society
- **Professional Integrity**: Maintaining ethical standards in the field

#### Ethical Challenges in Data Science
```python
# Conceptual framework for ethical considerations
ethical_framework = {
    'data_collection': {
        'issues': ['Privacy violation', 'Consent concerns', 'Biased sampling'],
        'principles': ['Informed consent', 'Purpose limitation', 'Data minimization']
    },
    'data_processing': {
        'issues': ['Discriminatory bias', 'Lack of transparency', 'Data quality problems'],
        'principles': ['Fairness', 'Accountability', 'Explainability']
    },
    'model_development': {
        'issues': ['Algorithmic bias', 'Unintended consequences', 'Over-reliance on models'],
        'principles': ['Robustness', 'Safety', 'Human oversight']
    },
    'deployment_usage': {
        'issues': ['Misuse of AI', 'Lack of accountability', 'Inequality amplification'],
        'principles': ['Beneficence', 'Non-maleficence', 'Justice']
    }
}

print("Ethical Framework for Data Science:")
print("=" * 50)
for phase, considerations in ethical_framework.items():
    print(f"\n{phase.upper().replace('_', ' ')}:")
    print(f"  Key Issues: {', '.join(considerations['issues'])}")
    print(f"  Guiding Principles: {', '.join(considerations['principles'])}")
```

### 1.2 Ethical Decision-Making Framework

#### Step-by-Step Ethical Analysis
1. **Identify Stakeholders**: Who is affected by the decision?
2. **Assess Potential Harm**: What negative impacts could occur?
3. **Evaluate Benefits**: What positive outcomes are expected?
4. **Consider Alternatives**: Are there less harmful approaches?
5. **Apply Ethical Principles**: Which principles should guide the decision?
6. **Seek Diverse Perspectives**: Include different viewpoints
7. **Document Decisions**: Record reasoning and trade-offs
8. **Monitor Outcomes**: Track actual impacts and adjust

## 2. Bias and Fairness in Data Science

### 2.1 Understanding Bias in Data and Algorithms

#### Types of Bias
```python
bias_types = {
    'data_bias': {
        'definition': 'Bias present in the training data',
        'examples': ['Sampling bias', 'Measurement bias', 'Historical bias'],
        'causes': ['Unrepresentative samples', 'Measurement errors', 'Societal prejudices']
    },
    'algorithmic_bias': {
        'definition': 'Bias introduced by the algorithm or model',
        'examples': ['Confirmation bias', 'Selection bias', 'Omitted variable bias'],
        'causes': ['Flawed assumptions', 'Incomplete features', 'Optimization objectives']
    },
    'human_bias': {
        'definition': 'Bias introduced by human decisions in the process',
        'examples': ['Confirmation bias', 'Anchoring bias', 'Availability bias'],
        'causes': ['Cognitive limitations', 'Time pressure', 'Limited perspective']
    },
    'deployment_bias': {
        'definition': 'Bias that emerges when models are deployed',
        'examples': ['Feedback loops', 'Concept drift', 'Population shift'],
        'causes': ['Changing environments', 'User behavior changes', 'Model limitations']
    }
}

print("Types of Bias in Data Science:")
print("=" * 40)
for bias_type, details in bias_types.items():
    print(f"\n{bias_type.upper().replace('_', ' ')}:")
    print(f"  Definition: {details['definition']}")
    print(f"  Examples: {', '.join(details['examples'])}")
    print(f"  Common Causes: {', '.join(details['causes'])}")
```

#### Detecting Bias in Datasets
```python
import pandas as pd
import numpy as np
from scipy import stats

class BiasDetector:
    """Comprehensive bias detection framework"""

    def __init__(self, df: pd.DataFrame, sensitive_attributes: list = None):
        self.df = df.copy()
        self.sensitive_attributes = sensitive_attributes or ['gender', 'race', 'age', 'income']

    def detect_demographic_bias(self, target_column: str):
        """Detect demographic bias in target variable distribution"""

        bias_analysis = {}

        for attribute in self.sensitive_attributes:
            if attribute in self.df.columns:
                # Calculate distribution by sensitive attribute
                distribution = self.df.groupby(attribute)[target_column].value_counts(normalize=True)
                distribution = distribution.unstack().fillna(0)

                # Calculate disparity metrics
                if len(distribution.columns) > 1:
                    # Representation disparity
                    expected_rate = self.df[target_column].mean()
                    group_rates = distribution.mean(axis=1)

                    disparity = (group_rates - expected_rate).abs()

                    bias_analysis[attribute] = {
                        'distribution': distribution.to_dict(),
                        'disparity_score': disparity.max(),
                        'most_disadvantaged_group': disparity.idxmax(),
                        'bias_detected': disparity.max() > 0.1  # Threshold for concern
                    }

        return bias_analysis

    def detect_feature_bias(self, features: list, target_column: str):
        """Detect bias in feature-target relationships"""

        feature_bias = {}

        for feature in features:
            if feature in self.df.columns and feature != target_column:
                # Calculate correlation with target
                if self.df[feature].dtype in ['int64', 'float64']:
                    correlation = self.df[feature].corr(self.df[target_column])
                else:
                    # For categorical features, use ANOVA or chi-square
                    try:
                        groups = [group[target_column] for name, group in self.df.groupby(feature)]
                        f_stat, p_value = stats.f_oneway(*groups)
                        correlation = np.sqrt(f_stat) if p_value < 0.05 else 0
                    except:
                        correlation = 0

                feature_bias[feature] = {
                    'correlation_with_target': correlation,
                    'potential_bias_concern': abs(correlation) > 0.7
                }

        return feature_bias

    def detect_label_bias(self, predicted_labels: np.ndarray, true_labels: np.ndarray,
                         protected_groups: dict):
        """Detect bias in model predictions"""

        prediction_bias = {}

        for group_name, group_mask in protected_groups.items():
            # Calculate performance metrics by group
            group_true = true_labels[group_mask]
            group_pred = predicted_labels[group_mask]

            if len(group_true) > 0:
                # Accuracy by group
                group_accuracy = np.mean(group_true == group_pred)

                # Overall accuracy
                overall_accuracy = np.mean(true_labels == predicted_labels)

                # Disparity
                accuracy_disparity = abs(group_accuracy - overall_accuracy)

                prediction_bias[group_name] = {
                    'group_accuracy': group_accuracy,
                    'overall_accuracy': overall_accuracy,
                    'accuracy_disparity': accuracy_disparity,
                    'bias_detected': accuracy_disparity > 0.05  # 5% threshold
                }

        return prediction_bias

    def generate_bias_report(self, target_column: str, features: list = None):
        """Generate comprehensive bias analysis report"""

        if features is None:
            features = [col for col in self.df.columns if col != target_column]

        report = {
            'demographic_bias': self.detect_demographic_bias(target_column),
            'feature_bias': self.detect_feature_bias(features, target_column),
            'recommendations': []
        }

        # Generate recommendations
        for attr, analysis in report['demographic_bias'].items():
            if analysis.get('bias_detected', False):
                report['recommendations'].append(
                    f"Address demographic bias in {attr}: {analysis['most_disadvantaged_group']} "
                    f"shows {analysis['disparity_score']:.3f} disparity"
                )

        for feature, analysis in report['feature_bias'].items():
            if analysis.get('potential_bias_concern', False):
                report['recommendations'].append(
                    f"Review feature {feature}: High correlation ({analysis['correlation_with_target']:.3f}) "
                    "with target may indicate bias"
                )

        return report

# Usage example
# bias_detector = BiasDetector(df, sensitive_attributes=['gender', 'race', 'age_group'])
# bias_report = bias_detector.generate_bias_report('loan_approved', features=['income', 'credit_score'])
# print("Bias Analysis Report:")
# for rec in bias_report['recommendations']:
#     print(f"• {rec}")
```

### 2.2 Fairness Metrics and Evaluation

#### Fairness Metrics Implementation
```python
import numpy as np
from typing import Dict, List, Tuple

class FairnessEvaluator:
    """Comprehensive fairness evaluation framework"""

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, protected_groups: Dict[str, np.ndarray]):
        self.y_true = y_true
        self.y_pred = y_pred
        self.protected_groups = protected_groups

    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
        """Calculate confusion matrix components"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

    def demographic_parity(self, group_mask: np.ndarray) -> float:
        """Calculate demographic parity (acceptance rate equality)"""
        group_pred_rate = np.mean(self.y_pred[group_mask])
        overall_pred_rate = np.mean(self.y_pred)

        return abs(group_pred_rate - overall_pred_rate)

    def equal_opportunity(self, group_mask: np.ndarray) -> float:
        """Calculate equal opportunity (true positive rate equality)"""
        group_true_positives = (self.y_true[group_mask] == 1) & (self.y_pred[group_mask] == 1)
        overall_true_positives = (self.y_true == 1) & (self.y_pred == 1)

        if np.sum(self.y_true == 1) == 0:
            return 0.0

        group_tpr = np.sum(group_true_positives) / np.sum(self.y_true[group_mask] == 1)
        overall_tpr = np.sum(overall_true_positives) / np.sum(self.y_true == 1)

        return abs(group_tpr - overall_tpr)

    def equalized_odds(self, group_mask: np.ndarray) -> Tuple[float, float]:
        """Calculate equalized odds (both TPR and FPR equality)"""
        # True Positive Rate difference
        tpr_diff = self.equal_opportunity(group_mask)

        # False Positive Rate difference
        group_false_positives = (self.y_true[group_mask] == 0) & (self.y_pred[group_mask] == 1)
        overall_false_positives = (self.y_true == 0) & (self.y_pred == 1)

        if np.sum(self.y_true == 0) == 0:
            fpr_diff = 0.0
        else:
            group_fpr = np.sum(group_false_positives) / np.sum(self.y_true[group_mask] == 0)
            overall_fpr = np.sum(overall_false_positives) / np.sum(self.y_true == 0)
            fpr_diff = abs(group_fpr - overall_fpr)

        return tpr_diff, fpr_diff

    def disparate_impact(self, group_mask: np.ndarray) -> float:
        """Calculate disparate impact ratio"""
        group_selection_rate = np.mean(self.y_pred[group_mask])
        overall_selection_rate = np.mean(self.y_pred)

        if overall_selection_rate == 0:
            return 1.0

        return group_selection_rate / overall_selection_rate

    def evaluate_fairness(self) -> Dict[str, Dict[str, float]]:
        """Comprehensive fairness evaluation"""

        fairness_metrics = {}

        for group_name, group_mask in self.protected_groups.items():
            metrics = {
                'demographic_parity': self.demographic_parity(group_mask),
                'equal_opportunity': self.equal_opportunity(group_mask),
                'disparate_impact': self.disparate_impact(group_mask)
            }

            tpr_diff, fpr_diff = self.equalized_odds(group_mask)
            metrics['equalized_odds_tpr'] = tpr_diff
            metrics['equalized_odds_fpr'] = fpr_diff

            # Overall fairness score (lower is better)
            metrics['fairness_score'] = np.mean([metrics['demographic_parity'],
                                               metrics['equal_opportunity'],
                                               tpr_diff, fpr_diff])

            fairness_metrics[group_name] = metrics

        return fairness_metrics

    def generate_fairness_report(self) -> Dict:
        """Generate comprehensive fairness report"""

        fairness_results = self.evaluate_fairness()

        report = {
            'fairness_metrics': fairness_results,
            'thresholds': {
                'demographic_parity': 0.05,  # 5% difference threshold
                'equal_opportunity': 0.05,
                'disparate_impact': 0.8,     # 80% rule threshold
                'equalized_odds': 0.05
            },
            'violations': {},
            'recommendations': []
        }

        # Check for violations
        thresholds = report['thresholds']

        for group_name, metrics in fairness_results.items():
            violations = []

            if metrics['demographic_parity'] > thresholds['demographic_parity']:
                violations.append('demographic_parity')
            if metrics['equal_opportunity'] > thresholds['equal_opportunity']:
                violations.append('equal_opportunity')
            if metrics['disparate_impact'] < thresholds['disparate_impact']:
                violations.append('disparate_impact')
            if metrics['equalized_odds_tpr'] > thresholds['equalized_odds'] or \
               metrics['equalized_odds_fpr'] > thresholds['equalized_odds']:
                violations.append('equalized_odds')

            if violations:
                report['violations'][group_name] = violations

                # Generate recommendations
                if 'demographic_parity' in violations:
                    report['recommendations'].append(
                        f"Address demographic parity for {group_name}: "
                        f"Selection rate disparity = {metrics['demographic_parity']:.3f}"
                    )
                if 'equal_opportunity' in violations:
                    report['recommendations'].append(
                        f"Address equal opportunity for {group_name}: "
                        f"True positive rate disparity = {metrics['equal_opportunity']:.3f}"
                    )

        return report

# Usage example
# protected_groups = {
#     'female': df['gender'] == 'female',
#     'minority': df['race'] != 'white'
# }
#
# fairness_evaluator = FairnessEvaluator(y_true, y_pred, protected_groups)
# fairness_report = fairness_evaluator.generate_fairness_report()
#
# print("Fairness Evaluation Report:")
# print(f"Groups with fairness violations: {list(fairness_report['violations'].keys())}")
# for rec in fairness_report['recommendations']:
#     print(f"• {rec}")
```

## 3. Privacy and Data Protection

### 3.1 Privacy-Preserving Techniques

#### Differential Privacy
```python
import numpy as np
from scipy import stats

class DifferentialPrivacy:
    """Differential privacy implementation for data analysis"""

    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon

    def add_laplace_noise(self, value: float, sensitivity: float) -> float:
        """Add Laplace noise for differential privacy"""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise

    def privatize_count(self, true_count: int) -> int:
        """Privatize a count query"""
        # Sensitivity for count queries is 1 (adding/removing one record changes count by 1)
        sensitivity = 1
        noisy_count = self.add_laplace_noise(true_count, sensitivity)
        return max(0, int(round(noisy_count)))  # Ensure non-negative

    def privatize_mean(self, data: np.ndarray) -> float:
        """Privatize mean calculation"""
        true_mean = np.mean(data)
        n = len(data)

        # Sensitivity for mean is 1/n (bounded data assumed to be in [0,1])
        sensitivity = 1.0 / n

        return self.add_laplace_noise(true_mean, sensitivity)

    def privatize_histogram(self, data: np.ndarray, bins: int = 10) -> np.ndarray:
        """Privatize histogram counts"""
        # Calculate true histogram
        hist, bin_edges = np.histogram(data, bins=bins)

        # Add noise to each bin count
        privatized_hist = []
        for count in hist:
            noisy_count = self.add_laplace_noise(count, sensitivity=1)
            privatized_hist.append(max(0, int(round(noisy_count))))

        return np.array(privatized_hist)

    def exponential_mechanism(self, candidates: list, scores: list, sensitivity: float):
        """Exponential mechanism for private selection"""
        # Calculate quality scores with noise
        noisy_scores = []
        for score in scores:
            noisy_score = self.add_laplace_noise(score, sensitivity)
            noisy_scores.append(noisy_score)

        # Select candidate with highest noisy score
        best_idx = np.argmax(noisy_scores)
        return candidates[best_idx]

# Usage example
# dp = DifferentialPrivacy(epsilon=0.1)  # Strong privacy (low epsilon)
#
# # Privatize a count
# true_count = 1000
# private_count = dp.privatize_count(true_count)
# print(f"True count: {true_count}, Private count: {private_count}")
#
# # Privatize a mean
# data = np.random.normal(0.5, 0.1, 1000)
# private_mean = dp.privatize_mean(data)
# print(f"True mean: {np.mean(data):.4f}, Private mean: {private_mean:.4f}")
```

#### Federated Learning
```python
import numpy as np
from typing import List, Dict, Any

class FederatedLearning:
    """Federated learning implementation for privacy-preserving ML"""

    def __init__(self, num_clients: int, model_architecture: dict):
        self.num_clients = num_clients
        self.global_model = self._initialize_model(model_architecture)
        self.client_models = [self._initialize_model(model_architecture)
                            for _ in range(num_clients)]

    def _initialize_model(self, architecture: dict):
        """Initialize a simple linear model"""
        return {
            'weights': np.random.randn(architecture.get('input_dim', 10),
                                     architecture.get('output_dim', 1)),
            'bias': np.random.randn(architecture.get('output_dim', 1))
        }

    def client_update(self, client_id: int, client_data: Dict[str, np.ndarray],
                     learning_rate: float = 0.01, epochs: int = 1):
        """Perform local training on client data"""

        X, y = client_data['X'], client_data['y']
        model = self.client_models[client_id]

        for epoch in range(epochs):
            # Forward pass
            predictions = X @ model['weights'] + model['bias']

            # Compute loss (MSE)
            loss = np.mean((predictions - y) ** 2)

            # Backward pass
            d_loss = 2 * (predictions - y) / len(X)
            d_weights = X.T @ d_loss
            d_bias = np.mean(d_loss, axis=0)

            # Update parameters
            model['weights'] -= learning_rate * d_weights
            model['bias'] -= learning_rate * d_bias

        return model

    def aggregate_models(self, client_updates: List[Dict[str, np.ndarray]]):
        """Aggregate client model updates using FedAvg"""

        # Initialize aggregated model
        aggregated_model = {
            'weights': np.zeros_like(self.global_model['weights']),
            'bias': np.zeros_like(self.global_model['bias'])
        }

        # Average client updates
        for client_update in client_updates:
            aggregated_model['weights'] += client_update['weights'] / self.num_clients
            aggregated_model['bias'] += client_update['bias'] / self.num_clients

        # Update global model
        self.global_model = aggregated_model

        # Update client models with global model
        for i in range(self.num_clients):
            self.client_models[i] = aggregated_model.copy()

        return self.global_model

    def federated_training_round(self, client_datasets: List[Dict[str, np.ndarray]],
                               learning_rate: float = 0.01, local_epochs: int = 1):
        """Perform one round of federated training"""

        # Client updates
        client_updates = []
        for client_id, client_data in enumerate(client_datasets):
            client_update = self.client_update(client_id, client_data,
                                             learning_rate, local_epochs)
            client_updates.append(client_update)

        # Aggregate updates
        global_model = self.aggregate_models(client_updates)

        return global_model

    def evaluate_global_model(self, test_data: Dict[str, np.ndarray]):
        """Evaluate the global model"""

        X_test, y_test = test_data['X'], test_data['y']

        predictions = X_test @ self.global_model['weights'] + self.global_model['bias']
        mse = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)

        return {'mse': mse, 'rmse': rmse}

# Usage example
# # Initialize federated learning
# fl = FederatedLearning(num_clients=3, model_architecture={'input_dim': 5, 'output_dim': 1})
#
# # Simulate client datasets (each client has different data)
# client_datasets = []
# for i in range(3):
#     X = np.random.randn(100, 5)
#     y = X @ np.array([1, 2, -1, 0.5, -0.5]) + np.random.randn(100) * 0.1
#     client_datasets.append({'X': X, 'y': y})
#
# # Perform federated training rounds
# for round_num in range(5):
#     global_model = fl.federated_training_round(client_datasets, learning_rate=0.01)
#     print(f"Round {round_num + 1} completed")
#
# # Evaluate final model
# test_data = {'X': np.random.randn(50, 5), 'y': np.random.randn(50)}
# evaluation = fl.evaluate_global_model(test_data)
# print(f"Final model RMSE: {evaluation['rmse']:.4f}")
```

### 3.2 Data Anonymization Techniques

#### K-Anonymity and L-Diversity
```python
import pandas as pd
import numpy as np
from typing import List, Set

class DataAnonymizer:
    """Data anonymization techniques for privacy protection"""

    def __init__(self, df: pd.DataFrame, quasi_identifiers: List[str]):
        self.df = df.copy()
        self.quasi_identifiers = quasi_identifiers

    def check_k_anonymity(self, k: int = 5) -> Dict[str, Any]:
        """Check if dataset satisfies k-anonymity"""

        # Group by quasi-identifiers
        grouped = self.df.groupby(self.quasi_identifiers).size()

        # Find groups with size < k
        violating_groups = grouped[grouped < k]

        anonymity_check = {
            'k_value': k,
            'total_records': len(self.df),
            'unique_groups': len(grouped),
            'groups_below_k': len(violating_groups),
            'records_at_risk': violating_groups.sum(),
            'k_anonymity_satisfied': len(violating_groups) == 0,
            'anonymity_level': len(grouped) / len(self.df) if len(self.df) > 0 else 0
        }

        return anonymity_check

    def generalize_data(self, generalization_rules: Dict[str, Dict]) -> pd.DataFrame:
        """Apply generalization to reduce uniqueness"""

        df_anonymized = self.df.copy()

        for column, rules in generalization_rules.items():
            if column in df_anonymized.columns:
                if 'bins' in rules:
                    # Numerical generalization using bins
                    df_anonymized[column] = pd.cut(
                        df_anonymized[column],
                        bins=rules['bins'],
                        labels=rules.get('labels', None)
                    )
                elif 'mapping' in rules:
                    # Categorical generalization using mapping
                    df_anonymized[column] = df_anonymized[column].map(rules['mapping'])

        return df_anonymized

    def add_noise(self, columns: List[str], noise_level: float = 0.1) -> pd.DataFrame:
        """Add noise to numerical columns for differential privacy"""

        df_noisy = self.df.copy()

        for column in columns:
            if column in df_noisy.columns and df_noisy[column].dtype in ['int64', 'float64']:
                # Add Laplace noise
                scale = noise_level * df_noisy[column].std()
                noise = np.random.laplace(0, scale, len(df_noisy))
                df_noisy[column] += noise

        return df_noisy

    def create_suppression_mask(self, k: int = 5) -> np.ndarray:
        """Create suppression mask for records that violate k-anonymity"""

        grouped = self.df.groupby(self.quasi_identifiers).size()
        small_groups = grouped[grouped < k]

        # Create mask for records in small groups
        suppression_mask = np.zeros(len(self.df), dtype=bool)

        for group_values in small_groups.index:
            if isinstance(group_values, tuple):
                # Multi-column group
                mask = np.ones(len(self.df), dtype=bool)
                for i, col in enumerate(self.quasi_identifiers):
                    mask &= (self.df[col] == group_values[i])
            else:
                # Single column group
                mask = (self.df[self.quasi_identifiers[0]] == group_values)

            suppression_mask |= mask

        return suppression_mask

    def apply_k_anonymity(self, k: int = 5, generalization_rules: Dict = None) -> pd.DataFrame:
        """Apply k-anonymity through generalization and suppression"""

        df_anonymized = self.df.copy()

        # Apply generalization if provided
        if generalization_rules:
            df_anonymized = self.generalize_data(generalization_rules)

        # Check current anonymity level
        anonymity_check = self.check_k_anonymity(k)

        if not anonymity_check['k_anonymity_satisfied']:
            print(f"Dataset does not satisfy {k}-anonymity. {anonymity_check['groups_below_k']} groups have < {k} records.")

            # Apply suppression to violating records
            suppression_mask = self.create_suppression_mask(k)
            df_anonymized = df_anonymized[~suppression_mask].copy()

            print(f"Suppressed {suppression_mask.sum()} records to achieve {k}-anonymity.")
            print(f"Remaining records: {len(df_anonymized)}")

        return df_anonymized

# Usage example
# anonymizer = DataAnonymizer(df, quasi_identifiers=['age', 'zipcode', 'gender'])
#
# # Check current anonymity
# anonymity_check = anonymizer.check_k_anonymity(k=5)
# print(f"K-anonymity satisfied: {anonymity_check['k_anonymity_satisfied']}")
#
# # Apply generalization rules
# generalization_rules = {
#     'age': {'bins': [0, 18, 35, 55, 100], 'labels': ['<18', '18-34', '35-54', '55+']},
#     'zipcode': {'mapping': lambda x: str(x)[:3] + '**'}  # Mask last 2 digits
# }
#
# # Apply k-anonymity
# df_anonymized = anonymizer.apply_k_anonymity(k=5, generalization_rules=generalization_rules)
```

## 4. Regulatory Compliance and Governance

### 4.1 GDPR Compliance Framework

#### Data Protection Impact Assessment (DPIA)
```python
class GDPRComplianceChecker:
    """GDPR compliance assessment framework"""

    def __init__(self, organization_data: Dict[str, Any]):
        self.organization_data = organization_data

    def assess_data_processing(self) -> Dict[str, Any]:
        """Assess data processing activities for GDPR compliance"""

        assessment = {
            'lawful_basis': self._check_lawful_basis(),
            'data_minimization': self._check_data_minimization(),
            'purpose_limitation': self._check_purpose_limitation(),
            'storage_limitation': self._check_storage_limitation(),
            'accuracy': self._check_accuracy(),
            'integrity_security': self._check_integrity_security(),
            'accountability': self._check_accountability(),
            'international_transfers': self._check_international_transfers(),
            'data_subject_rights': self._check_data_subject_rights()
        }

        # Calculate compliance score
        compliant_items = sum(1 for item in assessment.values() if item['compliant'])
        total_items = len(assessment)
        compliance_score = compliant_items / total_items * 100

        assessment['overall_compliance'] = {
            'score': compliance_score,
            'compliant_principles': compliant_items,
            'total_principles': total_items,
            'status': 'Compliant' if compliance_score >= 80 else 'Needs Attention'
        }

        return assessment

    def _check_lawful_basis(self) -> Dict[str, Any]:
        """Check lawful basis for processing"""
        lawful_bases = self.organization_data.get('lawful_bases', [])

        required_bases = ['consent', 'contract', 'legal_obligation',
                         'vital_interests', 'public_task', 'legitimate_interests']

        has_valid_basis = any(basis in lawful_bases for basis in required_bases)

        return {
            'compliant': has_valid_basis,
            'details': f"Valid lawful bases: {lawful_bases}",
            'recommendation': 'Document lawful basis for each processing activity' if not has_valid_basis else None
        }

    def _check_data_minimization(self) -> Dict[str, Any]:
        """Check data minimization principle"""
        data_retention = self.organization_data.get('data_retention_days', 365*7)  # Default 7 years
        data
