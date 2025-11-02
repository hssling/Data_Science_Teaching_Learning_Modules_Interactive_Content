# Module 7: Machine Learning

## Overview
Machine Learning is the heart of modern data science, enabling computers to learn patterns from data and make predictions without being explicitly programmed. This comprehensive module covers all major ML algorithms, from foundational concepts to advanced techniques, with practical implementations and real-world applications.

## Learning Objectives
By the end of this module, you will be able to:
- Understand the fundamentals of machine learning and its types
- Implement supervised learning algorithms (regression and classification)
- Apply unsupervised learning techniques (clustering and dimensionality reduction)
- Evaluate model performance using appropriate metrics
- Handle overfitting and underfitting through regularization and validation
- Deploy machine learning models in production environments
- Understand ethical considerations in machine learning

## 1. Introduction to Machine Learning

### 1.1 What is Machine Learning?

Machine Learning is a subset of artificial intelligence that enables systems to automatically learn and improve from experience without being explicitly programmed.

**Key Characteristics:**
- **Learning from Data**: Algorithms improve performance as they process more data
- **Pattern Recognition**: Identify patterns and relationships in data
- **Prediction**: Make informed predictions on new, unseen data
- **Adaptation**: Models can adapt to changing data patterns

### 1.2 Types of Machine Learning

#### Supervised Learning
- **Definition**: Learning from labeled training data
- **Goal**: Learn a mapping from inputs to outputs
- **Examples**: Classification, Regression
- **Algorithms**: Linear Regression, Logistic Regression, Decision Trees, Random Forest, SVM, Neural Networks

#### Unsupervised Learning
- **Definition**: Learning from unlabeled data
- **Goal**: Discover hidden patterns or structures
- **Examples**: Clustering, Dimensionality Reduction, Association Rules
- **Algorithms**: K-Means, Hierarchical Clustering, PCA, t-SNE, Apriori

#### Semi-Supervised Learning
- **Definition**: Learning from partially labeled data
- **Goal**: Combine supervised and unsupervised approaches
- **Use Cases**: When labeling is expensive but some labels exist

#### Reinforcement Learning
- **Definition**: Learning through interaction with environment
- **Goal**: Maximize cumulative reward
- **Examples**: Game playing, Robotics, Recommendation systems
- **Algorithms**: Q-Learning, Deep Q Networks, Policy Gradients

### 1.3 Machine Learning Workflow

1. **Problem Definition**: Clearly define the problem and success metrics
2. **Data Collection**: Gather relevant data from various sources
3. **Data Preprocessing**: Clean, normalize, and transform data
4. **Feature Engineering**: Create meaningful features from raw data
5. **Model Selection**: Choose appropriate algorithms for the problem
6. **Training**: Train models on prepared data
7. **Validation**: Evaluate model performance using cross-validation
8. **Hyperparameter Tuning**: Optimize model parameters
9. **Testing**: Evaluate final model on unseen test data
10. **Deployment**: Deploy model to production environment
11. **Monitoring**: Monitor model performance and retrain as needed

## 2. Supervised Learning - Regression

### 2.1 Linear Regression

#### Simple Linear Regression
**Model**: `y = β₀ + β₁x + ε`

Where:
- `y`: Dependent variable (target)
- `x`: Independent variable (feature)
- `β₀`: Intercept (bias term)
- `β₁`: Slope coefficient
- `ε`: Error term

#### Multiple Linear Regression
**Model**: `y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε`

#### Implementation
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
```

### 2.2 Polynomial Regression

#### Concept
Extends linear regression by adding polynomial terms to capture non-linear relationships.

**Model**: `y = β₀ + β₁x + β₂x² + β₃x³ + ... + ε`

#### Implementation
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Create polynomial features
poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(X)

# Create pipeline
model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
model.fit(X_train, y_train)
```

### 2.3 Regularization Techniques

#### Ridge Regression (L2 Regularization)
**Objective**: `minimize Σ(yᵢ - ŷᵢ)² + λΣβⱼ²`

- Adds penalty for large coefficients
- Prevents overfitting
- All features are kept but coefficients are shrunk

#### Lasso Regression (L1 Regularization)
**Objective**: `minimize Σ(yᵢ - ŷᵢ)² + λΣ|βⱼ|`

- Can force some coefficients to exactly zero
- Performs feature selection
- Good for sparse solutions

#### Elastic Net
**Objective**: `minimize Σ(yᵢ - ŷᵢ)² + λ₁Σ|βⱼ| + λ₂Σβⱼ²`

- Combines L1 and L2 regularization
- Benefits of both Lasso and Ridge

#### Implementation
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Elastic Net
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)
```

## 3. Supervised Learning - Classification

### 3.1 Logistic Regression

#### Binary Classification
**Model**: `P(y=1|x) = 1 / (1 + e^(-(β₀ + β₁x₁ + ... + βₚxₚ)))`

- Uses sigmoid function to map predictions to probabilities
- Decision boundary at 0.5 probability
- Can be extended to multi-class using One-vs-Rest or Softmax

#### Implementation
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create and train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```

### 3.2 Decision Trees

#### How Decision Trees Work
- **Root Node**: Starting point with entire dataset
- **Internal Nodes**: Decision points based on feature values
- **Leaf Nodes**: Final predictions (class labels for classification)
- **Splitting Criteria**: Gini impurity, entropy, or variance reduction

#### Advantages
- Easy to interpret and visualize
- Can handle both numerical and categorical data
- No need for feature scaling
- Can capture non-linear relationships

#### Disadvantages
- Prone to overfitting
- Can be unstable (small changes in data can result in different trees)
- Bias towards features with more categories

#### Implementation
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Create and train model
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Visualize tree
plt.figure(figsize=(20,10))
plot_tree(dt_model, feature_names=X.columns, class_names=['Class_0', 'Class_1'], filled=True)
plt.savefig('decision_tree.png')
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(feature_importance)
```

### 3.3 Random Forest

#### Ensemble Learning
- **Bagging**: Bootstrap Aggregation
- **Random Forest**: Ensemble of decision trees with random feature selection

#### How It Works
1. Create multiple bootstrap samples from training data
2. Train decision tree on each sample
3. For each tree, randomly select subset of features for splitting
4. Average predictions (regression) or majority vote (classification)

#### Advantages
- Reduced overfitting compared to single decision trees
- Handles missing values well
- Provides feature importance estimates
- Parallelizable training

#### Implementation
```python
from sklearn.ensemble import RandomForestClassifier

# Create and train model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Important Features:")
print(feature_importance.head(10))
```

### 3.4 Support Vector Machines (SVM)

#### Maximum Margin Classifier
- **Hyperplane**: Decision boundary that separates classes
- **Support Vectors**: Data points closest to the hyperplane
- **Margin**: Distance between hyperplane and closest support vectors
- **Goal**: Maximize the margin while correctly classifying training data

#### Kernel Trick
- **Linear Kernel**: For linearly separable data
- **Polynomial Kernel**: For polynomial decision boundaries
- **Radial Basis Function (RBF)**: For complex, non-linear boundaries
- **Sigmoid Kernel**: For neural network-like behavior

#### Implementation
```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Create SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# Hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
```

### 3.5 K-Nearest Neighbors (KNN)

#### Algorithm
1. Choose value of K (number of neighbors)
2. Calculate distance between new point and all training points
3. Find K nearest neighbors
4. Assign class based on majority vote (classification) or average (regression)

#### Distance Metrics
- **Euclidean Distance**: Straight-line distance
- **Manhattan Distance**: Sum of absolute differences
- **Minkowski Distance**: Generalized distance metric
- **Hamming Distance**: For categorical variables

#### Implementation
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Scale features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model
knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_model.fit(X_train_scaled, y_train)

# Find optimal K
k_values = range(1, 21)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    cv_scores.append(scores.mean())

# Plot K vs accuracy
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('K Value')
plt.ylabel('Cross-Validation Accuracy')
plt.title('KNN: K Value vs Accuracy')
plt.grid(True)
plt.savefig('knn_k_selection.png')
plt.show()
```

## 4. Unsupervised Learning

### 4.1 K-Means Clustering

#### Algorithm
1. **Initialization**: Choose K initial centroids randomly
2. **Assignment**: Assign each point to nearest centroid
3. **Update**: Calculate new centroids as mean of points in each cluster
4. **Repeat**: Steps 2-3 until convergence or max iterations

#### Choosing K
- **Elbow Method**: Plot sum of squared distances vs K
- **Silhouette Score**: Measure cluster cohesion and separation
- **Gap Statistic**: Compare within-cluster dispersion to reference distribution

#### Implementation
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Determine optimal K using elbow method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot elbow curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(K_range, inertias, marker='o')
ax1.set_xlabel('Number of clusters (K)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')
ax1.grid(True)

ax2.plot(K_range, silhouette_scores, marker='o', color='orange')
ax2.set_xlabel('Number of clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')
ax2.grid(True)

plt.tight_layout()
plt.savefig('kmeans_elbow_silhouette.png')
plt.show()

# Fit final model
optimal_k = 4  # Based on elbow and silhouette analysis
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)

# Visualize clusters (for 2D data)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1],
           s=300, c='red', marker='X', label='Centroids')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.title(f'K-Means Clustering (K={optimal_k})')
plt.colorbar(scatter)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('kmeans_clusters.png')
plt.show()
```

### 4.2 Hierarchical Clustering

#### Agglomerative Clustering
- **Bottom-up approach**: Start with individual points as clusters
- **Merge Strategy**: Combine closest clusters iteratively
- **Linkage Methods**:
  - **Single Linkage**: Distance between closest points
  - **Complete Linkage**: Distance between farthest points
  - **Average Linkage**: Average distance between all points
  - **Ward's Method**: Minimize within-cluster variance

#### Dendrogram
- Tree-like diagram showing merge history
- Height represents distance between merged clusters
- Cutting dendrogram at specific height gives clusters

#### Implementation
```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering

# Create linkage matrix
linkage_matrix = linkage(X_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.savefig('hierarchical_dendrogram.png')
plt.show()

# Perform clustering
n_clusters = 4
hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
clusters = hierarchical.fit_predict(X_scaled)

# Compare with K-means
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# K-means result
ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_clusters, cmap='viridis', alpha=0.6)
ax1.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1],
           s=200, c='red', marker='X')
ax1.set_title('K-Means Clustering')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

# Hierarchical result
scatter = ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='plasma', alpha=0.6)
ax2.set_title('Hierarchical Clustering')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')

plt.tight_layout()
plt.savefig('clustering_comparison.png')
plt.show()
```

### 4.3 Principal Component Analysis (PCA)

#### Dimensionality Reduction
- **Goal**: Reduce number of features while preserving variance
- **Method**: Find principal components (directions of maximum variance)
- **Principal Components**: Orthogonal eigenvectors of covariance matrix

#### Steps
1. **Standardize data**: Mean = 0, variance = 1
2. **Compute covariance matrix**
3. **Find eigenvalues and eigenvectors**
4. **Sort by eigenvalues** (explained variance)
5. **Select top K components**
6. **Project data onto new subspace**

#### Implementation
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Plot explained variance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Individual Explained Variance')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-', alpha=0.7)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
plt.legend()

plt.tight_layout()
plt.savefig('pca_variance_explained.png')
plt.show()

# Determine optimal number of components (95% variance)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components for 95% variance: {n_components_95}")

# Apply PCA with optimal components
pca_optimal = PCA(n_components=n_components_95)
X_pca_optimal = pca_optimal.fit_transform(X_scaled)

print(f"Original dimensions: {X.shape}")
print(f"Reduced dimensions: {X_pca_optimal.shape}")
print(f"Variance preserved: {cumulative_variance[n_components_95-1]:.3f}")
```

## 5. Model Evaluation and Validation

### 5.1 Cross-Validation Techniques

#### K-Fold Cross-Validation
- **Process**: Split data into K folds, train on K-1 folds, validate on remaining fold
- **Repeat K times**, each fold used once for validation
- **Final score**: Average performance across all folds

#### Stratified K-Fold
- **Purpose**: Maintain class distribution in each fold
- **Important for**: Imbalanced classification datasets

#### Implementation
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import make_scorer

# K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Stratified K-Fold for classification
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_stratified = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')

print(f"Stratified CV scores: {cv_scores_stratified}")
print(f"Mean stratified CV score: {cv_scores_stratified.mean():.4f}")

# Multiple scoring metrics
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
for metric in scoring_metrics:
    scores = cross_val_score(model, X, y, cv=skfold, scoring=metric)
    print(f"{metric.capitalize()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

### 5.2 Classification Metrics

#### Confusion Matrix
```
Predicted: 0    Predicted: 1
Actual: 0    TN          FP
Actual: 1    FN          TP
```

#### Key Metrics
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP) - Positive Predictive Value
- **Recall**: TP / (TP + FN) - True Positive Rate, Sensitivity
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Specificity**: TN / (TN + FP) - True Negative Rate

#### ROC Curve and AUC
- **ROC Curve**: Plot of TPR vs FPR at different thresholds
- **AUC**: Area under ROC curve (0.5 = random, 1.0 = perfect)

#### Implementation
```python
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import seaborn as sns

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig('roc_curve.png')
plt.show()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True, alpha=0.3)
plt.savefig('precision_recall_curve.png')
plt.show()
```

### 5.3 Regression Metrics

#### Common Metrics
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual
- **Mean Squared Error (MSE)**: Average squared difference between predicted and actual
- **Root Mean Squared Error (RMSE)**: Square root of MSE (same units as target)
- **R² Score**: Proportion of variance explained by model
- **Mean Absolute Percentage Error (MAPE)**: Average absolute percentage error

#### Implementation
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_metrics(y_true, y_pred, model_name="Model"):
    """Calculate and display regression metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"{model_name} Performance Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

# Calculate metrics
metrics = regression_metrics(y_test, y_pred, "Linear Regression")

# Residual analysis
residuals = y_test - y_pred

plt.figure(figsize=(12, 4))

# Residuals vs Fitted
plt.subplot(1, 3, 1)
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.grid(True, alpha=0.3)

# Q-Q plot for normality
plt.subplot(1, 3, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.grid(True, alpha=0.3)

# Residual histogram
plt.subplot(1, 3, 3)
plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residual_analysis.png')
plt.show()
```

## 6. Hyperparameter Tuning

### 6.1 Grid Search
- **Method**: Exhaustive search over specified parameter values
- **Pros**: Finds optimal parameters, thorough
- **Cons**: Computationally expensive for large parameter spaces

### 6.2 Random Search
- **Method**: Random sampling from parameter distributions
- **Pros**: More efficient than grid search, can find good parameters faster
- **Cons**: May miss optimal parameters

### 6.3 Bayesian Optimization
- **Method**: Uses probabilistic model to find optimal parameters
- **Pros**: Efficient, considers past evaluations
- **Cons**: More complex to implement

#### Implementation
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint

# Define parameter grids
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Grid Search
print("Performing Grid Search...")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters (Grid Search): {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Random Search
param_distributions = {
    'n_estimators': randint(50, 200),
    'max_depth': [None] + list(range(10, 31)),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['auto', 'sqrt', 'log2']
}

print("\nPerforming Random Search...")
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=50,  # Number of parameter settings sampled
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print(f"Best parameters (Random Search): {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.4f}")

# Compare with default model
default_model = RandomForestClassifier(random_state=42)
default_scores = cross_val_score(default_model, X_train, y_train, cv=5)
default_score = default_scores.mean()

print("
Model Comparison:")
print(f"Default model CV score: {default_score:.4f}")
print(f"Grid search CV score: {grid_search.best_score_:.4f}")
print(f"Random search CV score: {random_search.best_score_:.4f}")

# Use best model for final evaluation
best_model = random_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("
Final model performance on test set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_best))
```

## 7. Model Deployment and Production

### 7.1 Model Serialization
```python
import joblib
import pickle

# Save model using joblib (recommended for scikit-learn)
joblib.dump(best_model, 'best_model.joblib')

# Save model using pickle
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Load model
loaded_model = joblib.load('best_model.joblib')

# Make predictions with loaded model
new_predictions = loaded_model.predict(new_data)
```

### 7.2 Creating a REST API
```python
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('best_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()

        # Convert to numpy array
        features = np.array(data['features']).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        # Return response
        response = {
            'prediction': int(prediction),
            'probability': probability.tolist(),
            'model_version': '1.0.0'
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### 7.3 Model Monitoring and Maintenance

#### Key Metrics to Monitor
- **Model Performance**: Accuracy, precision, recall over time
- **Data Drift**: Changes in data distribution
- **Prediction Latency**: Response time requirements
- **Error Rates**: Unexpected failures or edge cases

#### Retraining Strategy
- **Scheduled Retraining**: Regular intervals (daily, weekly, monthly)
- **Performance-Based**: When accuracy drops below threshold
- **Data-Based**: When significant new data becomes available

## 8. Ethical Considerations in Machine Learning

### 8.1 Bias and Fairness
- **Selection Bias**: Non-representative training data
- **Label Bias**: Incorrect or biased labels
- **Algorithmic Bias**: Discriminatory decision-making

### 8.2 Transparency and Explainability
- **Black Box Models**: Difficult to interpret predictions
- **Model Interpretability**: Understanding how predictions are made
- **Stakeholder Communication**: Explaining model decisions

### 8.3 Privacy and Security
- **Data Privacy**: Protecting sensitive information
- **Model Inversion Attacks**: Reconstructing training data
- **Adversarial Examples**: Manipulating model inputs

### 8.4 Best Practices
- **Diverse Data Collection**: Representative and inclusive datasets
- **Bias Audits**: Regular assessment of model fairness
- **Explainable AI**: Using interpretable models when possible
- **Ethical Review**: Human oversight of AI systems

## 9. Practical Applications and Case Studies

### 9.1 Customer Churn Prediction
- **Problem**: Predict which customers are likely to churn
- **Data**: Customer demographics, usage patterns, billing history
- **Models**: Logistic Regression, Random Forest, Gradient Boosting
- **Impact**: Targeted retention campaigns, reduced churn rate

### 9.2 Fraud Detection
- **Problem**: Identify fraudulent transactions
- **Data**: Transaction details, user behavior, historical patterns
- **Models**: Isolation Forest, Autoencoders, Ensemble methods
- **Impact**: Reduced financial losses, improved security

### 9.3 Recommendation Systems
- **Problem**: Suggest relevant products to users
- **Data**: User-item interactions, ratings, browsing history
- **Models**: Collaborative Filtering, Matrix Factorization, Neural Networks
- **Impact**: Increased engagement and sales

### 9.4 Image Classification
- **Problem**: Automatically classify images into categories
- **Data**: Labeled images from various categories
- **Models**: Convolutional Neural Networks (CNNs)
- **Impact**: Automated content moderation, medical diagnosis

## 10. Resources and Further Reading

### Books
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "Elements of Statistical Learning" by Hastie, Tibshirani, Friedman

### Online Courses
- Coursera: Andrew Ng's Machine Learning
- Coursera: Deep Learning Specialization
- edX: Microsoft Professional Program in Data Science

### Research Papers
- "A Few Useful Things to Know about Machine Learning" by Pedro Domingos
- "Why Should I Trust You?" Explaining the Predictions of Any Classifier
- "The Fairness of Machine Learning in Criminal Justice"

### Tools and Libraries
- **scikit-learn**: Core ML library for Python
- **TensorFlow/Keras**: Deep learning framework
- **PyTorch**: Deep learning framework
- **XGBoost/LightGBM**: Gradient boosting libraries
- **MLflow**: ML lifecycle management

## Next Steps

Congratulations on mastering machine learning fundamentals! You now have the skills to build and deploy ML models. In the next module, we'll explore deep learning and neural networks for more complex problems.

**Ready to continue?** Proceed to [Module 8: Deep Learning](../08_deep_learning/)
