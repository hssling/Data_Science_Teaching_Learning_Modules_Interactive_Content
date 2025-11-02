"""
Module 1: Introduction to Data Science - Code Examples
====================================================

This file contains practical examples demonstrating basic data science concepts
using Python. These examples complement the theoretical content in the README.
"""

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== Module 1: Introduction to Data Science Examples ===\n")

# Example 1: Creating and exploring a simple dataset
print("1. Creating and Exploring a Simple Dataset")
print("-" * 50)

# Create sample employee data
employee_data = {
    'Employee_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Wilson',
             'Frank Miller', 'Grace Lee', 'Henry Davis', 'Ivy Chen', 'Jack Taylor'],
    'Department': ['Engineering', 'Sales', 'Marketing', 'Engineering', 'HR',
                   'Sales', 'Engineering', 'Marketing', 'HR', 'Sales'],
    'Age': [28, 35, 42, 31, 29, 38, 33, 41, 27, 36],
    'Salary': [75000, 65000, 60000, 80000, 55000, 70000, 85000, 58000, 52000, 72000],
    'Years_Experience': [3, 8, 15, 5, 2, 10, 7, 12, 1, 9],
    'Performance_Rating': [4.2, 3.8, 4.5, 4.7, 3.9, 4.0, 4.8, 3.7, 4.1, 4.3]
}

# Create DataFrame
df = pd.DataFrame(employee_data)

# Display basic information
print("Dataset Overview:")
print(df.head())
print(f"\nDataset Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Basic statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Example 2: Data Types and Information
print("\n2. Data Types and Information")
print("-" * 35)

print("Data Types:")
print(df.dtypes)

print("\nMissing Values Check:")
print(df.isnull().sum())

# Example 3: Basic Data Visualization
print("\n3. Basic Data Visualizations")
print("-" * 32)

# Set up the plotting area
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Employee Data Analysis - Module 1 Introduction', fontsize=16, fontweight='bold')

# Salary distribution
axes[0, 0].hist(df['Salary'], bins=8, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Salary Distribution')
axes[0, 0].set_xlabel('Salary ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(df['Salary'].mean(), color='red', linestyle='--', label=f'Mean: ${df["Salary"].mean():.0f}')
axes[0, 0].legend()

# Age vs Salary scatter plot
axes[0, 1].scatter(df['Age'], df['Salary'], alpha=0.7, s=100)
axes[0, 1].set_title('Age vs Salary')
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Salary ($)')

# Department distribution
department_counts = df['Department'].value_counts()
axes[1, 0].bar(department_counts.index, department_counts.values, alpha=0.7)
axes[1, 0].set_title('Employees by Department')
axes[1, 0].set_xlabel('Department')
axes[1, 0].set_ylabel('Number of Employees')
axes[1, 0].tick_params(axis='x', rotation=45)

# Experience vs Performance
axes[1, 1].scatter(df['Years_Experience'], df['Performance_Rating'], alpha=0.7, s=100)
axes[1, 1].set_title('Experience vs Performance Rating')
axes[1, 1].set_xlabel('Years of Experience')
axes[1, 1].set_ylabel('Performance Rating')

plt.tight_layout()
plt.savefig('employee_data_analysis.png', dpi=300, bbox_inches='tight')
print("Visualizations saved as 'employee_data_analysis.png'")

# Example 4: Basic Statistical Analysis
print("\n4. Basic Statistical Analysis")
print("-" * 33)

print("Correlation Matrix:")
correlation_matrix = df[['Age', 'Salary', 'Years_Experience', 'Performance_Rating']].corr()
print(correlation_matrix)

# Create correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap - Employee Data')
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Correlation heatmap saved as 'correlation_heatmap.png'")

# Example 5: Department-wise Analysis
print("\n5. Department-wise Analysis")
print("-" * 30)

department_stats = df.groupby('Department').agg({
    'Salary': ['mean', 'median', 'std', 'count'],
    'Age': 'mean',
    'Performance_Rating': 'mean'
}).round(2)

print("Department Statistics:")
print(department_stats)

# Example 6: Simple Predictive Analysis (Linear Relationship)
print("\n6. Simple Predictive Analysis")
print("-" * 32)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Simple linear regression: Experience vs Salary
X = df[['Years_Experience']]
y = df['Salary']

model = LinearRegression()
model.fit(X, y)

# Predictions
predictions = model.predict(X)

print(f"Linear Regression: Experience vs Salary")
print(f"Coefficient: ${model.coef_[0]:.2f} per year of experience")
print(f"Intercept: ${model.intercept_:.2f}")
print(f"R² Score: {r2_score(y, predictions):.3f}")

# Plot regression line
plt.figure(figsize=(10, 6))
plt.scatter(df['Years_Experience'], df['Salary'], alpha=0.7, s=100, label='Actual Data')
plt.plot(df['Years_Experience'], predictions, color='red', linewidth=2, label='Regression Line')
plt.title('Experience vs Salary with Linear Regression')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('salary_regression.png', dpi=300, bbox_inches='tight')
print("Regression plot saved as 'salary_regression.png'")

# Example 7: Data Science Workflow Demonstration
print("\n7. Data Science Workflow Demonstration")
print("-" * 42)

def demonstrate_data_science_workflow():
    """
    Demonstrates the complete data science workflow on a simple example
    """
    print("Step 1: Problem Definition")
    print("   - Predict employee salary based on experience and performance")

    print("\nStep 2: Data Collection")
    print("   - Data already collected (our employee dataset)")

    print("\nStep 3: Data Preparation")
    print("   - Data is already clean, but let's check for any preprocessing needed")

    # Check for outliers using IQR method
    Q1 = df['Salary'].quantile(0.25)
    Q3 = df['Salary'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df['Salary'] < (Q1 - 1.5 * IQR)) | (df['Salary'] > (Q3 + 1.5 * IQR))]
    print(f"   - Outliers detected: {len(outliers)}")

    print("\nStep 4: Exploratory Data Analysis")
    print("   - Basic statistics computed above")
    print("   - Visualizations created")

    print("\nStep 5: Modeling")
    print("   - Simple linear regression performed")

    print("\nStep 6: Evaluation")
    print(".3f")

    print("\nStep 7: Communication")
    print("   - Results visualized and saved as images")
    print("   - Key insights: Experience strongly correlates with salary")

demonstrate_data_science_workflow()

# Example 8: Key Insights Summary
print("\n8. Key Insights from Analysis")
print("-" * 31)

insights = [
    f"Average employee age: {df['Age'].mean():.1f} years",
    f"Average salary: ${df['Salary'].mean():.0f}",
    f"Highest paid department: {df.groupby('Department')['Salary'].mean().idxmax()}",
    f"Experience-Salary correlation: {df['Years_Experience'].corr(df['Salary']):.3f}",
    f"Top performer: {df.loc[df['Performance_Rating'].idxmax(), 'Name']} (Rating: {df['Performance_Rating'].max()})",
    f"Youngest employee: {df.loc[df['Age'].idxmin(), 'Name']} ({df['Age'].min()} years old)"
]

for insight in insights:
    print(f"• {insight}")

print("\n=== End of Module 1 Examples ===")
print("Generated files:")
print("- employee_data_analysis.png")
print("- correlation_heatmap.png")
print("- salary_regression.png")
