"""
Module 1: Introduction to Data Science - Exercises
=================================================

This file contains practical exercises to reinforce the concepts learned
in Module 1: Introduction to Data Science. Complete each exercise and
compare your solutions with the provided answer key.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Set style for visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== Module 1: Introduction to Data Science - Exercises ===\n")

# =============================================================================
# EXERCISE 1: Data Science Fundamentals
# =============================================================================

print("Exercise 1: Data Science Fundamentals")
print("=" * 40)

"""
Question 1.1: Define data science in your own words (1-2 sentences)
Answer: [Your answer here]

Question 1.2: List the three main components of the data science Venn diagram
Answer: [Your answer here]

Question 1.3: What is the main difference between data science and data analytics?
Answer: [Your answer here]

Question 1.4: Name three industries where data science has significant applications
Answer: [Your answer here]

Question 1.5: List the 7 main steps in the data science workflow
Answer: [Your answer here]
"""

# =============================================================================
# EXERCISE 2: Basic Data Exploration
# =============================================================================

print("\nExercise 2: Basic Data Exploration")
print("=" * 35)

# Create a sample dataset for exercises
sales_data = {
    'Product_ID': range(1, 21),
    'Product_Name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Printer',
                     'Tablet', 'Smartphone', 'Headphones', 'Webcam', 'Router',
                     'External HDD', 'USB Drive', 'Graphics Card', 'RAM', 'SSD',
                     'Power Supply', 'Case', 'Cooler', 'Microphone', 'Speakers'],
    'Category': ['Computer', 'Accessory', 'Accessory', 'Display', 'Printer',
                 'Tablet', 'Phone', 'Audio', 'Camera', 'Network',
                 'Storage', 'Storage', 'Component', 'Component', 'Storage',
                 'Component', 'Component', 'Cooling', 'Audio', 'Audio'],
    'Price': [1200, 25, 75, 300, 150, 400, 800, 100, 80, 120,
              90, 20, 500, 80, 120, 70, 60, 40, 60, 50],
    'Units_Sold': [45, 120, 85, 30, 25, 60, 90, 110, 40, 35,
                   55, 200, 15, 75, 65, 45, 30, 95, 70, 85],
    'Rating': [4.5, 4.2, 4.0, 4.3, 3.8, 4.4, 4.6, 4.1, 3.9, 4.0,
               4.2, 4.3, 4.7, 4.4, 4.5, 3.9, 4.1, 4.0, 4.2, 4.3]
}

df_sales = pd.DataFrame(sales_data)

print("Dataset Overview:")
print(df_sales.head(10))

# Question 2.1: Display basic information about the dataset
print("\nQuestion 2.1: Basic dataset information")
# TODO: Display the shape, columns, and data types of the dataset
print("Shape:", df_sales.shape)
print("Columns:", list(df_sales.columns))
print("Data types:")
print(df_sales.dtypes)

# Question 2.2: Calculate basic statistics
print("\nQuestion 2.2: Basic statistics")
# TODO: Calculate mean, median, min, max for Price and Units_Sold
price_stats = df_sales['Price'].describe()
units_stats = df_sales['Units_Sold'].describe()
print("Price Statistics:")
print(price_stats[['mean', '50%', 'min', 'max']])
print("\nUnits Sold Statistics:")
print(units_stats[['mean', '50%', 'min', 'max']])

# Question 2.3: Check for missing values
print("\nQuestion 2.3: Missing values check")
# TODO: Check for missing values in each column
print("Missing values per column:")
print(df_sales.isnull().sum())

# =============================================================================
# EXERCISE 3: Data Visualization
# =============================================================================

print("\nExercise 3: Data Visualization")
print("=" * 30)

# Question 3.1: Create a histogram of product prices
print("Question 3.1: Price distribution histogram")
plt.figure(figsize=(10, 6))
plt.hist(df_sales['Price'], bins=10, edgecolor='black', alpha=0.7)
plt.title('Distribution of Product Prices')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.axvline(df_sales['Price'].mean(), color='red', linestyle='--', label=f'Mean: ${df_sales["Price"].mean():.2f}')
plt.legend()
plt.savefig('exercise_3_1_price_histogram.png', dpi=300, bbox_inches='tight')
plt.show()
print("Histogram saved as 'exercise_3_1_price_histogram.png'")

# Question 3.2: Create a scatter plot of Price vs Rating
print("\nQuestion 3.2: Price vs Rating scatter plot")
plt.figure(figsize=(10, 6))
plt.scatter(df_sales['Price'], df_sales['Rating'], alpha=0.7, s=100)
plt.title('Product Price vs Customer Rating')
plt.xlabel('Price ($)')
plt.ylabel('Rating (out of 5)')
plt.grid(True, alpha=0.3)
plt.savefig('exercise_3_2_price_rating_scatter.png', dpi=300, bbox_inches='tight')
plt.show()
print("Scatter plot saved as 'exercise_3_2_price_rating_scatter.png'")

# Question 3.3: Create a bar chart of units sold by category
print("\nQuestion 3.3: Units sold by category")
category_sales = df_sales.groupby('Category')['Units_Sold'].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
category_sales.plot(kind='bar', alpha=0.7)
plt.title('Total Units Sold by Category')
plt.xlabel('Category')
plt.ylabel('Total Units Sold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('exercise_3_3_category_sales.png', dpi=300, bbox_inches='tight')
plt.show()
print("Bar chart saved as 'exercise_3_3_category_sales.png'")

# =============================================================================
# EXERCISE 4: Statistical Analysis
# =============================================================================

print("\nExercise 4: Statistical Analysis")
print("=" * 30)

# Question 4.1: Calculate correlation matrix
print("Question 4.1: Correlation analysis")
correlation_matrix = df_sales[['Price', 'Units_Sold', 'Rating']].corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Create correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap - Sales Data')
plt.savefig('exercise_4_1_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("Heatmap saved as 'exercise_4_1_correlation_heatmap.png'")

# Question 4.2: Category-wise analysis
print("\nQuestion 4.2: Category-wise statistics")
category_stats = df_sales.groupby('Category').agg({
    'Price': ['mean', 'median', 'std'],
    'Units_Sold': ['sum', 'mean'],
    'Rating': 'mean'
}).round(2)
print("Category Statistics:")
print(category_stats)

# Question 4.3: Top products analysis
print("\nQuestion 4.3: Top products analysis")
# Top 5 products by units sold
top_sold = df_sales.nlargest(5, 'Units_Sold')[['Product_Name', 'Units_Sold', 'Price', 'Rating']]
print("Top 5 Products by Units Sold:")
print(top_sold)

# Top 5 products by revenue (Price * Units_Sold)
df_sales['Revenue'] = df_sales['Price'] * df_sales['Units_Sold']
top_revenue = df_sales.nlargest(5, 'Revenue')[['Product_Name', 'Revenue', 'Units_Sold', 'Price']]
print("\nTop 5 Products by Revenue:")
print(top_revenue)

# =============================================================================
# EXERCISE 5: Simple Predictive Modeling
# =============================================================================

print("\nExercise 5: Simple Predictive Modeling")
print("=" * 35)

# Question 5.1: Linear regression - Price vs Units Sold
print("Question 5.1: Linear regression analysis")
X = df_sales[['Price']]
y = df_sales['Units_Sold']

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

print(f"Linear Regression Results:")
print(f"Coefficient: {model.coef_[0]:.4f} (units sold change per $1 price increase)")
print(f"Intercept: {model.intercept_:.2f}")
print(f"R² Score: {r2_score(y, predictions):.3f}")

# Visualize the regression
plt.figure(figsize=(10, 6))
plt.scatter(df_sales['Price'], df_sales['Units_Sold'], alpha=0.7, s=100, label='Actual Data')
plt.plot(df_sales['Price'], predictions, color='red', linewidth=2, label='Regression Line')
plt.title('Price vs Units Sold - Linear Regression')
plt.xlabel('Price ($)')
plt.ylabel('Units Sold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('exercise_5_1_price_regression.png', dpi=300, bbox_inches='tight')
plt.show()
print("Regression plot saved as 'exercise_5_1_price_regression.png'")

# Question 5.2: Model interpretation
print("\nQuestion 5.2: Model interpretation")
print("Interpret the linear regression results:")
print("- What does the coefficient tell us about the relationship between price and sales?")
print("- What does the R² score indicate about the model's performance?")
print("- Based on this model, should we lower prices to increase sales?")

# =============================================================================
# EXERCISE 6: Data Science Workflow Application
# =============================================================================

print("\nExercise 6: Data Science Workflow Application")
print("=" * 45)

def analyze_product_performance():
    """
    Apply the complete data science workflow to analyze product performance
    """
    print("Applying Data Science Workflow to Product Performance Analysis:")

    print("\n1. Problem Definition:")
    print("   - Analyze which products are performing well and identify improvement opportunities")

    print("\n2. Data Collection:")
    print("   - Using the sales dataset with product information")

    print("\n3. Data Preparation:")
    print("   - Data is already clean and structured")

    print("\n4. Exploratory Data Analysis:")
    print("   - Key findings:")
    print(f"   - Total products: {len(df_sales)}")
    print(f"   - Average price: ${df_sales['Price'].mean():.2f}")
    print(f"   - Total units sold: {df_sales['Units_Sold'].sum()}")
    print(f"   - Average rating: {df_sales['Rating'].mean():.2f}")
    print(f"   - Best performing category: {df_sales.groupby('Category')['Units_Sold'].sum().idxmax()}")

    print("\n5. Modeling:")
    print("   - Performed linear regression on price vs sales")

    print("\n6. Evaluation:")
    print("   - Model explains {:.1f}% of variance in sales".format(r2_score(y, predictions) * 100))

    print("\n7. Communication:")
    print("   - Created visualizations and statistical summaries")
    print("   - Identified key insights for business decisions")

analyze_product_performance()

# =============================================================================
# EXERCISE 7: Critical Thinking Questions
# =============================================================================

print("\nExercise 7: Critical Thinking Questions")
print("=" * 38)

"""
Question 7.1: Business Insights
Based on your analysis, what recommendations would you give to improve sales?

Question 7.2: Data Quality
What additional data would be helpful for better analysis?

Question 7.3: Limitations
What are the limitations of this analysis?

Question 7.4: Next Steps
What would be your next steps if you were working on this project in a real company?

Question 7.5: Ethics
What ethical considerations should be kept in mind when analyzing sales data?
"""

# =============================================================================
# ANSWER KEY (Uncomment to see solutions)
# =============================================================================

"""
ANSWER KEY - EXERCISE 1
1.1: Data science is an interdisciplinary field that uses scientific methods,
     processes, algorithms, and systems to extract knowledge and insights from
     structured and unstructured data.

1.2: 1) Hacking Skills (Programming & Data Manipulation)
     2) Math & Statistics Knowledge (Statistical Analysis & Modeling)
     3) Substantive Expertise (Domain Knowledge & Communication)

1.3: Data analytics focuses on analyzing existing data to answer specific questions,
     while data science involves the entire process from data collection to model
     deployment and includes predictive modeling.

1.4: Healthcare, Finance, Retail/E-commerce, Technology, Transportation

1.5: 1) Problem Definition, 2) Data Collection, 3) Data Preparation,
     4) Exploratory Data Analysis, 5) Modeling, 6) Deployment & Monitoring,
     7) Communication
"""

print("\n=== End of Module 1 Exercises ===")
print("Generated visualization files:")
print("- exercise_3_1_price_histogram.png")
print("- exercise_3_2_price_rating_scatter.png")
print("- exercise_3_3_category_sales.png")
print("- exercise_4_1_correlation_heatmap.png")
print("- exercise_5_1_price_regression.png")

print("\nKey Takeaways:")
print("- Explored basic data manipulation with pandas")
print("- Created various types of visualizations")
print("- Performed statistical analysis and correlation studies")
print("- Built and evaluated a simple predictive model")
print("- Applied the complete data science workflow")
