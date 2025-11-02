# Module 3: Programming Foundations

## Overview
This module provides a comprehensive introduction to the programming languages and tools essential for data science. You'll learn Python (the primary language for data science), R (powerful for statistical analysis), SQL (for database querying), and Git (for version control). These tools form the foundation for implementing data science concepts in practice.

## Learning Objectives
By the end of this module, you will be able to:
- Write efficient Python code for data manipulation and analysis
- Use R for statistical computing and visualization
- Query databases using SQL
- Manage code versions with Git
- Set up and use integrated development environments
- Follow best practices for data science programming

## 1. Python Programming

### 1.1 Python Basics

#### Data Types and Variables
```python
# Basic data types
integer_var = 42
float_var = 3.14159
string_var = "Hello, Data Science!"
boolean_var = True

# Collections
list_var = [1, 2, 3, 4, 5]
tuple_var = (1, 2, 3)
dict_var = {'key': 'value', 'name': 'Alice'}
set_var = {1, 2, 3, 4}
```

#### Control Structures
```python
# Conditional statements
if condition:
    # do something
elif another_condition:
    # do something else
else:
    # default action

# Loops
for item in iterable:
    # process item

while condition:
    # repeat while condition is true

# List comprehensions (Pythonic way)
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]
```

### 1.2 Functions and Modules

#### Defining Functions
```python
def calculate_mean(numbers):
    """Calculate the arithmetic mean of a list of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

def process_data(data, operation='mean'):
    """Process data with specified operation."""
    if operation == 'mean':
        return calculate_mean(data)
    elif operation == 'sum':
        return sum(data)
    elif operation == 'count':
        return len(data)
    else:
        raise ValueError("Unsupported operation")
```

#### Working with Modules
```python
# Importing modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Creating custom modules
# Save functions in data_utils.py
# Then import: from data_utils import calculate_mean, process_data
```

### 1.3 Error Handling and Debugging

#### Exception Handling
```python
try:
    # Code that might raise an exception
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")
    result = float('inf')
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    # Always execute this block
    print("Operation completed")
```

#### Debugging Techniques
```python
# Using print statements for debugging
def debug_function(x):
    print(f"Input: {x}")
    result = x * 2
    print(f"Result: {result}")
    return result

# Using assertions
def validate_data(data):
    assert isinstance(data, list), "Data must be a list"
    assert len(data) > 0, "Data cannot be empty"
    assert all(isinstance(x, (int, float)) for x in data), "All elements must be numeric"
    return True
```

### 1.4 File Input/Output

#### Reading and Writing Files
```python
# Reading text files
with open('data.txt', 'r') as file:
    content = file.read()
    lines = file.readlines()

# Writing to files
with open('output.txt', 'w') as file:
    file.write("Hello, World!\n")
    file.writelines(["Line 1\n", "Line 2\n"])

# Working with CSV files
import csv

# Reading CSV
with open('data.csv', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)
    data = list(reader)

# Writing CSV
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Age', 'City'])
    writer.writerows([['Alice', 25, 'NYC'], ['Bob', 30, 'LA']])
```

## 2. Data Manipulation with Python

### 2.1 NumPy Arrays

#### Array Creation and Operations
```python
import numpy as np

# Creating arrays
arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
random_arr = np.random.rand(3, 3)

# Array operations
arr_sum = arr1d + 10
arr_product = arr1d * 2
dot_product = np.dot(arr1d, arr1d)
matrix_mult = np.dot(arr2d, arr2d.T)
```

#### Array Indexing and Slicing
```python
# Basic indexing
first_element = arr1d[0]
last_element = arr1d[-1]

# Slicing
first_three = arr1d[:3]
middle = arr1d[1:4]
every_other = arr1d[::2]

# 2D array indexing
element = arr2d[0, 1]  # Row 0, Column 1
first_row = arr2d[0, :]
first_column = arr2d[:, 0]
submatrix = arr2d[0:2, 1:3]
```

#### Array Functions
```python
# Statistical functions
mean_val = np.mean(arr1d)
std_val = np.std(arr1d)
min_val = np.min(arr1d)
max_val = np.max(arr1d)

# Mathematical functions
sqrt_arr = np.sqrt(arr1d)
exp_arr = np.exp(arr1d)
log_arr = np.log(arr1d + 1)  # Add 1 to avoid log(0)

# Reshaping
reshaped = arr1d.reshape(5, 1)
flattened = arr2d.flatten()
```

### 2.2 Pandas DataFrames

#### Creating and Loading Data
```python
import pandas as pd

# Creating DataFrame from dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['NYC', 'LA', 'Chicago']
}
df = pd.DataFrame(data)

# Loading data from files
csv_df = pd.read_csv('data.csv')
excel_df = pd.read_excel('data.xlsx')
json_df = pd.read_json('data.json')
```

#### Data Exploration
```python
# Basic information
print(df.head())  # First 5 rows
print(df.tail())  # Last 5 rows
print(df.info())  # Data types and non-null counts
print(df.describe())  # Statistical summary

# Shape and columns
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Data types
print(df.dtypes)

# Missing values
print(df.isnull().sum())
```

#### Data Selection and Filtering
```python
# Selecting columns
names = df['Name']
multiple_cols = df[['Name', 'Age']]

# Selecting rows by position
first_row = df.iloc[0]
first_three_rows = df.iloc[0:3]

# Selecting by label
row_by_index = df.loc[0]
rows_by_condition = df.loc[df['Age'] > 25]

# Boolean filtering
young_people = df[df['Age'] < 30]
nyc_residents = df[df['City'] == 'NYC']
```

#### Data Manipulation
```python
# Adding new columns
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 100],
                        labels=['Young', 'Middle', 'Senior'])

# Modifying existing columns
df['Name_Upper'] = df['Name'].str.upper()

# Grouping and aggregation
grouped = df.groupby('City').agg({
    'Age': ['mean', 'min', 'max'],
    'Name': 'count'
})

# Sorting
sorted_df = df.sort_values('Age', ascending=False)

# Merging DataFrames
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['A', 'B', 'C']})
df2 = pd.DataFrame({'ID': [1, 2, 4], 'Score': [85, 90, 88]})
merged = pd.merge(df1, df2, on='ID', how='inner')
```

#### Handling Missing Data
```python
# Detecting missing values
missing_counts = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

# Dropping missing values
df_clean = df.dropna()  # Drop rows with any NaN
df_clean_cols = df.dropna(axis=1)  # Drop columns with any NaN

# Filling missing values
df_filled = df.fillna(0)  # Fill with 0
df_filled_mean = df.fillna(df.mean())  # Fill with mean
df_forward_fill = df.fillna(method='ffill')  # Forward fill
df_interpolated = df.interpolate()  # Interpolate
```

## 3. R Programming

### 3.1 R Basics

#### Data Types and Structures
```r
# Basic data types
numeric_var <- 42.5
integer_var <- 42L
character_var <- "Hello, R!"
logical_var <- TRUE

# Vectors
numeric_vector <- c(1, 2, 3, 4, 5)
character_vector <- c("apple", "banana", "cherry")
logical_vector <- c(TRUE, FALSE, TRUE)

# Matrices
matrix_data <- matrix(1:9, nrow = 3, ncol = 3)
matrix_by_row <- matrix(1:9, nrow = 3, byrow = TRUE)

# Data Frames
df <- data.frame(
  Name = c("Alice", "Bob", "Charlie"),
  Age = c(25, 30, 35),
  City = c("NYC", "LA", "Chicago")
)

# Lists
my_list <- list(
  numbers = c(1, 2, 3),
  text = "Hello",
  dataframe = df
)
```

#### Control Structures
```r
# Conditional statements
if (condition) {
  # do something
} else if (another_condition) {
  # do something else
} else {
  # default action
}

# Loops
for (item in vector) {
  # process item
}

while (condition) {
  # repeat while condition is true
}

# Apply functions (vectorized operations)
squares <- sapply(1:10, function(x) x^2)
```

### 3.2 Data Manipulation with dplyr and tidyr

#### Loading and Installing Packages
```r
# Installing packages
install.packages("dplyr")
install.packages("tidyr")
install.packages("ggplot2")

# Loading packages
library(dplyr)
library(tidyr)
library(ggplot2)
```

#### Data Manipulation with dplyr
```r
# Loading sample data
data(mtcars)

# Selecting columns
selected <- select(mtcars, mpg, cyl, hp)
excluded <- select(mtcars, -cyl)

# Filtering rows
filtered <- filter(mtcars, mpg > 20, cyl == 4)

# Creating new columns
mutated <- mutate(mtcars,
                  mpg_per_hp = mpg / hp,
                  efficiency = ifelse(mpg > median(mpg), "High", "Low"))

# Grouping and summarizing
summarized <- mtcars %>%
  group_by(cyl) %>%
  summarise(
    avg_mpg = mean(mpg),
    max_hp = max(hp),
    count = n()
  )

# Arranging (sorting)
arranged <- arrange(mtcars, desc(mpg))
```

#### Data Reshaping with tidyr
```r
# Sample wide data
wide_data <- data.frame(
  student = c("Alice", "Bob"),
  math_2020 = c(85, 78),
  math_2021 = c(88, 82),
  science_2020 = c(92, 75),
  science_2021 = c(90, 80)
)

# Gather (wide to long)
long_data <- gather(wide_data,
                   key = "subject_year",
                   value = "score",
                   -student)

# Separate columns
separated <- separate(long_data,
                     subject_year,
                     into = c("subject", "year"),
                     sep = "_")

# Spread (long to wide)
wide_again <- spread(separated,
                    key = subject,
                    value = score)
```

### 3.3 Statistical Analysis in R

#### Basic Statistics
```r
# Summary statistics
summary(mtcars$mpg)
mean(mtcars$mpg)
median(mtcars$mpg)
sd(mtcars$mpg)  # Standard deviation
var(mtcars$mpg) # Variance
min(mtcars$mpg)
max(mtcars$mpg)
quantile(mtcars$mpg)

# Correlation
cor(mtcars$mpg, mtcars$hp)
cor_matrix <- cor(mtcars[, c("mpg", "hp", "wt", "qsec")])
```

#### Linear Regression
```r
# Simple linear regression
model <- lm(mpg ~ hp, data = mtcars)
summary(model)

# Multiple regression
multi_model <- lm(mpg ~ hp + wt + cyl, data = mtcars)
summary(multi_model)

# Model diagnostics
plot(multi_model)
```

## 4. SQL for Data Analysis

### 4.1 Basic SQL Queries

#### SELECT Statements
```sql
-- Select all columns
SELECT * FROM customers;

-- Select specific columns
SELECT customer_id, first_name, last_name FROM customers;

-- Select with aliases
SELECT customer_id AS id, first_name AS fname FROM customers;

-- Select distinct values
SELECT DISTINCT country FROM customers;
```

#### Filtering Data
```sql
-- WHERE clause
SELECT * FROM products WHERE price > 50;
SELECT * FROM customers WHERE country = 'USA';

-- Multiple conditions
SELECT * FROM orders
WHERE order_date >= '2023-01-01' AND total_amount > 100;

-- IN operator
SELECT * FROM products WHERE category IN ('Electronics', 'Books');

-- LIKE operator (pattern matching)
SELECT * FROM customers WHERE first_name LIKE 'A%';
SELECT * FROM customers WHERE email LIKE '%@gmail.com';

-- NULL checks
SELECT * FROM customers WHERE phone IS NULL;
SELECT * FROM customers WHERE phone IS NOT NULL;
```

#### Sorting and Limiting
```sql
-- ORDER BY
SELECT * FROM products ORDER BY price DESC;
SELECT * FROM customers ORDER BY last_name, first_name;

-- LIMIT (number of rows)
SELECT * FROM products ORDER BY price DESC LIMIT 10;

-- OFFSET (skip rows)
SELECT * FROM products ORDER BY price DESC LIMIT 10 OFFSET 20;
```

### 4.2 Aggregation and Grouping

#### Aggregate Functions
```sql
-- COUNT
SELECT COUNT(*) FROM customers;
SELECT COUNT(DISTINCT country) FROM customers;

-- SUM, AVG, MIN, MAX
SELECT SUM(total_amount) FROM orders;
SELECT AVG(price) FROM products;
SELECT MIN(price), MAX(price) FROM products;

-- Multiple aggregates
SELECT
    COUNT(*) as total_orders,
    SUM(total_amount) as total_revenue,
    AVG(total_amount) as avg_order_value,
    MIN(order_date) as first_order,
    MAX(order_date) as last_order
FROM orders;
```

#### GROUP BY Clause
```sql
-- Group by single column
SELECT country, COUNT(*) as customer_count
FROM customers
GROUP BY country
ORDER BY customer_count DESC;

-- Group by multiple columns
SELECT country, city, COUNT(*) as customer_count
FROM customers
GROUP BY country, city
ORDER BY country, customer_count DESC;

-- HAVING clause (filter groups)
SELECT country, COUNT(*) as customer_count
FROM customers
GROUP BY country
HAVING COUNT(*) > 10
ORDER BY customer_count DESC;
```

### 4.3 Joins and Relationships

#### INNER JOIN
```sql
-- Basic inner join
SELECT
    o.order_id,
    c.first_name,
    c.last_name,
    o.order_date,
    o.total_amount
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id;

-- Multiple joins
SELECT
    o.order_id,
    c.first_name,
    p.product_name,
    oi.quantity,
    oi.unit_price
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
INNER JOIN order_items oi ON o.order_id = oi.order_id
INNER JOIN products p ON oi.product_id = p.product_id;
```

#### LEFT JOIN
```sql
-- Customers with their orders (including customers with no orders)
SELECT
    c.customer_id,
    c.first_name,
    c.last_name,
    COUNT(o.order_id) as order_count,
    COALESCE(SUM(o.total_amount), 0) as total_spent
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name;
```

#### Subqueries
```sql
-- Subquery in WHERE clause
SELECT * FROM products
WHERE price > (SELECT AVG(price) FROM products);

-- Subquery in FROM clause
SELECT
    category,
    AVG(avg_price) as overall_avg_price
FROM (
    SELECT category, AVG(price) as avg_price
    FROM products
    GROUP BY category
) category_avgs
GROUP BY category;
```

### 4.4 Advanced SQL Concepts

#### Window Functions
```sql
-- ROW_NUMBER, RANK, DENSE_RANK
SELECT
    product_name,
    price,
    ROW_NUMBER() OVER (ORDER BY price DESC) as row_num,
    RANK() OVER (ORDER BY price DESC) as rank,
    DENSE_RANK() OVER (ORDER BY price DESC) as dense_rank
FROM products;

-- Running totals and moving averages
SELECT
    order_date,
    total_amount,
    SUM(total_amount) OVER (ORDER BY order_date) as running_total,
    AVG(total_amount) OVER (ORDER BY order_date ROWS 2 PRECEDING) as moving_avg_3
FROM orders;
```

#### Common Table Expressions (CTEs)
```sql
WITH monthly_sales AS (
    SELECT
        DATE_TRUNC('month', order_date) as month,
        SUM(total_amount) as monthly_revenue,
        COUNT(*) as order_count
    FROM orders
    GROUP BY DATE_TRUNC('month', order_date)
),
monthly_growth AS (
    SELECT
        month,
        monthly_revenue,
        LAG(monthly_revenue) OVER (ORDER BY month) as prev_month_revenue,
        (monthly_revenue - LAG(monthly_revenue) OVER (ORDER BY month)) /
        LAG(monthly_revenue) OVER (ORDER BY month) * 100 as growth_pct
    FROM monthly_sales
)
SELECT * FROM monthly_growth
ORDER BY month;
```

## 5. Version Control with Git

### 5.1 Git Basics

#### Repository Initialization
```bash
# Initialize a new repository
git init

# Clone an existing repository
git clone https://github.com/user/repo.git

# Check repository status
git status
```

#### Basic Workflow
```bash
# Add files to staging area
git add filename.py
git add .  # Add all files

# Commit changes
git commit -m "Add data analysis script"

# View commit history
git log
git log --oneline
```

#### Branching
```bash
# Create and switch to new branch
git checkout -b feature-branch

# List branches
git branch

# Switch between branches
git checkout main
git checkout feature-branch

# Merge branches
git checkout main
git merge feature-branch
```

### 5.2 Collaboration

#### Remote Repositories
```bash
# Add remote repository
git remote add origin https://github.com/user/repo.git

# Push to remote
git push origin main

# Pull from remote
git pull origin main

# Fetch without merging
git fetch origin
```

#### Handling Merge Conflicts
```bash
# Check for conflicts
git status

# Edit conflicted files to resolve conflicts
# Remove conflict markers (<<<<<<<, =======, >>>>>>>)

# Add resolved files
git add resolved_file.py

# Complete merge
git commit
```

### 5.3 Best Practices

#### Commit Messages
```bash
# Good commit messages
git commit -m "Fix bug in data preprocessing function"
git commit -m "Add feature: customer segmentation analysis"
git commit -m "Update documentation for API endpoints"

# Bad commit messages
git commit -m "fix"
git commit -m "changes"
git commit -m "update"
```

#### .gitignore File
```
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/

# Data files
*.csv
*.xlsx
*.json
data/
models/

# Jupyter Notebook
.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

## 6. Development Environments

### 6.1 Jupyter Notebook

#### Basic Usage
```python
# Install Jupyter
pip install jupyter

# Launch Jupyter Notebook
jupyter notebook

# Launch JupyterLab (modern interface)
pip install jupyterlab
jupyter lab
```

#### Notebook Features
- **Cells**: Code, Markdown, and Raw cells
- **Kernel**: Python interpreter running in background
- **Magic Commands**: Special commands for enhanced functionality

```python
# Useful magic commands
%matplotlib inline  # Display plots inline
%timeit sum(range(1000))  # Time execution
%%time  # Time entire cell
# Multi-line code here

# List all variables
%whos

# Run external Python files
%run script.py
```

### 6.2 Integrated Development Environments

#### VS Code for Data Science
- **Extensions**: Python, Jupyter, GitLens, Excel Viewer
- **Features**: IntelliSense, debugging, Git integration
- **Data Science Tools**: Variable explorer, plot viewer

#### RStudio
- **Console**: R interpreter
- **Source Editor**: Script writing with syntax highlighting
- **Environment/History**: Variable and command history
- **Files/Plots/Packages/Help**: Integrated panes

## 7. Best Practices

### 7.1 Code Style and Documentation

#### Python PEP 8
```python
# Good naming conventions
def calculate_mean(values):  # snake_case for functions
    """Calculate arithmetic mean of values."""  # Docstrings
    if not values:  # Clear conditional
        return 0.0
    return sum(values) / len(values)

class DataProcessor:  # PascalCase for classes
    def __init__(self, data):
        self.data = data
        self._private_var = None  # Private variables with underscore
```

#### R Style Guide
```r
# Function definition
calculate_mean <- function(values) {
  # Input validation
  if (length(values) == 0) {
    return(0)
  }

  # Calculation
  mean_value <- sum(values) / length(values)
  return(mean_value)
}
```

### 7.2 Project Organization

#### Directory Structure
```
data_science_project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── modeling.py
│   └── visualization.py
├── models/
│   └── saved_models/
├── reports/
│   └── figures/
├── requirements.txt
├── README.md
└── .gitignore
```

### 7.3 Performance Optimization

#### Vectorized Operations (Python)
```python
import numpy as np

# Inefficient (loop-based)
def slow_sum(arr):
    total = 0
    for x in arr:
        total += x
    return total

# Efficient (vectorized)
def fast_sum(arr):
    return np.sum(arr)  # Uses optimized C code
```

#### Memory Management
```python
# Process large files in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process chunk
    process_data(chunk)

# Use appropriate data types
df['category'] = df['category'].astype('category')  # Memory efficient for strings
df['small_int'] = df['small_int'].astype('int8')    # Smaller integer types
```

## 8. Assessment

### Quiz Questions
1. What are the main differences between Python lists and NumPy arrays?
2. How do you handle missing values in a pandas DataFrame?
3. Explain the difference between INNER JOIN and LEFT JOIN in SQL.
4. What is the purpose of the Git staging area?
5. How do you create a new branch and switch to it in Git?

### Practical Exercises
1. Write a Python function to calculate statistical measures (mean, median, mode)
2. Use pandas to clean and preprocess a dataset with missing values
3. Write SQL queries to analyze sales data from multiple tables
4. Create a Git repository and demonstrate branching workflow
5. Build a data analysis script that combines Python, pandas, and visualization

## 9. Resources

### Python
- "Python for Data Analysis" by Wes McKinney
- "Automate the Boring Stuff with Python" by Al Sweigart
- Python Documentation: https://docs.python.org/

### R
- "R for Data Science" by Hadley Wickham
- "Advanced R" by Hadley Wickham
- R Documentation: https://cran.r-project.org/

### SQL
- "SQL for Data Scientists" by Renee M. P. Teate
- SQLZoo: https://sqlzoo.net/
- LeetCode SQL problems

### Git
- "Pro Git" by Scott Chacon
- Git Documentation: https://git-scm.com/doc
- GitHub Learning Lab

## Next Steps

Congratulations on mastering the programming foundations! You now have the tools to implement data science concepts in code. In the next module, we'll dive into data collection and storage techniques.

**Ready to continue?** Proceed to [Module 4: Data Collection and Storage](../04_data_collection_storage/)
