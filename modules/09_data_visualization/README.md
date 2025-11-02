# Module 9: Data Visualization

## Overview
Data visualization is the art and science of communicating insights from data through visual representations. This comprehensive module covers everything from basic plots to advanced interactive dashboards, teaching you how to create compelling visualizations that effectively communicate complex data insights to various audiences.

## Learning Objectives
By the end of this module, you will be able to:
- Create effective static visualizations using matplotlib and seaborn
- Build interactive visualizations with plotly and bokeh
- Design comprehensive dashboards for data exploration
- Apply data visualization best practices and principles
- Choose appropriate visualization types for different data types
- Create publication-ready visualizations
- Understand color theory and visual perception
- Communicate data insights effectively to stakeholders

## 1. Introduction to Data Visualization

### 1.1 Why Visualization Matters

#### The Power of Visual Communication
- **Human Brain Processing**: 90% of information transmitted to the brain is visual
- **Pattern Recognition**: Visual patterns are processed 60,000 times faster than text
- **Memory Retention**: People remember 80% of what they see vs 20% of what they read
- **Decision Making**: Visual data leads to faster and more accurate decisions

#### Goals of Data Visualization
- **Explore**: Understand data distributions and relationships
- **Explain**: Communicate findings clearly to others
- **Persuade**: Convince stakeholders with compelling evidence
- **Discover**: Uncover hidden patterns and insights

### 1.2 Visualization Types and When to Use Them

#### Comparison Visualizations
- **Bar Charts**: Compare categories or discrete values
- **Column Charts**: Similar to bar charts, vertical orientation
- **Line Charts**: Show trends over time or continuous variables
- **Slope Charts**: Show changes between two time points

#### Distribution Visualizations
- **Histograms**: Show distribution of continuous variables
- **Box Plots**: Display quartiles and outliers
- **Violin Plots**: Show distribution density
- **Density Plots**: Smooth representation of distributions

#### Relationship Visualizations
- **Scatter Plots**: Show relationships between two continuous variables
- **Bubble Charts**: Add third dimension with bubble size
- **Heatmaps**: Show correlations or matrix data
- **Pair Plots**: Show relationships between multiple variables

#### Composition Visualizations
- **Pie Charts**: Show parts of a whole (use sparingly)
- **Stacked Bar Charts**: Show composition within categories
- **Treemaps**: Show hierarchical data
- **Sunburst Charts**: Radial representation of hierarchies

#### Time Series Visualizations
- **Line Charts**: Standard for time series
- **Area Charts**: Emphasize magnitude
- **Candlestick Charts**: Financial data
- **Calendar Heatmaps**: Show patterns over time

## 2. Matplotlib: The Foundation

### 2.1 Basic Plotting

#### Figure and Axes
```python
import matplotlib.pyplot as plt
import numpy as np

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Simple line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y, label='sin(x)')

# Customize plot
ax.set_title('Sine Wave', fontsize=16, fontweight='bold')
ax.set_xlabel('x values')
ax.set_ylabel('sin(x)')
ax.legend()
ax.grid(True, alpha=0.3)

# Save plot
plt.savefig('sine_wave.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Multiple Subplots
```python
# Create multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Multiple Plots Example', fontsize=16, fontweight='bold')

# Plot 1: Line plot
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x), 'b-', linewidth=2)
axes[0, 0].set_title('Sine Wave')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Scatter plot
x_scatter = np.random.normal(0, 1, 100)
y_scatter = np.random.normal(0, 1, 100)
axes[0, 1].scatter(x_scatter, y_scatter, alpha=0.6, c='red', s=50)
axes[0, 1].set_title('Random Scatter')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Histogram
data = np.random.normal(0, 1, 1000)
axes[1, 0].hist(data, bins=30, alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Normal Distribution')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Bar chart
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]
axes[1, 1].bar(categories, values, alpha=0.7, color='green')
axes[1, 1].set_title('Bar Chart')
axes[1, 1].set_xlabel('Categories')
axes[1, 1].set_ylabel('Values')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multiple_subplots.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 2.2 Advanced Matplotlib Features

#### Customizing Colors and Styles
```python
# Color options
colors = ['red', 'blue', 'green', 'orange', 'purple']
linestyles = ['-', '--', '-.', ':', '-']
markers = ['o', 's', '^', 'D', 'v']

# Custom color maps
cmap = plt.cm.viridis
colors_from_cmap = cmap(np.linspace(0, 1, 10))

# Style sheets
plt.style.use('seaborn-v0_8-darkgrid')
# Available styles: 'default', 'classic', 'bmh', 'dark_background', etc.
```

#### Annotations and Text
```python
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
y = x**2
ax.plot(x, y, 'b-', linewidth=2)

# Add annotations
ax.annotate('Maximum point', xy=(10, 100), xytext=(7, 80),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=12, ha='center')

# Add text
ax.text(2, 20, 'y = x²', fontsize=14, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

# Add vertical and horizontal lines
ax.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='x = 5')
ax.axhline(y=25, color='green', linestyle='--', alpha=0.7, label='y = 25')

ax.set_title('Annotated Plot', fontsize=16, fontweight='bold')
ax.legend()
plt.savefig('annotated_plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 3. Seaborn: Statistical Visualization

### 3.1 Distribution Plots

#### Histograms and Density Plots
```python
import seaborn as sns
import pandas as pd
from scipy import stats

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Load sample data
tips = sns.load_dataset("tips")

# Histogram with KDE
plt.figure(figsize=(10, 6))
sns.histplot(data=tips, x='total_bill', kde=True, bins=30, alpha=0.7)
plt.title('Distribution of Total Bill Amounts', fontsize=16, fontweight='bold')
plt.xlabel('Total Bill ($)')
plt.ylabel('Frequency')
plt.savefig('histogram_kde.png', dpi=300, bbox_inches='tight')
plt.show()

# Multiple distributions
plt.figure(figsize=(10, 6))
sns.histplot(data=tips, x='total_bill', hue='sex', kde=True, alpha=0.6)
plt.title('Total Bill Distribution by Gender', fontsize=16, fontweight='bold')
plt.xlabel('Total Bill ($)')
plt.ylabel('Frequency')
plt.savefig('histogram_by_category.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Box Plots and Violin Plots
```python
# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=tips, x='day', y='total_bill', palette='Set3')
plt.title('Total Bill Distribution by Day', fontsize=16, fontweight='bold')
plt.xlabel('Day')
plt.ylabel('Total Bill ($)')
plt.savefig('boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=tips, x='day', y='total_bill', palette='Set2', inner='quartile')
plt.title('Total Bill Distribution by Day (Violin Plot)', fontsize=16, fontweight='bold')
plt.xlabel('Day')
plt.ylabel('Total Bill ($)')
plt.savefig('violinplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Combined box and violin
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=tips, x='day', y='total_bill', palette='Set3')
plt.title('Box Plot')

plt.subplot(1, 2, 2)
sns.violinplot(data=tips, x='day', y='total_bill', palette='Set3')
plt.title('Violin Plot')

plt.tight_layout()
plt.savefig('box_violin_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 3.2 Relationship Plots

#### Scatter Plots
```python
# Basic scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=tips, x='total_bill', y='tip', alpha=0.6, s=80)
plt.title('Total Bill vs Tip Amount', fontsize=16, fontweight='bold')
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.savefig('scatter_basic.png', dpi=300, bbox_inches='tight')
plt.show()

# Scatter plot with hue and size
plt.figure(figsize=(10, 6))
sns.scatterplot(data=tips, x='total_bill', y='tip',
                hue='day', size='size', sizes=(50, 200), alpha=0.7)
plt.title('Total Bill vs Tip (by Day and Party Size)', fontsize=16, fontweight='bold')
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.savefig('scatter_hue_size.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Regression Plots
```python
# Linear regression plot
plt.figure(figsize=(10, 6))
sns.regplot(data=tips, x='total_bill', y='tip',
            scatter_kws={'alpha':0.6, 's':60},
            line_kws={'color':'red', 'linewidth':2})
plt.title('Total Bill vs Tip with Regression Line', fontsize=16, fontweight='bold')
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.savefig('regression_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Multiple regression plots
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.regplot(data=tips, x='total_bill', y='tip')
plt.title('All Data')

plt.subplot(1, 3, 2)
sns.regplot(data=tips[tips['smoker']=='Yes'], x='total_bill', y='tip', color='red')
plt.title('Smokers')

plt.subplot(1, 3, 3)
sns.regplot(data=tips[tips['smoker']=='No'], x='total_bill', y='tip', color='blue')
plt.title('Non-Smokers')

plt.tight_layout()
plt.savefig('multiple_regression.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 3.3 Categorical Plots

#### Bar Plots
```python
# Bar plot with error bars
plt.figure(figsize=(10, 6))
sns.barplot(data=tips, x='day', y='total_bill',
            estimator=np.mean, errorbar=('ci', 95), capsize=0.1)
plt.title('Average Total Bill by Day', fontsize=16, fontweight='bold')
plt.xlabel('Day')
plt.ylabel('Average Total Bill ($)')
plt.savefig('barplot_with_error.png', dpi=300, bbox_inches='tight')
plt.show()

# Grouped bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=tips, x='day', y='total_bill', hue='sex', palette='Set2')
plt.title('Average Total Bill by Day and Gender', fontsize=16, fontweight='bold')
plt.xlabel('Day')
plt.ylabel('Average Total Bill ($)')
plt.legend(title='Gender')
plt.savefig('grouped_barplot.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Count Plots
```python
# Count plot
plt.figure(figsize=(10, 6))
sns.countplot(data=tips, x='day', palette='Set3', order=['Thur', 'Fri', 'Sat', 'Sun'])
plt.title('Number of Observations by Day', fontsize=16, fontweight='bold')
plt.xlabel('Day')
plt.ylabel('Count')
plt.savefig('countplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Stacked count plot
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=tips, x='day', hue='smoker', palette='Set1')
plt.title('Observations by Day and Smoking Status', fontsize=16, fontweight='bold')
plt.xlabel('Day')
plt.ylabel('Count')
plt.legend(title='Smoker')
plt.savefig('stacked_countplot.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 3.4 Matrix and Correlation Plots

#### Heatmaps
```python
# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = tips.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Tips Dataset', fontsize=16, fontweight='bold')
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Custom heatmap
# Create sample data
data = np.random.rand(10, 10)
mask = np.triu(np.ones_like(data, dtype=bool))  # Upper triangle mask

plt.figure(figsize=(10, 8))
sns.heatmap(data, mask=mask, cmap='YlGnBu', annot=True, fmt='.2f',
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Custom Heatmap with Mask', fontsize=16, fontweight='bold')
plt.savefig('custom_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Pair Plots
```python
# Pair plot
numeric_cols = ['total_bill', 'tip', 'size']
plt.figure(figsize=(10, 8))
pair_plot = sns.pairplot(tips[numeric_cols], diag_kind='kde', plot_kws={'alpha':0.6})
pair_plot.fig.suptitle('Pair Plot of Numeric Variables', fontsize=16, fontweight='bold', y=1.02)
plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Pair plot with hue
plt.figure(figsize=(10, 8))
pair_plot_hue = sns.pairplot(tips, vars=numeric_cols, hue='sex', palette='Set1',
                            diag_kind='hist', plot_kws={'alpha':0.6})
pair_plot_hue.fig.suptitle('Pair Plot by Gender', fontsize=16, fontweight='bold', y=1.02)
plt.savefig('pairplot_hue.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 4. Plotly: Interactive Visualizations

### 4.1 Basic Interactive Plots

#### Interactive Line and Scatter Plots
```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load sample data
df_stocks = px.data.stocks()

# Interactive line plot
fig = px.line(df_stocks, x='date', y=['GOOG', 'AAPL', 'AMZN', 'FB', 'NFLX', 'MSFT'],
              title='Stock Prices Over Time')
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Stock Price',
    legend_title='Company'
)
fig.write_html('interactive_stocks.html')
fig.show()

# Interactive scatter plot
df_iris = px.data.iris()
fig = px.scatter(df_iris, x='sepal_width', y='sepal_length',
                 color='species', size='petal_length',
                 hover_data=['petal_width'],
                 title='Iris Dataset - Sepal Dimensions')
fig.update_layout(
    xaxis_title='Sepal Width',
    yaxis_title='Sepal Length'
)
fig.write_html('interactive_scatter.html')
fig.show()
```

#### 3D Plots
```python
# 3D scatter plot
fig = px.scatter_3d(df_iris, x='sepal_length', y='sepal_width', z='petal_length',
                    color='species', size='petal_width',
                    title='3D Iris Dataset Visualization')
fig.update_layout(
    scene=dict(
        xaxis_title='Sepal Length',
        yaxis_title='Sepal Width',
        zaxis_title='Petal Length'
    )
)
fig.write_html('3d_scatter.html')
fig.show()

# 3D surface plot
import numpy as np

x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])
fig.update_layout(
    title='3D Surface Plot: sin(√(x² + y²))',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    )
)
fig.write_html('3d_surface.html')
fig.show()
```

### 4.2 Advanced Interactive Visualizations

#### Animated Plots
```python
# Animated scatter plot
df_gapminder = px.data.gapminder()

fig = px.scatter(df_gapminder, x='gdpPercap', y='lifeExp',
                 size='pop', color='continent', hover_name='country',
                 animation_frame='year', animation_group='country',
                 log_x=True, size_max=60,
                 range_x=[100, 100000], range_y=[25, 90],
                 title='Life Expectancy vs GDP Per Capita (Animated)')
fig.write_html('animated_scatter.html')
fig.show()

# Animated bar chart
fig = px.bar(df_gapminder, x='continent', y='pop', color='continent',
             animation_frame='year', animation_group='country',
             range_y=[0, 4000000000],
             title='Population by Continent Over Time')
fig.write_html('animated_bar.html')
fig.show()
```

#### Interactive Maps
```python
# Choropleth map
df_world = px.data.gapminder().query("year == 2007")

fig = px.choropleth(df_world, locations='iso_alpha', color='lifeExp',
                    hover_name='country', color_continuous_scale='Viridis',
                    title='Life Expectancy by Country (2007)')
fig.write_html('choropleth_map.html')
fig.show()

# Scatter plot on map
fig = px.scatter_geo(df_world, locations='iso_alpha', color='continent',
                     hover_name='country', size='pop',
                     projection='natural earth',
                     title='World Population by Country')
fig.write_html('scatter_map.html')
fig.show()
```

#### Dashboard-Style Layouts
```python
# Create subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Scatter Plot', 'Histogram', 'Box Plot', 'Violin Plot'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# Add scatter plot
fig.add_trace(
    go.Scatter(x=tips['total_bill'], y=tips['tip'], mode='markers',
               name='Tips', marker=dict(color='blue', size=8, opacity=0.6)),
    row=1, col=1
)

# Add histogram
fig.add_trace(
    go.Histogram(x=tips['total_bill'], name='Total Bill',
                 marker_color='lightblue', opacity=0.7),
    row=1, col=2
)

# Add box plot
fig.add_trace(
    go.Box(y=tips['total_bill'], name='Total Bill',
           marker_color='green', boxmean=True),
    row=2, col=1
)

# Add violin plot
fig.add_trace(
    go.Violin(y=tips['tip'], name='Tip',
              marker_color='orange', box_visible=True, meanline_visible=True),
    row=2, col=2
)

# Update layout
fig.update_layout(
    title_text="Tips Dataset - Multiple Visualizations",
    showlegend=False,
    height=800
)

# Update axis labels
fig.update_xaxes(title_text="Total Bill ($)", row=1, col=1)
fig.update_yaxes(title_text="Tip ($)", row=1, col=1)
fig.update_xaxes(title_text="Total Bill ($)", row=1, col=2)
fig.update_yaxes(title_text="Frequency", row=1, col=2)
fig.update_yaxes(title_text="Total Bill ($)", row=2, col=1)
fig.update_yaxes(title_text="Tip ($)", row=2, col=2)

fig.write_html('dashboard_layout.html')
fig.show()
```

## 5. Data Visualization Best Practices

### 5.1 Design Principles

#### The Four Pillars of Visualization
1. **Purpose**: Know why you're creating the visualization
2. **Audience**: Understand who will consume the visualization
3. **Data**: Ensure data quality and relevance
4. **Story**: Craft a compelling narrative

#### Chart Selection Guidelines
- **Bar Charts**: For comparing categories
- **Line Charts**: For showing trends over time
- **Scatter Plots**: For showing relationships between variables
- **Pie Charts**: Only for showing parts of a whole (≤ 5 categories)
- **Heatmaps**: For showing patterns in matrix data

### 5.2 Color Theory

#### Color Psychology
- **Red**: Danger, urgency, excitement
- **Blue**: Trust, calm, professionalism
- **Green**: Growth, success, nature
- **Yellow**: Optimism, attention, caution
- **Purple**: Luxury, creativity, wisdom

#### Color Accessibility
```python
# Colorblind-friendly palettes
colorblind_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# High contrast colors
high_contrast = ['#000000', '#FFFFFF', '#FF0000', '#00FF00', '#0000FF', '#FFFF00']

# Using color maps
import matplotlib.colors as mcolors

# Sequential colormap (good for ordered data)
sequential_cmap = plt.cm.Blues

# Diverging colormap (good for data with a meaningful center)
diverging_cmap = plt.cm.RdYlBu

# Qualitative colormap (good for categorical data)
qualitative_cmap = plt.cm.Set1
```

#### Color Usage Best Practices
```python
# Good color usage
plt.figure(figsize=(12, 5))

# Sequential colors for ordered data
plt.subplot(1, 3, 1)
data = [1, 2, 3, 4, 5]
colors_seq = plt.cm.Blues(np.linspace(0.3, 0.9, len(data)))
bars = plt.bar(range(len(data)), data, color=colors_seq)
plt.title('Sequential Colors')
plt.colorbar(plt.cm.ScalarMappable(cmap='Blues'), ax=plt.gca())

# Diverging colors for centered data
plt.subplot(1, 3, 2)
data_centered = [-2, -1, 0, 1, 2]
colors_div = plt.cm.RdYlBu_r(np.linspace(0, 1, len(data_centered)))
bars = plt.bar(range(len(data_centered)), data_centered, color=colors_div)
plt.title('Diverging Colors')
plt.colorbar(plt.cm.ScalarMappable(cmap='RdYlBu_r'), ax=plt.gca())

# Qualitative colors for categories
plt.subplot(1, 3, 3)
categories = ['A', 'B', 'C', 'D', 'E']
data_cat = [3, 7, 2, 8, 4]
colors_qual = plt.cm.Set1(np.linspace(0, 1, len(categories)))
bars = plt.bar(categories, data_cat, color=colors_qual[:len(categories)])
plt.title('Qualitative Colors')

plt.tight_layout()
plt.savefig('color_usage_examples.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 5.3 Typography and Layout

#### Font Selection
```python
# Good font practices
plt.figure(figsize=(10, 6))

# Title font
plt.title('Sales Performance Analysis', fontsize=18, fontweight='bold', pad=20)

# Axis labels
plt.xlabel('Month', fontsize=14, fontweight='medium')
plt.ylabel('Sales ($)', fontsize=14, fontweight='medium')

# Tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Legend
plt.legend(fontsize=12, loc='upper left')

# Grid and spines
plt.grid(True, alpha=0.3)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.savefig('typography_example.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Layout and Spacing
```python
# Good layout practices
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comprehensive Sales Dashboard', fontsize=20, fontweight='bold', y=0.95)

# Adjust spacing
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Plot 1: Main KPI
axes[0, 0].text(0.5, 0.5, 'Total Sales\n$1,234,567',
                fontsize=24, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightblue'))
axes[0, 0].set_title('Key Metric', fontsize=16, fontweight='bold')
axes[0, 0].set_xlim(0, 1)
axes[0, 0].set_ylim(0, 1)
axes[0, 0].axis('off')

# Plot 2: Trend line
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
sales = [100, 120, 140, 130, 160, 180]
axes[0, 1].plot(months, sales, marker='o', linewidth=3, markersize=8, color='green')
axes[0, 1].set_title('Sales Trend', fontsize=16, fontweight='bold')
axes[0, 1].set_ylabel('Sales ($K)')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Category breakdown
categories = ['Electronics', 'Clothing', 'Books', 'Home']
values = [45, 25, 20, 10]
axes[1, 0].bar(categories, values, color='skyblue', alpha=0.8)
axes[1, 0].set_title('Sales by Category', fontsize=16, fontweight='bold')
axes[1, 0].set_ylabel('Percentage (%)')
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Performance indicator
performance = 85
axes[1, 1].barh(['Performance'], [performance], color='orange', alpha=0.8)
axes[1, 1].axvline(x=80, color='red', linestyle='--', alpha=0.7, label='Target')
axes[1, 1].set_xlim(0, 100)
axes[1, 1].set_title('Performance Score', fontsize=16, fontweight='bold')
axes[1, 1].set_xlabel('Score (%)')
axes[1, 1].legend()

plt.savefig('dashboard_layout.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
```

## 6. Dashboard Creation

### 6.1 Streamlit Dashboards

#### Basic Streamlit App
```python
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Data Science Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Interactive Data Science Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("Dashboard Controls")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/sales_data.csv")

df = load_data()

# Filters
st.sidebar.subheader("Filters")
selected_category = st.sidebar.multiselect(
    "Select Categories",
    options=df['category'].unique(),
    default=df['category'].unique()
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df['date'].min(), df['date'].max()]
)

# Filter data
filtered_df = df[
    (df['category'].isin(selected_category)) &
    (df['date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
]

# Main content
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Sales", f"${filtered_df['sales'].sum():,.0f}")

with col2:
    st.metric("Average Order Value", f"${filtered_df['sales'].mean():,.2f}")

with col3:
    st.metric("Total Orders", f"{len(filtered_df):,}")

with col4:
    st.metric("Unique Customers", f"{filtered_df['customer_id'].nunique():,}")

st.markdown("---")

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales by Category")
    fig, ax = plt.subplots(figsize=(8, 6))
    category_sales = filtered_df.groupby('category')['sales'].sum()
    ax.pie(category_sales.values, labels=category_sales.index, autopct='%1.1f%%')
    ax.set_title("Sales Distribution by Category")
    st.pyplot(fig)

with col2:
    st.subheader("Sales Trend")
    fig, ax = plt.subplots(figsize=(8, 6))
    daily_sales = filtered_df.groupby('date')['sales'].sum()
    ax.plot(daily_sales.index, daily_sales.values, marker='o')
    ax.set_title("Daily Sales Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales ($)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Data table
st.subheader("Raw Data")
st.dataframe(filtered_df.head(100))

# Download button
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download filtered data as CSV",
    data=csv,
    file_name='filtered_sales_data.csv',
    mime='text/csv',
    key='download-csv'
)

# Run with: streamlit run app.py
```

### 6.2 Tableau/Public Dashboards

#### Tableau Features
- **Drag-and-drop interface** for creating visualizations
- **Live data connections** to various sources
- **Advanced calculations** and statistical functions
- **Dashboard interactivity** with filters and parameters
- **Sharing and collaboration** features

#### Dashboard Best Practices
1. **Clear hierarchy**: Most important insights at the top
2. **Consistent design**: Use same colors, fonts, and styles
3. **Logical flow**: Guide the viewer's eye through the story
4. **Performance**: Optimize for fast loading
5. **Mobile responsiveness**: Ensure usability on different devices

## 7. Advanced Visualization Techniques

### 7.1 Time Series Visualizations

#### Calendar Heatmaps
```python
import calmap
import pandas as pd

# Create sample time series data
dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
values = np.random.randn(len(dates)).cumsum() + 100

# Create DataFrame
df_calendar = pd.DataFrame({'date': dates, 'value': values})
df_calendar.set_index('date', inplace=True)

# Plot calendar heatmap
plt.figure(figsize=(16, 8))
calmap.calendarplot(df_calendar['value'], monthticks=3, daylabels='MTWTFSS',
                   dayticks=[0, 2, 4, 6], cmap='YlOrRd',
                   fillcolor='grey', linewidth=0.5,
                   fig_kws=dict(figsize=(16, 8)))
plt.title('Calendar Heatmap - Daily Values', fontsize=16, fontweight='bold')
plt.savefig('calendar_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Candlestick Charts (Financial Data)
```python
import mplfinance as mpf

# Sample financial data
data = {
    'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
    'Open': np.random.uniform(100, 110, 100),
    'High': np.random.uniform(105, 115, 100),
    'Low': np.random.uniform(95, 105, 100),
    'Close': np.random.uniform(100, 110, 100),
    'Volume': np.random.randint(1000, 10000, 100)
}

df_candlestick = pd.DataFrame(data)
df_candlestick.set_index('Date', inplace=True)

# Ensure OHLC order is correct
df_candlestick = df_candlestick[['Open', 'High', 'Low', 'Close', 'Volume']]

# Plot candlestick chart
fig, ax = mpf.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
mpf.plot(df_candlestick, type='candle', style='charles',
         volume=True, ylabel='Price ($)', ylabel_lower='Volume',
         ax=ax, warn_too_much_data=1000)
plt.title('Candlestick Chart with Volume', fontsize=16, fontweight='bold')
plt.savefig('candlestick_chart.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 7.2 Network Graphs

#### Basic Network Visualization
```python
import networkx as nx

# Create a sample network
G = nx.Graph()

# Add nodes
nodes = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank']
G.add_nodes_from(nodes)

# Add edges (connections)
edges = [('Alice', 'Bob'), ('Alice', 'Charlie'), ('Bob', 'Charlie'),
         ('Bob', 'Diana'), ('Charlie', 'Diana'), ('Diana', 'Eve'),
         ('Eve', 'Frank'), ('Charlie', 'Frank')]
G.add_edges_from(edges)

# Draw the network
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)  # Position nodes using spring layout

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue',
                      edgecolors='black', linewidths=2, alpha=0.8)

# Draw edges
nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color='gray')

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

plt.title('Social Network Graph', fontsize=16, fontweight='bold')
plt.axis('off')
plt.savefig('network_graph.png', dpi=300, bbox_inches='tight')
plt.show()

# Network metrics
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
print(f"Network density: {nx.density(G):.3f}")
```

### 7.3 Geospatial Visualizations

#### Choropleth Maps with Geopandas
```python
import geopandas as gpd
from shapely.geometry import Point, Polygon

# Create sample geospatial data
# Note: This requires actual shapefiles for real geographic data

# Sample approach for creating choropleth maps
# 1. Load shapefile data
# world = gpd.read_file('path/to/world_shapefile.shp')

# 2. Merge with your data
# world_data = world.merge(your_dataframe, on='country_column')

# 3. Create choropleth map
# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# world_data.plot(column='your_value_column', ax=ax,
#                 legend=True, cmap='YlOrRd',
#                 legend_kwds={'label': "Value",
#                             'orientation': "horizontal"})
# ax.set_title('Choropleth Map Example', fontsize=16, fontweight='bold')
# plt.savefig('choropleth_map.png', dpi=300, bbox_inches='tight')
# plt.show()
```

## 8. Visualization Tools Comparison

### 8.1 When to Use Each Tool

| Tool | Best For | Pros | Cons |
|------|----------|------|------|
| **Matplotlib** | Publication-quality static plots | Highly customizable, extensive control | Steep learning curve, verbose code |
| **Seaborn** | Statistical visualizations | Beautiful defaults, statistical functions | Less flexible than matplotlib |
| **Plotly** | Interactive web visualizations | Interactive, web-ready, 3D support | Can be slow with large datasets |
| **Tableau** | Business dashboards | User-friendly, powerful, no coding | Expensive, less customizable |
| **Streamlit** | Data science apps | Python-native, fast prototyping | Limited to Python ecosystem |

### 8.2 Performance Considerations

#### Optimizing for Speed
```python
# 1. Use vectorized operations
import numpy as np
x = np.linspace(0, 10, 1000000)
y = np.sin(x)  # Fast vectorized operation

# 2. Avoid loops when possible
# Bad: for loop
result = []
for i in range(len(x)):
    result.append(x[i] ** 2)

# Good: vectorized
result = x ** 2

# 3. Use appropriate data types
df['category'] = df['category'].astype('category')  # Memory efficient

# 4. Sample large datasets for exploration
sample_df = df.sample(n=10000, random_state=42)

# 5. Use downsampling for time series
df_resampled = df.resample('D').mean()  # Daily averages
```

## 9. Storytelling with Data

### 9.1 The Data Storytelling Framework

#### 1. Capture Attention
- Start with a surprising fact or question
- Use compelling visuals
- Create emotional connection

#### 2. Build Context
- Provide necessary background
- Explain why the data matters
- Set expectations

#### 3. Tell the Story
- Present data chronologically or logically
- Use transitions between insights
- Build toward key conclusions

#### 4. End with Action
- Clear call-to-action
- Specific recommendations
- Measurable outcomes

### 9.2 Common Storytelling Patterns

#### The Challenge-Solution Pattern
1. Present the problem/challenge
2. Show current state data
3. Reveal insights and solutions
4. Demonstrate impact of solution

#### The Before-After Pattern
1. Show "before" state
2. Highlight the change event
3. Present "after" state
4. Quantify the improvement

#### The Drill-Down Pattern
1. Start with high-level overview
2. Gradually reveal more detail
3. Focus on key insights
4. Provide actionable recommendations

## 10. Resources and Further Reading

### Books
- "The Visual Display of Quantitative Information" by Edward Tufte
- "Storytelling with Data" by Cole Nussbaumer Knaflic
- "Data Visualization: A Practical Introduction" by Kieran Healy

### Online Resources
- **Data Visualization Society**: https://www.datavisualizationsociety.com/
- **The Python Graph Gallery**: https://python-graph-gallery.com/
- **Plotly Documentation**: https://plotly.com/python/

### Tools and Libraries
- **Matplotlib**: https://matplotlib.org/
- **Seaborn**: https://seaborn.pydata.org/
- **Plotly**: https://plotly.com/python/
- **Bokeh**: https://bokeh.org/
- **Streamlit**: https://streamlit.io/

### Color Resources
- **ColorBrewer**: https://colorbrewer2.org/
- **Adobe Color**: https://color.adobe.com/
- **Coolors**: https://coolors.co/

## 11. Assessment

### Quiz Questions
1. What are the main differences between matplotlib and seaborn?
2. When should you use a pie chart versus a bar chart?
3. How do you ensure your visualizations are accessible to colorblind users?
4. What are the key components of an effective dashboard?
5. How does data storytelling differ from just presenting data?

### Practical Exercises
1. Create a comprehensive exploratory data analysis visualization suite
2. Build an interactive dashboard using Streamlit
3. Design a data story presentation using multiple visualization types
4. Optimize a slow visualization for better performance
5. Create publication-ready visualizations following best practices

## Next Steps

Congratulations on mastering data visualization! You now have the skills to create compelling visual representations of data that effectively communicate insights to any audience. In the next module, we'll explore big data technologies for handling large-scale datasets.

**Ready to continue?** Proceed to [Module 10: Big Data Technologies](../10_big_data_technologies/)
