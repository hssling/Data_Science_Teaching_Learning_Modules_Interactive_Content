# Module 4: Data Collection and Storage

## Overview
Data collection and storage form the foundation of any data science project. This module covers comprehensive techniques for acquiring data from various sources, understanding different data formats, and implementing robust storage solutions. You'll learn to work with APIs, web scraping, databases, data lakes, and cloud storage systems.

## Learning Objectives
By the end of this module, you will be able to:
- Collect data from APIs, web scraping, and public datasets
- Work with various data formats (JSON, CSV, XML, Parquet)
- Design and implement database schemas for data science
- Choose appropriate storage solutions for different use cases
- Implement data pipelines for automated collection
- Handle data quality and validation during collection
- Understand data governance and compliance considerations

## 1. Introduction to Data Collection

### 1.1 Data Sources and Types

#### Primary Data Sources
- **First-party data**: Data collected directly from your own systems
- **Second-party data**: Data obtained from trusted partners
- **Third-party data**: Data purchased from data providers or public sources

#### Data Collection Methods
- **Manual collection**: Surveys, forms, direct entry
- **Automated collection**: APIs, web scraping, sensors, logs
- **Observational data**: User behavior tracking, system monitoring
- **Experimental data**: A/B tests, controlled experiments

### 1.2 Data Collection Planning

#### Key Considerations
- **Purpose and scope**: What data do you need and why?
- **Data quality requirements**: Accuracy, completeness, timeliness
- **Volume and velocity**: How much data and how fast?
- **Legal and ethical constraints**: Privacy laws, consent requirements
- **Cost and feasibility**: Budget constraints and technical limitations

#### Data Collection Strategy
```python
# Example data collection planning framework
data_requirements = {
    'purpose': 'Customer churn prediction',
    'scope': {
        'demographics': ['age', 'gender', 'location'],
        'behavioral': ['purchase_history', 'usage_patterns', 'support_tickets'],
        'temporal': 'last_12_months'
    },
    'sources': [
        'internal_databases',
        'crm_system',
        'web_analytics',
        'customer_surveys'
    ],
    'quality_checks': [
        'completeness_validation',
        'accuracy_verification',
        'timeliness_assessment'
    ]
}
```

## 2. Working with APIs

### 2.1 REST API Fundamentals

#### HTTP Methods
- **GET**: Retrieve data from a resource
- **POST**: Create a new resource
- **PUT**: Update an existing resource
- **DELETE**: Remove a resource
- **PATCH**: Partially update a resource

#### API Response Formats
- **JSON**: Most common format for web APIs
- **XML**: Legacy format, still used in some systems
- **CSV**: Simple tabular data
- **Binary**: Images, files, or custom formats

### 2.2 Making API Requests with Python

#### Basic API Interaction
```python
import requests
import json
from typing import Dict, List, Optional

class APIClient:
    """Generic API client for data collection"""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()

        # Set default headers
        self.session.headers.update({
            'User-Agent': 'DataScience-Collector/1.0',
            'Accept': 'application/json'
        })

        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})

    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make a GET request to the API"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()  # Raise exception for bad status codes

            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return {}

    def post(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make a POST request to the API"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return {}

# Example usage
api_client = APIClient("https://api.example.com", api_key="your_api_key")

# Get user data
users = api_client.get("users", params={"limit": 100, "status": "active"})
print(f"Retrieved {len(users.get('data', []))} users")

# Create new user
new_user = {
    "name": "John Doe",
    "email": "john@example.com",
    "department": "Engineering"
}
result = api_client.post("users", data=new_user)
print(f"Created user with ID: {result.get('id')}")
```

### 2.3 Popular APIs for Data Science

#### Social Media APIs
```python
# Twitter API example (requires authentication)
import tweepy

class TwitterCollector:
    """Collect tweets for sentiment analysis"""

    def __init__(self, api_key: str, api_secret: str, access_token: str, access_token_secret: str):
        auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_token_secret)
        self.api = tweepy.API(auth, wait_on_rate_limit=True)

    def collect_tweets(self, query: str, count: int = 100) -> List[Dict]:
        """Collect recent tweets matching query"""
        tweets_data = []

        try:
            tweets = self.api.search_tweets(q=query, count=count, tweet_mode='extended')

            for tweet in tweets:
                tweet_info = {
                    'id': tweet.id,
                    'text': tweet.full_text,
                    'created_at': tweet.created_at,
                    'user_id': tweet.user.id,
                    'username': tweet.user.screen_name,
                    'followers_count': tweet.user.followers_count,
                    'retweet_count': tweet.retweet_count,
                    'favorite_count': tweet.favorite_count,
                    'hashtags': [tag['text'] for tag in tweet.entities['hashtags']],
                    'mentions': [mention['screen_name'] for mention in tweet.entities['user_mentions']]
                }
                tweets_data.append(tweet_info)

        except tweepy.TweepyException as e:
            print(f"Twitter API error: {e}")

        return tweets_data

# Usage
twitter_collector = TwitterCollector(api_key, api_secret, access_token, access_token_secret)
tweets = twitter_collector.collect_tweets("data science", count=50)
print(f"Collected {len(tweets)} tweets about data science")
```

#### Financial Data APIs
```python
# Alpha Vantage API for stock data
class StockDataCollector:
    """Collect financial data from Alpha Vantage API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def get_daily_stock_data(self, symbol: str, outputsize: str = 'compact') -> pd.DataFrame:
        """Get daily stock prices for a symbol"""
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': outputsize,  # 'compact' (100 points) or 'full' (20+ years)
            'apikey': self.api_key
        }

        response = requests.get(self.base_url, params=params)
        data = response.json()

        if 'Time Series (Daily)' not in data:
            print(f"Error: {data.get('Note', 'Unknown error')}")
            return pd.DataFrame()

        # Parse the time series data
        time_series = data['Time Series (Daily)']
        df_data = []

        for date, values in time_series.items():
            row = {
                'date': date,
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': int(values['5. volume'])
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        return df

# Usage
stock_collector = StockDataCollector("your_alpha_vantage_api_key")
apple_data = stock_collector.get_daily_stock_data("AAPL", outputsize='compact')
print(f"Collected {len(apple_data)} days of AAPL stock data")
print(apple_data.head())
```

#### Weather and Environmental Data
```python
# OpenWeatherMap API
class WeatherDataCollector:
    """Collect weather data for analysis"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"

    def get_current_weather(self, city: str, country_code: str = 'US') -> Dict:
        """Get current weather for a city"""
        params = {
            'q': f"{city},{country_code}",
            'appid': self.api_key,
            'units': 'metric'  # Celsius
        }

        response = requests.get(f"{self.base_url}/weather", params=params)

        if response.status_code == 200:
            data = response.json()
            weather_info = {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'weather_description': data['weather'][0]['description'],
                'timestamp': pd.to_datetime(data['dt'], unit='s')
            }
            return weather_info
        else:
            print(f"Weather API error: {response.status_code}")
            return {}

    def get_historical_weather(self, city: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical weather data (requires paid plan)"""
        # Note: Historical data requires paid OpenWeatherMap plan
        # This is a placeholder for the concept
        pass

# Usage
weather_collector = WeatherDataCollector("your_openweather_api_key")
current_weather = weather_collector.get_current_weather("New York", "US")
print(f"Current weather in {current_weather.get('city')}: {current_weather.get('temperature')}°C")
```

## 3. Web Scraping

### 3.1 HTML and CSS Selectors

#### Understanding Web Structure
```html
<!-- Example HTML structure -->
<html>
<head>
    <title>Sample Page</title>
</head>
<body>
    <div class="container">
        <h1 id="main-title">Main Title</h1>
        <div class="content">
            <p class="text">First paragraph</p>
            <p class="text">Second paragraph</p>
            <table id="data-table">
                <tr>
                    <th>Name</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Item 1</td>
                    <td>100</td>
                </tr>
            </table>
        </div>
    </div>
</body>
</html>
```

#### CSS Selectors for Data Extraction
```python
# CSS Selector examples
selectors = {
    'title': '#main-title',                    # ID selector
    'paragraphs': 'p.text',                   # Class selector
    'table': '#data-table',                   # ID selector
    'table_rows': '#data-table tr',           # Descendant selector
    'table_data': '#data-table td',           # Descendant selector
    'first_paragraph': 'p.text:first-child',  # Pseudo-class
    'links': 'a[href]',                       # Attribute selector
    'external_links': 'a[href^="http"]'       # Attribute starts with
}
```

### 3.2 Web Scraping with BeautifulSoup

#### Basic Web Scraping
```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import List, Dict

class WebScraper:
    """Web scraper for data collection"""

    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_page_content(self, url: str) -> BeautifulSoup:
        """Fetch and parse a web page"""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def scrape_product_data(self, url: str) -> List[Dict]:
        """Scrape product information from an e-commerce site"""
        soup = self.get_page_content(url)
        if not soup:
            return []

        products = []

        # Example selectors (adjust based on actual site structure)
        product_containers = soup.find_all('div', class_='product-item')

        for container in product_containers:
            try:
                product_info = {
                    'name': container.find('h3', class_='product-name').text.strip(),
                    'price': container.find('span', class_='price').text.strip(),
                    'rating': container.find('div', class_='rating').get('data-rating', 'N/A'),
                    'url': container.find('a')['href'] if container.find('a') else None
                }
                products.append(product_info)

            except AttributeError:
                # Skip products with missing information
                continue

        return products

    def scrape_table_data(self, url: str, table_class: str = None) -> pd.DataFrame:
        """Scrape tabular data from a webpage"""
        soup = self.get_page_content(url)
        if not soup:
            return pd.DataFrame()

        # Find the target table
        if table_class:
            table = soup.find('table', class_=table_class)
        else:
            table = soup.find('table')

        if not table:
            print("No table found on the page")
            return pd.DataFrame()

        # Extract headers
        headers = []
        header_row = table.find('thead').find('tr') if table.find('thead') else table.find('tr')

        for th in header_row.find_all('th'):
            headers.append(th.text.strip())

        # Extract data rows
        data = []
        rows = table.find_all('tr')[1:]  # Skip header row

        for row in rows:
            row_data = []
            for td in row.find_all('td'):
                row_data.append(td.text.strip())
            if row_data:  # Only add non-empty rows
                data.append(row_data)

        # Create DataFrame
        df = pd.DataFrame(data, columns=headers)
        return df

# Usage examples
scraper = WebScraper(delay=2.0)  # Respectful delay between requests

# Scrape product data
products = scraper.scrape_product_data("https://example.com/products")
print(f"Scraped {len(products)} products")

# Scrape table data
table_data = scraper.scrape_table_data("https://example.com/data-table", "data-table")
print(f"Scraped table with {len(table_data)} rows and {len(table_data.columns)} columns")
print(table_data.head())
```

### 3.3 Handling Dynamic Content

#### Selenium for JavaScript-Heavy Sites
```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

class DynamicWebScraper:
    """Scraper for dynamic websites using Selenium"""

    def __init__(self, headless: bool = True):
        self.options = Options()
        if headless:
            self.options.add_argument('--headless')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')

        # Initialize driver (requires ChromeDriver)
        self.driver = webdriver.Chrome(options=self.options)

    def scrape_dynamic_content(self, url: str) -> str:
        """Scrape content that loads dynamically"""
        try:
            self.driver.get(url)

            # Wait for dynamic content to load
            wait = WebDriverWait(self.driver, 10)

            # Example: Wait for a specific element to appear
            element = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "dynamic-content"))
            )

            # Scroll down to load more content (if needed)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Wait for content to load

            # Get the page source
            page_source = self.driver.page_source

            return page_source

        except Exception as e:
            print(f"Error scraping dynamic content: {e}")
            return ""

    def scrape_infinite_scroll(self, url: str, max_scrolls: int = 5) -> List[Dict]:
        """Scrape content from infinite scroll pages"""
        self.driver.get(url)
        time.sleep(3)  # Initial load

        items = []

        for scroll in range(max_scrolls):
            # Get current items
            current_items = self.driver.find_elements(By.CLASS_NAME, "item")

            for item in current_items[len(items):]:  # Only process new items
                try:
                    item_data = {
                        'title': item.find_element(By.CLASS_NAME, "title").text,
                        'description': item.find_element(By.CLASS_NAME, "description").text,
                        'url': item.find_element(By.TAG_NAME, "a").get_attribute("href")
                    }
                    items.append(item_data)
                except Exception as e:
                    continue

            # Scroll down
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Wait for new content

            # Check if we've reached the end
            new_items = self.driver.find_elements(By.CLASS_NAME, "item")
            if len(new_items) == len(items):
                break  # No new items loaded

        return items

    def close(self):
        """Close the browser"""
        self.driver.quit()

# Usage
dynamic_scraper = DynamicWebScraper(headless=False)  # Set to False to see browser

# Scrape dynamic content
content = dynamic_scraper.scrape_dynamic_content("https://example.com/dynamic-page")

# Scrape infinite scroll
items = dynamic_scraper.scrape_infinite_scroll("https://example.com/infinite-scroll", max_scrolls=3)
print(f"Scraped {len(items)} items from infinite scroll")

dynamic_scraper.close()
```

### 3.4 Best Practices and Ethics

#### Responsible Web Scraping
```python
import time
import random
from fake_useragent import UserAgent

class EthicalWebScraper:
    """Web scraper with ethical considerations"""

    def __init__(self, base_delay: float = 1.0, random_delay: float = 2.0):
        self.base_delay = base_delay
        self.random_delay = random_delay

        # Use random user agents to avoid detection
        self.ua = UserAgent()

        self.session = requests.Session()

    def respectful_request(self, url: str) -> requests.Response:
        """Make a respectful HTTP request"""

        # Rotate user agents
        self.session.headers.update({'User-Agent': self.ua.random})

        # Add random delay to avoid overwhelming servers
        delay = self.base_delay + random.uniform(0, self.random_delay)
        time.sleep(delay)

        try:
            response = self.session.get(url)

            # Check robots.txt (simplified)
            if self.check_robots_txt(url):
                print(f"Access to {url} blocked by robots.txt")
                return None

            # Respect rate limits
            if response.status_code == 429:  # Too Many Requests
                retry_after = int(response.headers.get('Retry-After', 60))
                print(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                return self.respectful_request(url)  # Retry

            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

    def check_robots_txt(self, url: str) -> bool:
        """Check if scraping is allowed by robots.txt"""
        from urllib.parse import urlparse
        from urllib.robotparser import RobotFileParser

        parsed_url = urlparse(url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"

        try:
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()

            # Check if our user agent can fetch the URL
            return not rp.can_fetch(self.session.headers.get('User-Agent', '*'), url)

        except Exception:
            # If we can't read robots.txt, assume it's allowed
            return False

    def save_checkpoint(self, data: List[Dict], filename: str):
        """Save progress to avoid losing work"""
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Checkpoint saved: {len(data)} items")

# Usage
ethical_scraper = EthicalWebScraper(base_delay=2.0, random_delay=3.0)

# Scrape with respect for server resources
response = ethical_scraper.respectful_request("https://example.com/data")
if response:
    print(f"Successfully fetched {len(response.content)} bytes")
```

## 4. Database Systems

### 4.1 Relational Databases (SQL)

#### Database Design Principles
```sql
-- Example database schema for e-commerce analytics
CREATE DATABASE ecommerce_analytics;

USE ecommerce_analytics;

-- Customers table
CREATE TABLE customers (
    customer_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20),
    registration_date DATE NOT NULL,
    customer_segment VARCHAR(20) DEFAULT 'Regular'
);

-- Products table
CREATE TABLE products (
    product_id INT PRIMARY KEY AUTO_INCREMENT,
    product_name VARCHAR(100) NOT NULL,
    category VARCHAR(50) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    cost DECIMAL(10, 2),
    stock_quantity INT DEFAULT 0,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE orders (
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT NOT NULL,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(10, 2) NOT NULL,
    status VARCHAR(20) DEFAULT 'Pending',
    shipping_address TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Order items table (junction table)
CREATE TABLE order_items (
    order_item_id INT PRIMARY KEY AUTO_INCREMENT,
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    quantity INT NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Indexes for performance
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_products_category ON products(category);
```

#### Python Database Operations
```python
import mysql.connector
import pandas as pd
from typing import List, Dict, Optional

class DatabaseManager:
    """Database manager for data science workflows"""

    def __init__(self, host: str, user: str, password: str, database: str):
        self.config = {
            'host': host,
            'user': user,
            'password': password,
            'database': database
        }
        self.connection = None

    def connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            print("Database connection established")
        except mysql.connector.Error as e:
            print(f"Database connection failed: {e}")

    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("Database connection closed")

    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """Execute a SELECT query and return results"""
        if not self.connection:
            self.connect()

        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params or ())

            results = cursor.fetchall()
            cursor.close()

            return results

        except mysql.connector.Error as e:
            print(f"Query execution failed: {e}")
            return []

    def execute_update(self, query: str, params: Optional[tuple] = None) -> bool:
        """Execute an INSERT, UPDATE, or DELETE query"""
        if not self.connection:
            self.connect()

        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params or ())
            self.connection.commit()
            cursor.close()

            return True

        except mysql.connector.Error as e:
            print(f"Update execution failed: {e}")
            self.connection.rollback()
            return False

    def load_table_to_dataframe(self, table_name: str, conditions: Optional[str] = None) -> pd.DataFrame:
        """Load a database table into a pandas DataFrame"""
        query = f"SELECT * FROM {table_name}"
        if conditions:
            query += f" WHERE {conditions}"

        results = self.execute_query(query)

        if results:
            df = pd.DataFrame(results)
            print(f"Loaded {len(df)} rows from {table_name}")
            return df
        else:
            return pd.DataFrame()

    def save_dataframe_to_table(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> bool:
        """Save a pandas DataFrame to a database table"""
        if if_exists not in ['append', 'replace']:
            print("if_exists must be 'append' or 'replace'")
            return False

        try:
            # Create table if it doesn't exist (basic implementation)
            if if_exists == 'replace':
                columns = ', '.join([f"{col} VARCHAR(255)" for col in df.columns])
                create_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
                self.execute_update(create_query)

            # Insert data
            columns = ', '.join(df.columns)
            placeholders = ', '.join(['%s'] * len(df.columns))

            insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

            for _, row in df.iterrows():
                self.execute_update(insert_query, tuple(row))

            print(f"Saved {len(df)} rows to {table_name}")
            return True

        except Exception as e:
            print(f"Failed to save DataFrame: {e}")
            return False

# Usage example
db_manager = DatabaseManager(
    host="localhost",
    user="data_science_user",
    password="secure_password",
    database="ecommerce_analytics"
)

# Load customer data
customers_df = db_manager.load_table_to_dataframe("customers")
print(f"Loaded {len(customers_df)} customers")

# Execute custom query
high_value_orders = db_manager.execute_query("""
    SELECT o.order_id, c.first_name, c.last_name, o.total_amount
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.total_amount > 500
    ORDER BY o.total_amount DESC
    LIMIT 10
""")

print(f"Found {len(high_value_orders)} high-value orders")

# Save analysis results
analysis_results = pd.DataFrame({
    'metric': ['total_revenue', 'avg_order_value', 'total_customers'],
    'value': [15420.50, 89.75, 1250]
})

db_manager.save_dataframe_to_table(analysis_results, "analysis_results")

db_manager.disconnect()
```

### 4.2 NoSQL Databases

#### MongoDB for Document Storage
```python
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import json
from datetime import datetime

class MongoDBManager:
    """MongoDB manager for flexible data storage"""

    def __init__(self, connection_string: str, database_name: str):
        self.connection_string = connection_string
        self.database_name = database_name
        self.client = None
        self.db = None

    def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(self.connection_string)
            self.db = self.client[self.database_name]
            # Test connection
            self.client.admin.command('ping')
            print("MongoDB connection established")
        except ConnectionFailure:
            print("MongoDB connection failed")

    def disconnect(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("MongoDB connection closed")

    def insert_document(self, collection_name: str, document: Dict) -> str:
        """Insert a document into a collection"""
        try:
            collection = self.db[collection_name]
            result = collection.insert_one(document)
            print(f"Inserted document with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            print(f"Insert failed: {e}")
            return None

    def insert_many_documents(self, collection_name: str, documents: List[Dict]) -> int:
        """Insert multiple documents"""
        try:
            collection = self.db[collection_name]
            result = collection.insert_many(documents)
            print(f"Inserted {len(result.inserted_ids)} documents")
            return len(result.inserted_ids)
        except Exception as e:
            print(f"Batch insert failed: {e}")
            return 0

    def find_documents(self, collection_name: str, query: Dict = None,
                      projection: Dict = None, limit: int = None) -> List[Dict]:
        """Find documents matching a query"""
        try:
            collection = self.db[collection_name]

            cursor = collection.find(query or {}, projection or {})

            if limit:
                cursor = cursor.limit(limit)

            results = list(cursor)
            print(f"Found {len(results)} documents")
            return results

        except Exception as e:
            print(f"Query failed: {e}")
            return []

    def update_document(self, collection_name: str, query: Dict, update: Dict) -> bool:
        """Update a document"""
        try:
            collection = self.db[collection_name]
            result = collection.update_one(query, {"$set": update})

            if result.modified_count > 0:
                print("Document updated successfully")
                return True
            else:
                print("No document was updated")
                return False

        except Exception as e:
            print(f"Update failed: {e}")
            return False

    def aggregate_data(self, collection_name: str, pipeline: List[Dict]) -> List[Dict]:
        """Perform aggregation operations"""
        try:
            collection = self.db[collection_name]
            results = list(collection.aggregate(pipeline))
            print(f"Aggregation returned {len(results)} results")
            return results

        except Exception as e:
            print(f"Aggregation failed: {e}")
            return []

# Usage example
mongo_manager = MongoDBManager(
    connection_string="mongodb://localhost:27017/",
    database_name="social_media_analytics"
)

mongo_manager.connect()

# Insert social media posts
posts = [
    {
        "post_id": "12345",
        "platform": "twitter",
        "content": "Excited about the new data science conference! #DataScience",
        "author": "data_science_user",
        "timestamp": datetime.now(),
        "engagement": {
            "likes": 25,
            "retweets": 8,
            "replies": 3
        },
        "hashtags": ["DataScience"],
        "sentiment": "positive"
    },
    {
        "post_id": "12346",
        "platform": "instagram",
        "content": "Beautiful visualization of climate change data 📊",
        "author": "climate_scientist",
        "timestamp": datetime.now(),
        "engagement": {
            "likes": 150,
            "comments": 12,
            "shares": 5
        },
        "hashtags": ["ClimateChange", "DataViz"],
        "sentiment": "neutral"
    }
]

mongo_manager.insert_many_documents("posts", posts)

# Query posts with high engagement
high_engagement_posts = mongo_manager.find_documents(
    "posts",
    {"engagement.likes": {"$gte": 20}},
    {"content": 1, "engagement": 1, "author": 1}
)

# Aggregation: Average engagement by platform
engagement_pipeline = [
    {
        "$group": {
            "_id": "$platform",
            "avg_likes": {"$avg": "$engagement.likes"},
            "total_posts": {"$sum": 1}
        }
    },
    {
        "$sort": {"avg_likes": -1}
    }
]

platform_stats = mongo_manager.aggregate_data("posts", engagement_pipeline)

for stat in platform_stats:
    print(f"Platform: {stat['_id']}, Avg Likes: {stat['avg_likes']:.1f}, Total Posts: {stat['total_posts']}")

mongo_manager.disconnect()
```

## 5. Data Lakes and Cloud Storage

### 5.1 Data Lake Architecture

#### Data Lake vs Data Warehouse
- **Data Lake**: Store raw data in native format, schema-on-read
- **Data Warehouse**: Structured data storage, schema-on-write
- **Use Cases**: Data lakes for big data analytics, warehouses for business intelligence

#### Data Lake Zones
1. **Landing Zone**: Raw data ingestion
2. **Clean Zone**: Processed and cleaned data
3. **Curated Zone**: Business-ready data with schemas
4. **Consumption Zone**: Data optimized for specific use cases

### 5.2 Cloud Storage Solutions

#### AWS S3 for Data Lakes
```python
import boto3
from botocore.exceptions import NoCredentialsError
import pandas as pd
from io import StringIO, BytesIO

class S3DataManager:
    """AWS S3 manager for data lake operations"""

    def __init__(self, aws_access_key: str, aws_secret_key: str, region: str = 'us-east-1'):
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.region = region
        self.s3_client = None
        self.bucket_name = None

    def connect(self, bucket_name: str):
        """Connect to S3 bucket"""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.region
            )
            self.bucket_name = bucket_name
            print(f"Connected to S3 bucket: {bucket_name}")
        except NoCredentialsError:
            print("AWS credentials not found")

    def upload_dataframe(self, df: pd.DataFrame, key: str, format: str = 'csv') -> bool:
        """Upload pandas DataFrame to S3"""
        try:
            if format.lower() == 'csv':
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=csv_buffer.getvalue()
                )
            elif format.lower() == 'parquet':
                # Requires pyarrow or fastparquet
                parquet_buffer = BytesIO()
                df.to_parquet(parquet_buffer, index=False)
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=parquet_buffer.getvalue()
                )
            else:
                raise ValueError("Format must be 'csv' or 'parquet'")

            print(f"Uploaded {len(df)} rows to s3://{self.bucket_name}/{key}")
            return True

        except Exception as e:
            print(f"Upload failed: {e}")
            return False

    def download_dataframe(self, key: str) -> pd.DataFrame:
        """Download data from S3 as pandas DataFrame"""
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            body = obj['Body'].read()

            # Determine format from file extension
            if key.endswith('.csv'):
                df = pd.read_csv(BytesIO(body))
            elif key.endswith('.parquet'):
                df = pd.read_parquet(BytesIO(body))
            else:
                # Assume CSV
                df = pd.read_csv(BytesIO(body))

            print(f"Downloaded {len(df)} rows from s3://{self.bucket_name}/{key}")
            return df

        except Exception as e:
            print(f"Download failed: {e}")
            return pd.DataFrame()

    def list_objects(self, prefix: str = "") -> List[str]:
        """List objects in bucket with optional prefix"""
        try:
            objects = []
            paginator = self.s3_client.get_paginator('list_objects_v2')

            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects.append(obj['Key'])

            return objects

        except Exception as e:
            print(f"List objects failed: {e}")
            return []

# Usage example
s3_manager = S3DataManager("your_access_key", "your_secret_key")
s3_manager.connect("my-data-lake")

# Upload data
sample_data = pd.DataFrame({
    'customer_id': range(1, 101),
    'revenue': np.random.uniform(10, 1000, 100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

s3_manager.upload_dataframe(sample_data, "raw/customer_data_2023.csv")

# List objects
objects = s3_manager.list_objects("raw/")
print(f"Found {len(objects)} objects in raw/ folder")

# Download data
downloaded_data = s3_manager.download_dataframe("raw/customer_data_2023.csv")
print(f"Downloaded data shape: {downloaded_data.shape}")
```

#### Google Cloud Storage
```python
from google.cloud import storage
import pandas as pd
from io import StringIO

class GCSDataManager:
    """Google Cloud Storage manager"""

    def __init__(self, project_id: str, credentials_path: str = None):
        self.project_id = project_id
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

        self.client = storage.Client(project=project_id)
        self.bucket = None

    def connect_bucket(self, bucket_name: str):
        """Connect to a GCS bucket"""
        try:
            self.bucket = self.client.bucket(bucket_name)
            print(f"Connected to GCS bucket: {bucket_name}")
        except Exception as e:
            print(f"GCS connection failed: {e}")

    def upload_dataframe(self, df: pd.DataFrame, blob_name: str) -> bool:
        """Upload DataFrame to GCS"""
        try:
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)

            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')

            print(f"Uploaded {len(df)} rows to gs://{self.bucket.name}/{blob_name}")
            return True

        except Exception as e:
            print(f"Upload failed: {e}")
            return False

    def download_dataframe(self, blob_name: str) -> pd.DataFrame:
        """Download data from GCS as DataFrame"""
        try:
            blob = self.bucket.blob(blob_name)
            csv_data = blob.download_as_text()

            df = pd.read_csv(StringIO(csv_data))
            print(f"Downloaded {len(df)} rows from gs://{self.bucket.name}/{blob_name}")
            return df

        except Exception as e:
            print(f"Download failed: {e}")
            return pd.DataFrame()

# Usage
gcs_manager = GCSDataManager("your-project-id", "path/to/credentials.json")
gcs_manager.connect_bucket("my-data-lake")
gcs_manager.upload_dataframe(sample_data, "processed/customer_data_2023.csv")
```

## 6. Data Quality and Validation

### 6.1 Data Quality Checks

#### Automated Quality Validation
```python
import pandas as pd
import numpy as np
from typing import Dict, List

class DataQualityChecker:
    """Automated data quality validation"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.quality_report = {}

    def check_completeness(self) -> Dict[str, float]:
        """Check data completeness (non-null percentages)"""
        completeness = {}
        total_rows = len(self.df)

        for column in self.df.columns:
            non_null_count = self.df[column].notna().sum()
            completeness[column] = (non_null_count / total_rows) * 100

        self.quality_report['completeness'] = completeness
        return completeness

    def check_uniqueness(self) -> Dict[str, float]:
        """Check uniqueness of values"""
        uniqueness = {}

        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                unique_count = self.df[column].nunique()
                total_count = len(self.df[column])
                uniqueness[column] = (unique_count / total_count) * 100
            else:
                uniqueness[column] = None  # Not applicable for numeric

        self.quality_report['uniqueness'] = uniqueness
        return uniqueness

    def check_validity(self, rules: Dict[str, callable]) -> Dict[str, bool]:
        """Check data validity against custom rules"""
        validity = {}

        for column, rule_func in rules.items():
            if column in self.df.columns:
                try:
                    valid_count = self.df[column].apply(rule_func).sum()
                    validity[column] = (valid_count == len(self.df[column]))
                except Exception as e:
                    validity[column] = False
                    print(f"Validity check failed for {column}: {e}")
            else:
                validity[column] = None

        self.quality_report['validity'] = validity
        return validity

    def check_consistency(self) -> Dict[str, bool]:
        """Check logical consistency"""
        consistency = {}

        # Example: Check if end_date is after start_date
        date_columns = [col for col in self.df.columns if 'date' in col.lower()]
        if len(date_columns) >= 2:
            try:
                consistency['date_order'] = (
                    pd.to_datetime(self.df[date_columns[1]]) >=
                    pd.to_datetime(self.df[date_columns[0]])
                ).all()
            except:
                consistency['date_order'] = False

        # Example: Check if price is positive
        if 'price' in self.df.columns:
            consistency['positive_price'] = (self.df['price'] > 0).all()

        self.quality_report['consistency'] = consistency
        return consistency

    def generate_report(self) -> Dict:
        """Generate comprehensive quality report"""
        self.check_completeness()
        self.check_uniqueness()
        self.check_consistency()

        return self.quality_report

    def get_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)"""
        if not self.quality_report:
            self.generate_report()

        scores = []

        # Completeness score
        completeness_scores = [v for v in self.quality_report.get('completeness', {}).values() if v is not None]
        if completeness_scores:
            scores.append(np.mean(completeness_scores))

        # Consistency score
        consistency_results = self.quality_report.get('consistency', {})
        if consistency_results:
            consistency_score = sum(consistency_results.values()) / len(consistency_results) * 100
            scores.append(consistency_score)

        return np.mean(scores) if scores else 0.0

# Usage example
# Sample data with quality issues
sample_data = pd.DataFrame({
    'customer_id': range(1, 101),
    'name': ['Customer_' + str(i) for i in range(1, 101)],
    'email': ['customer' + str(i) + '@example.com' for i in range(1, 101)],
    'age': np.random.normal(35, 10, 100),
    'price': np.random.uniform(10, 100, 100),
    'signup_date': pd.date_range('2020-01-01', periods=100, freq='D'),
    'last_purchase': pd.date_range('2023-01-01', periods=100, freq='D')
})

# Introduce some quality issues
sample_data.loc[10:15, 'email'] = None  # Missing emails
sample_data.loc[20, 'price'] = -50      # Negative price
sample_data.loc[30:35, 'age'] = None    # Missing ages

# Validate data quality
quality_checker = DataQualityChecker(sample_data)

# Define validation rules
validation_rules = {
    'email': lambda x: '@' in str(x),  # Must contain @
    'price': lambda x: x > 0,          # Must be positive
    'age': lambda x: 18 <= x <= 100    # Must be reasonable age
}

# Run quality checks
completeness = quality_checker.check_completeness()
validity = quality_checker.check_validity(validation_rules)
consistency = quality_checker.check_consistency()

print("Data Quality Report:")
print(f"Completeness: {completeness}")
print(f"Validity: {validity}")
print(f"Consistency: {consistency}")
print(f"Overall Quality Score: {quality_checker.get_quality_score():.1f}%")
```

## 7. Data Pipeline Orchestration

### 7.1 Building Data Pipelines

#### Simple ETL Pipeline
```python
import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
from typing import Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPipeline:
    """Simple ETL data pipeline"""

    def __init__(self, config: Dict):
        self.config = config
        self.extracted_data = {}
        self.transformed_data = {}

    def extract(self) -> bool:
        """Extract data from various sources"""
        logger.info("Starting data extraction...")

        try:
            # Extract from API
            if 'api_sources' in self.config:
                for api_config in self.config['api_sources']:
                    api_client = APIClient(api_config['url'], api_config.get('key'))
                    data = api_client.get(api_config['endpoint'])
                    self.extracted_data[api_config['name']] = pd.DataFrame(data)

            # Extract from database
            if 'database_sources' in self.config:
                for db_config in self.config['database_sources']:
                    db_manager = DatabaseManager(**db_config)
                    df = db_manager.load_table_to_dataframe(db_config['table'])
                    self.extracted_data[db_config['name']] = df
                    db_manager.disconnect()

            # Extract from files
            if 'file_sources' in self.config:
                for file_config in self.config['file_sources']:
                    if file_config['format'] == 'csv':
                        df = pd.read_csv(file_config['path'])
                    elif file_config['format'] == 'json':
                        df = pd.read_json(file_config['path'])
                    self.extracted_data[file_config['name']] = df

            logger.info(f"Extracted {len(self.extracted_data)} datasets")
            return True

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False

    def transform(self) -> bool:
        """Transform extracted data"""
        logger.info("Starting data transformation...")

        try:
            for name, df in self.extracted_data.items():
                # Clean data
                df_clean = self._clean_data(df)

                # Validate data
                df_validated = self._validate_data(df_clean)

                # Enrich data
                df_enriched = self._enrich_data(df_validated)

                # Aggregate data
                df_aggregated = self._aggregate_data(df_enriched)

                self.transformed_data[name] = df_aggregated

            logger.info(f"Transformed {len(self.transformed_data)} datasets")
            return True

        except Exception as e:
            logger.error(f"Transformation failed: {e}")
            return False

    def load(self) -> bool:
        """Load transformed data to destination"""
        logger.info("Starting data loading...")

        try:
            for name, df in self.transformed_data.items():
                # Load to database
                if 'database_destination' in self.config:
                    db_config = self.config['database_destination']
                    db_manager = DatabaseManager(**db_config)
                    success = db_manager.save_dataframe_to_table(df, f"processed_{name}")
                    if not success:
                        raise Exception(f"Failed to load {name} to database")

                # Load to cloud storage
                if 'cloud_destination' in self.config:
                    cloud_config = self.config['cloud_destination']
                    if cloud_config['provider'] == 's3':
                        s3_manager = S3DataManager(cloud_config['key'], cloud_config['secret'])
                        s3_manager.connect(cloud_config['bucket'])
                        s3_manager.upload_dataframe(df, f"processed/{name}_{datetime.now().date()}.csv")

            logger.info("Data loading completed successfully")
            return True

        except Exception as e:
            logger.error(f"Loading failed: {e}")
            return False

    def run_pipeline(self) -> bool:
        """Run the complete ETL pipeline"""
        logger.info("Starting ETL pipeline...")

        success = True
        success &= self.extract()
        success &= self.transform()
        success &= self.load()

        if success:
            logger.info("ETL pipeline completed successfully!")
        else:
            logger.error("ETL pipeline failed!")

        return success

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw data"""
        # Remove duplicates
        df = df.drop_duplicates()

        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))

        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        return df

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality"""
        # Remove rows with too many missing values
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=threshold)

        # Validate data types
        # Add custom validation logic here

        return df

    def _enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich data with additional information"""
        # Add derived columns
        if 'date' in df.columns:
            df['year'] = pd.to_datetime(df['date']).dt.year
            df['month'] = pd.to_datetime(df['date']).dt.month
            df['day_of_week'] = pd.to_datetime(df['date']).dt.day_name()

        # Add calculated columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df['row_sum'] = df[numeric_cols].sum(axis=1)
            df['row_mean'] = df[numeric_cols].mean(axis=1)

        return df

    def _aggregate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data for analysis"""
        # Group by categorical columns and aggregate numeric columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            agg_dict = {col: ['count', 'mean', 'sum', 'std'] for col in numeric_cols}
            df_agg = df.groupby(list(categorical_cols)).agg(agg_dict)

            # Flatten column names
            df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
            df_agg = df_agg.reset_index()

            return df_agg

        return df

# Pipeline configuration
pipeline_config = {
    'api_sources': [
        {
            'name': 'user_data',
            'url': 'https://api.example.com',
            'endpoint': 'users',
            'key': 'your_api_key'
        }
    ],
    'database_sources': [
        {
            'name': 'sales_data',
            'host': 'localhost',
            'user': 'data_user',
            'password': 'secure_password',
            'database': 'sales_db',
            'table': 'transactions'
        }
    ],
    'file_sources': [
        {
            'name': 'external_data',
            'path': 'data/external/customers.csv',
            'format': 'csv'
        }
    ],
    'database_destination': {
        'host': 'localhost',
        'user': 'data_user',
        'password': 'secure_password',
        'database': 'analytics_db'
    },
    'cloud_destination': {
        'provider': 's3',
        'key': 'your_aws_key',
        'secret': 'your_aws_secret',
        'bucket': 'data-lake-bucket'
    }
}

# Run the pipeline
pipeline = DataPipeline(pipeline_config)
success = pipeline.run_pipeline()

if success:
    print("🎉 Data pipeline executed successfully!")
else:
    print("❌ Data pipeline failed. Check logs for details.")
```

## 8. Legal and Ethical Considerations

### 8.1 Data Privacy Laws

#### GDPR (Europe)
- **Personal Data**: Any information relating to an identified or identifiable natural person
- **Data Subject Rights**: Right to access, rectify, erase, restrict processing
- **Consent**: Must be freely given, specific, informed, and unambiguous
- **Data Protection Officer**: Required for certain organizations

#### CCPA (California)
- **Personal Information**: Information that identifies, relates to, or could reasonably be linked with a consumer
- **Right to Know**: What personal information is collected and how it's used
- **Right to Delete**: Ability to request deletion of personal information
- **Right to Opt-out**: Ability to opt-out of sale of personal information

### 8.2 Data Governance Best Practices

#### Data Classification
```python
class DataClassifier:
    """Classify data sensitivity levels"""

    SENSITIVITY_LEVELS = {
        'public': ['name', 'company', 'job_title'],
        'internal': ['email', 'phone', 'address'],
        'confidential': ['ssn', 'financial_data', 'health_records'],
        'restricted': ['passwords', 'encryption_keys', 'trade_secrets']
    }

    @staticmethod
    def classify_column(column_name: str, sample_values: List = None) -> str:
        """Classify a data column's sensitivity"""
        column_lower = column_name.lower()

        for level, keywords in DataClassifier.SENSITIVITY_LEVELS.items():
            if any(keyword in column_lower for keyword in keywords):
                return level

        # Check sample values for patterns
        if sample_values:
            # Check for email patterns
            if any('@' in str(val) for val in sample_values[:10]):
                return 'internal'

            # Check for phone patterns
            phone_pattern = r'^\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$'
            if any(re.match(phone_pattern, str(val)) for val in sample_values[:10]):
                return 'internal'

        return 'public'

    @staticmethod
    def get_retention_policy(sensitivity_level: str) -> Dict:
        """Get data retention policy based on sensitivity"""
        policies = {
            'public': {'retention_years': 7, 'encryption': False, 'access_control': 'basic'},
            'internal': {'retention_years': 5, 'encryption': True, 'access_control': 'role_based'},
            'confidential': {'retention_years': 3, 'encryption': True, 'access_control': 'strict'},
            'restricted': {'retention_years': 1, 'encryption': True, 'access_control': 'need_to_know'}
        }

        return policies.get(sensitivity_level, policies['public'])
```

#### Data Lineage Tracking
```python
from datetime import datetime
import hashlib

class DataLineageTracker:
    """Track data lineage and transformations"""

    def __init__(self):
        self.lineage = {}

    def track_source(self, source_name: str, source_type: str, metadata: Dict):
        """Track data source"""
        source_id = hashlib.md5(f"{source_name}_{datetime.now()}".encode()).hexdigest()[:8]

        self.lineage[source_id] = {
            'type': 'source',
            'name': source_name,
            'source_type': source_type,
            'metadata': metadata,
            'timestamp': datetime.now(),
            'transformations': []
        }

        return source_id

    def track_transformation(self, source_id: str, transformation_name: str,
                           transformation_type: str, parameters: Dict):
        """Track data transformation"""
        if source_id not in self.lineage:
            raise ValueError(f"Source {source_id} not found")

        transformation = {
            'name': transformation_name,
            'type': transformation_type,
            'parameters': parameters,
            'timestamp': datetime.now(),
            'output_schema': {}  # Would be populated with actual schema
        }

        self.lineage[source_id]['transformations'].append(transformation)

    def get_lineage(self, source_id: str) -> Dict:
        """Get complete lineage for a data source"""
        return self.lineage.get(source_id, {})

    def export_lineage(self, filepath: str):
        """Export lineage information to file"""
        import json

        # Convert datetime objects to strings for JSON serialization
        export_data = {}
        for source_id, data in self.lineage.items():
            export_data[source_id] = data.copy()
            export_data[source_id]['timestamp'] = data['timestamp'].isoformat()
            for transformation in export_data[source_id]['transformations']:
                transformation['timestamp'] = transformation['timestamp'].isoformat()

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

# Usage
lineage_tracker = DataLineageTracker()

# Track data source
source_id = lineage_tracker.track_source(
    source_name="customer_database",
    source_type="postgresql",
    metadata={
        'host': 'db.example.com',
        'database': 'customers',
        'table': 'user_profiles',
        'row_count': 100000
    }
)

# Track transformations
lineage_tracker.track_transformation(
    source_id=source_id,
    transformation_name="remove_duplicates",
    transformation_type="data_cleaning",
    parameters={'columns': ['email', 'phone']}
)

lineage_tracker.track_transformation(
    source_id=source_id,
    transformation_name="normalize_names",
    transformation_type="data_standardization",
    parameters={'method': 'lowercase_strip'}
)

# Export lineage
lineage_tracker.export_lineage("data_lineage.json")
```

## 9. Assessment

### Quiz Questions
1. What are the main differences between APIs, web scraping, and direct database access for data collection?
2. How would you handle rate limiting when collecting data from APIs?
3. What are the key considerations when designing a database schema for data science?
4. How do data lakes differ from data warehouses?
5. What are the main components of a data quality validation framework?

### Practical Exercises
1. Build an API client to collect data from a public API (e.g., weather, stocks)
2. Create a web scraper to collect product information from an e-commerce site
3. Design and implement a database schema for a retail analytics system
4. Build a data pipeline that extracts, transforms, and loads data
5. Implement automated data quality checks for a dataset

## 10. Resources

### APIs and Data Sources
- **Public APIs**: https://github.com/public-apis/public-apis
- **Kaggle Datasets**: https://www.kaggle.com/datasets
- **Google Dataset Search**: https://datasetsearch.research.google.com/
- **AWS Open Data**: https://registry.opendata.aws/

### Web Scraping
- **BeautifulSoup Documentation**: https://www.crummy.com/software/BeautifulSoup/
- **Scrapy Framework**: https://scrapy.org/
- **Selenium Documentation**: https://selenium.dev/documentation/

### Databases
- **PostgreSQL Documentation**: https://www.postgresql.org/docs/
- **MongoDB Documentation**: https://docs.mongodb.com/
- **SQLZoo**: https://sqlzoo.net/ (SQL practice)

### Cloud Storage
- **AWS S3 Documentation**: https://docs.aws.amazon.com/s3/
- **Google Cloud Storage**: https://cloud.google.com/storage/docs
- **Azure Blob Storage**: https://docs.microsoft.com/en-us/azure/storage/blobs/

## Next Steps

Congratulations on mastering data collection and storage! You now have the skills to acquire data from various sources and implement robust storage solutions. In the next module, we'll explore data cleaning and preprocessing techniques to prepare your data for analysis.

**Ready to continue?** Proceed to [Module 5: Data Cleaning and Preprocessing](../05_data_cleaning_preprocessing/)
