# Module 10: Big Data Technologies

## Overview
Big Data technologies enable processing and analysis of massive datasets that traditional tools cannot handle. This module covers distributed computing frameworks, NoSQL databases, stream processing, and modern data lake architectures that power today's data-intensive applications.

## Learning Objectives
By the end of this module, you will be able to:
- Understand distributed computing principles and architectures
- Work with Apache Hadoop ecosystem for batch processing
- Implement real-time stream processing with Apache Kafka and Spark Streaming
- Design and manage NoSQL databases for big data applications
- Build scalable data pipelines using modern big data tools
- Optimize performance for large-scale data processing
- Choose appropriate technologies for different big data use cases

## 1. Introduction to Big Data

### 1.1 The Big Data Landscape

#### The 5 V's of Big Data
- **Volume**: Scale of data (terabytes to petabytes)
- **Velocity**: Speed of data generation and processing
- **Variety**: Different types of data (structured, semi-structured, unstructured)
- **Veracity**: Quality and trustworthiness of data
- **Value**: Business value extracted from data

#### Big Data Challenges
- **Storage**: How to store massive amounts of data cost-effectively
- **Processing**: How to process data faster than it arrives
- **Analysis**: How to extract insights from diverse data types
- **Privacy**: How to handle sensitive data at scale
- **Cost**: Balancing performance with infrastructure costs

### 1.2 Distributed Computing Fundamentals

#### Horizontal vs Vertical Scaling
```python
# Conceptual comparison of scaling approaches
scaling_comparison = {
    'vertical_scaling': {
        'approach': 'Scale up single machine',
        'pros': ['Simpler architecture', 'Easier management', 'Better consistency'],
        'cons': ['Hardware limits', 'Single point of failure', 'Expensive at scale'],
        'use_case': 'Small to medium datasets'
    },
    'horizontal_scaling': {
        'approach': 'Scale out across multiple machines',
        'pros': ['Near unlimited scalability', 'Fault tolerance', 'Cost-effective'],
        'cons': ['Complex architecture', 'Consistency challenges', 'Network overhead'],
        'use_case': 'Large-scale distributed systems'
    }
}

print("Scaling Approaches Comparison:")
for approach, details in scaling_comparison.items():
    print(f"\n{approach.upper()}:")
    print(f"  Approach: {details['approach']}")
    print(f"  Use Case: {details['use_case']}")
    print(f"  Pros: {', '.join(details['pros'])}")
    print(f"  Cons: {', '.join(details['cons'])}")
```

#### CAP Theorem
- **Consistency**: All nodes see the same data simultaneously
- **Availability**: System remains operational despite node failures
- **Partition Tolerance**: System continues despite network partitions

*You can only guarantee 2 out of 3 properties in distributed systems*

## 2. Apache Hadoop Ecosystem

### 2.1 Hadoop Distributed File System (HDFS)

#### HDFS Architecture
```python
# Conceptual HDFS implementation
class HDFSConcept:
    """Conceptual representation of HDFS architecture"""

    def __init__(self, block_size: int = 128):
        self.block_size = block_size  # MB
        self.name_node = NameNode()
        self.data_nodes = []

    def add_data_node(self, node_id: str, capacity_gb: int):
        """Add a data node to the cluster"""
        data_node = DataNode(node_id, capacity_gb)
        self.data_nodes.append(data_node)
        print(f"Added DataNode {node_id} with {capacity_gb}GB capacity")

    def store_file(self, filename: str, file_size_mb: int):
        """Store a file in HDFS"""
        # Calculate number of blocks needed
        num_blocks = (file_size_mb + self.block_size - 1) // self.block_size

        # Replicate across data nodes (default replication factor = 3)
        replication_factor = 3
        total_blocks = num_blocks * replication_factor

        print(f"Storing {filename} ({file_size_mb}MB):")
        print(f"  Blocks needed: {num_blocks}")
        print(f"  Total blocks with replication: {total_blocks}")
        print(f"  Block size: {self.block_size}MB")

        # Distribute blocks across available data nodes
        available_nodes = len(self.data_nodes)
        if available_nodes >= replication_factor:
            blocks_per_node = total_blocks // available_nodes
            print(f"  Blocks per node: ~{blocks_per_node}")
        else:
            print("  Warning: Insufficient data nodes for replication"

class NameNode:
    """Represents HDFS NameNode (metadata management)"""
    def __init__(self):
        self.metadata = {}  # filename -> block locations

class DataNode:
    """Represents HDFS DataNode (data storage)"""
    def __init__(self, node_id: str, capacity_gb: int):
        self.node_id = node_id
        self.capacity_gb = capacity_gb
        self.used_gb = 0
        self.blocks = []

# Usage example
hdfs = HDFSConcept(block_size=128)  # 128MB blocks

# Add data nodes
hdfs.add_data_node("datanode1", 1000)  # 1TB
hdfs.add_data_node("datanode2", 1000)  # 1TB
hdfs.add_data_node("datanode3", 1000)  # 1TB

# Store a file
hdfs.store_file("large_dataset.csv", 2048)  # 2GB file
```

#### HDFS Operations with Python
```python
from hdfs import InsecureClient
import pandas as pd

class HDFSClient:
    """HDFS operations client"""

    def __init__(self, namenode_host: str = 'localhost', namenode_port: int = 50070):
        self.client = InsecureClient(f'http://{namenode_host}:{namenode_port}')

    def upload_dataframe(self, df: pd.DataFrame, hdfs_path: str):
        """Upload pandas DataFrame to HDFS"""
        # Convert to CSV string
        csv_data = df.to_csv(index=False)

        # Upload to HDFS
        with self.client.write(hdfs_path, encoding='utf-8') as writer:
            writer.write(csv_data)

        print(f"Uploaded DataFrame ({len(df)} rows) to {hdfs_path}")

    def download_dataframe(self, hdfs_path: str) -> pd.DataFrame:
        """Download data from HDFS as DataFrame"""
        with self.client.read(hdfs_path, encoding='utf-8') as reader:
            df = pd.read_csv(reader)

        print(f"Downloaded DataFrame ({len(df)} rows) from {hdfs_path}")
        return df

    def list_directory(self, hdfs_path: str = '/'):
        """List contents of HDFS directory"""
        try:
            files = self.client.list(hdfs_path)
            print(f"Contents of {hdfs_path}:")
            for file in files:
                status = self.client.status(f"{hdfs_path.rstrip('/')}/{file}")
                size_mb = status['length'] / (1024 * 1024)
                print(f"  {file}: {size_mb:.2f} MB")
            return files
        except Exception as e:
            print(f"Error listing directory: {e}")
            return []

    def get_file_info(self, hdfs_path: str):
        """Get detailed file information"""
        try:
            status = self.client.status(hdfs_path)
            info = {
                'path': hdfs_path,
                'size_bytes': status['length'],
                'size_mb': status['length'] / (1024 * 1024),
                'replication': status.get('replication', 'unknown'),
                'block_size': status.get('blockSize', 'unknown'),
                'modification_time': status.get('modificationTime', 'unknown')
            }

            print(f"File Information for {hdfs_path}:")
            for key, value in info.items():
                print(f"  {key}: {value}")

            return info
        except Exception as e:
            print(f"Error getting file info: {e}")
            return None

# Usage example
# hdfs_client = HDFSClient('namenode-host', 50070)
# hdfs_client.upload_dataframe(my_dataframe, '/data/processed_data.csv')
# downloaded_df = hdfs_client.download_dataframe('/data/processed_data.csv')
```

### 2.2 MapReduce Programming Model

#### MapReduce Concept
```python
from typing import List, Dict, Iterator, Tuple
import collections

def map_function(document: str) -> Iterator[Tuple[str, int]]:
    """Map function: emit word-count pairs"""
    for word in document.lower().split():
        # Remove punctuation
        word = ''.join(c for c in word if c.isalnum())
        if word:
            yield (word, 1)

def reduce_function(word: str, counts: List[int]) -> Tuple[str, int]:
    """Reduce function: sum counts for each word"""
    return (word, sum(counts))

class MapReduceEngine:
    """Simple MapReduce engine implementation"""

    def __init__(self):
        self.intermediate_results = collections.defaultdict(list)

    def map_phase(self, data: List[str]) -> Dict[str, List[int]]:
        """Execute map phase"""
        for document in data:
            for key, value in map_function(document):
                self.intermediate_results[key].append(value)

        return dict(self.intermediate_results)

    def shuffle_sort_phase(self) -> Dict[str, List[int]]:
        """Shuffle and sort intermediate results"""
        # Results are already grouped by key in defaultdict
        return dict(self.intermediate_results)

    def reduce_phase(self, intermediate_data: Dict[str, List[int]]) -> Dict[str, int]:
        """Execute reduce phase"""
        final_results = {}

        for word, counts in intermediate_data.items():
            _, total_count = reduce_function(word, counts)
            final_results[word] = total_count

        return final_results

    def execute(self, data: List[str]) -> Dict[str, int]:
        """Execute complete MapReduce job"""
        # Map phase
        self.map_phase(data)

        # Shuffle and sort phase
        intermediate_data = self.shuffle_sort_phase()

        # Reduce phase
        final_results = self.reduce_phase(intermediate_data)

        return final_results

# Usage example
documents = [
    "Hello world, this is a test document",
    "World hello, another test document here",
    "This is the third document with test words",
    "Hello world, testing the MapReduce functionality"
]

mr_engine = MapReduceEngine()
word_counts = mr_engine.execute(documents)

print("Word Count Results:")
for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {word}: {count}")
```

#### Word Count with Hadoop Streaming
```python
#!/usr/bin/env python3
# mapper.py
import sys

for line in sys.stdin:
    line = line.strip()
    words = line.lower().split()

    for word in words:
        # Remove punctuation
        word = ''.join(c for c in word if c.isalnum())
        if word:
            print(f"{word}\t1")
```

```python
#!/usr/bin/env python3
# reducer.py
import sys
from collections import defaultdict

word_counts = defaultdict(int)

for line in sys.stdin:
    line = line.strip()
    if line:
        word, count = line.split('\t', 1)
        word_counts[word] += int(count)

# Output final counts
for word, count in sorted(word_counts.items()):
    print(f"{word}\t{count}")
```

```bash
# Run MapReduce job
hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-*.jar \
    -input /input/documents.txt \
    -output /output/wordcount \
    -mapper mapper.py \
    -reducer reducer.py \
    -file mapper.py \
    -file reducer.py
```

## 3. Apache Spark

### 3.1 Spark Architecture and RDDs

#### Resilient Distributed Datasets (RDDs)
```python
# PySpark RDD operations
from pyspark.sql import SparkSession
from pyspark import SparkContext

def demonstrate_rdd_operations():
    """Demonstrate basic RDD operations"""

    # Create Spark session
    spark = SparkSession.builder \
        .appName("RDD_Demonstration") \
        .getOrCreate()

    sc = spark.sparkContext

    # Create RDD from list
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rdd = sc.parallelize(data, numSlices=4)

    print(f"RDD partitions: {rdd.getNumPartitions()}")

    # Transformations (lazy operations)
    squared_rdd = rdd.map(lambda x: x * x)
    filtered_rdd = squared_rdd.filter(lambda x: x > 10)

    # Actions (trigger computation)
    results = filtered_rdd.collect()
    count = filtered_rdd.count()

    print(f"Filtered results: {results}")
    print(f"Count of filtered items: {count}")

    # Word count example
    text_data = [
        "hello world spark",
        "world hello hadoop",
        "spark hadoop big data"
    ]

    text_rdd = sc.parallelize(text_data)

    word_counts = text_rdd \
        .flatMap(lambda line: line.split()) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a + b) \
        .collect()

    print("Word counts:")
    for word, count in word_counts:
        print(f"  {word}: {count}")

    spark.stop()

# Usage
# demonstrate_rdd_operations()
```

#### DataFrame API
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, sum

def demonstrate_dataframe_operations():
    """Demonstrate Spark DataFrame operations"""

    spark = SparkSession.builder \
        .appName("DataFrame_Demonstration") \
        .getOrCreate()

    # Create sample data
    data = [
        ("Alice", 25, "Engineering", 75000),
        ("Bob", 30, "Marketing", 65000),
        ("Charlie", 35, "Engineering", 85000),
        ("Diana", 28, "Sales", 55000),
        ("Eve", 32, "Engineering", 95000)
    ]

    columns = ["name", "age", "department", "salary"]

    df = spark.createDataFrame(data, columns)

    print("Original DataFrame:")
    df.show()

    # DataFrame operations
    print("\nDepartment-wise statistics:")
    dept_stats = df.groupBy("department") \
        .agg(
            count("name").alias("employee_count"),
            avg("age").alias("avg_age"),
            avg("salary").alias("avg_salary")
        )
    dept_stats.show()

    # Filter and sort
    print("\nEngineering employees with salary > 80000:")
    high_paid_engineers = df \
        .filter((col("department") == "Engineering") & (col("salary") > 80000)) \
        .orderBy(col("salary").desc())
    high_paid_engineers.show()

    # SQL queries
    df.createOrReplaceTempView("employees")

    print("\nSQL Query - Average salary by department:")
    sql_result = spark.sql("""
        SELECT department,
               COUNT(*) as employee_count,
               AVG(salary) as avg_salary,
               MAX(salary) as max_salary
        FROM employees
        GROUP BY department
        ORDER BY avg_salary DESC
    """)
    sql_result.show()

    spark.stop()

# Usage
# demonstrate_dataframe_operations()
```

### 3.2 Spark SQL and DataFrames

#### Advanced DataFrame Operations
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

def advanced_dataframe_operations():
    """Demonstrate advanced DataFrame operations"""

    spark = SparkSession.builder \
        .appName("Advanced_DataFrame") \
        .getOrCreate()

    # Create sample sales data
    sales_data = [
        ("2023-01-01", "Product_A", "North", 100, 10.0),
        ("2023-01-01", "Product_B", "North", 150, 15.0),
        ("2023-01-01", "Product_A", "South", 200, 10.0),
        ("2023-01-02", "Product_A", "North", 120, 10.0),
        ("2023-01-02", "Product_B", "South", 180, 15.0),
        ("2023-01-03", "Product_A", "North", 90, 10.0),
        ("2023-01-03", "Product_B", "North", 160, 15.0),
    ]

    sales_df = spark.createDataFrame(sales_data,
                                    ["date", "product", "region", "quantity", "price"])

    # Add calculated columns
    sales_df = sales_df.withColumn("revenue", col("quantity") * col("price"))
    sales_df = sales_df.withColumn("date", to_date(col("date")))

    print("Sales Data with Revenue:")
    sales_df.show()

    # Window functions
    window_spec = Window.partitionBy("product").orderBy("date")

    sales_with_window = sales_df \
        .withColumn("running_total", sum("revenue").over(window_spec)) \
        .withColumn("product_rank", rank().over(Window.orderBy(desc("revenue"))))

    print("\nSales Data with Window Functions:")
    sales_with_window.show()

    # Pivot operations
    pivot_df = sales_df.groupBy("date") \
        .pivot("product") \
        .agg(sum("revenue").alias("total_revenue"))

    print("\nPivoted Sales Data:")
    pivot_df.show()

    # Complex aggregations
    complex_agg = sales_df.groupBy("region", "product") \
        .agg(
            sum("revenue").alias("total_revenue"),
            avg("quantity").alias("avg_quantity"),
            count("*").alias("transaction_count"),
            max("revenue").alias("max_transaction")
        ) \
        .orderBy(desc("total_revenue"))

    print("\nComplex Aggregations by Region and Product:")
    complex_agg.show()

    spark.stop()

# Usage
# advanced_dataframe_operations()
```

### 3.3 Spark Streaming

#### Real-time Data Processing
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

def create_spark_streaming_app():
    """Create a Spark Streaming application"""

    spark = SparkSession.builder \
        .appName("Streaming_Analysis") \
        .getOrCreate()

    # Define schema for streaming data
    schema = StructType([
        StructField("timestamp", TimestampType(), True),
        StructField("user_id", StringType(), True),
        StructField("event_type", StringType(), True),
        StructField("value", DoubleType(), True)
    ])

    # Create streaming DataFrame (from Kafka, socket, etc.)
    # For demonstration, we'll simulate streaming

    # Example: Read from Kafka
    """
    streaming_df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "user_events") \
        .load()

    # Parse JSON data
    parsed_df = streaming_df \
        .select(from_json(col("value").cast("string"), schema).alias("data")) \
        .select("data.*")
    """

    # Simulate streaming with static data for demonstration
    static_data = [
        ("2023-01-01 10:00:00", "user1", "click", 1.0),
        ("2023-01-01 10:01:00", "user2", "purchase", 99.99),
        ("2023-01-01 10:02:00", "user1", "view", 0.0),
    ]

    static_df = spark.createDataFrame(static_data, ["timestamp", "user_id", "event_type", "value"])
    static_df = static_df.withColumn("timestamp", to_timestamp(col("timestamp")))

    # Streaming-like operations on static data
    print("Static Data Analysis (simulating streaming):")
    static_df.show()

    # Real-time aggregations
    real_time_agg = static_df \
        .groupBy(
            window(col("timestamp"), "1 hour"),  # 1-hour tumbling windows
            "event_type"
        ) \
        .agg(
            count("*").alias("event_count"),
            sum("value").alias("total_value"),
            avg("value").alias("avg_value")
        )

    print("\nReal-time Aggregations:")
    real_time_agg.show()

    # User session analysis
    user_sessions = static_df \
        .withWatermark("timestamp", "10 minutes") \
        .groupBy(
            "user_id",
            window(col("timestamp"), "30 minutes")
        ) \
        .agg(
            count("*").alias("session_events"),
            sum("value").alias("session_value"),
            min("timestamp").alias("session_start"),
            max("timestamp").alias("session_end")
        )

    print("\nUser Session Analysis:")
    user_sessions.show()

    spark.stop()

# Usage
# create_spark_streaming_app()
```

## 4. Apache Kafka

### 4.1 Kafka Architecture

#### Producer-Consumer Pattern
```python
from kafka import KafkaProducer, KafkaConsumer
import json
import time
from typing import Dict, Any

class KafkaManager:
    """Kafka producer and consumer management"""

    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        self.bootstrap_servers = bootstrap_servers

    def create_producer(self) -> KafkaProducer:
        """Create Kafka producer"""
        producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: str(k).encode('utf-8') if k else None
        )
        return producer

    def create_consumer(self, topic: str, group_id: str = 'default_group') -> KafkaConsumer:
        """Create Kafka consumer"""
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            key_deserializer=lambda x: x.decode('utf-8') if x else None,
            auto_offset_reset='earliest',
            enable_auto_commit=True
        )
        return consumer

    def produce_messages(self, topic: str, messages: list, delay: float = 0.1):
        """Produce messages to Kafka topic"""
        producer = self.create_producer()

        for message in messages:
            key = message.get('key')
            value = message.get('value', message)

            future = producer.send(topic, key=key, value=value)
            record_metadata = future.get(timeout=10)

            print(f"Produced message to {record_metadata.topic} "
                  f"partition {record_metadata.partition} "
                  f"offset {record_metadata.offset}")

            time.sleep(delay)

        producer.close()

    def consume_messages(self, topic: str, group_id: str = 'default_group',
                        max_messages: int = 10, timeout: float = 10.0):
        """Consume messages from Kafka topic"""
        consumer = self.create_consumer(topic, group_id)

        messages_consumed = 0
        start_time = time.time()

        try:
            while messages_consumed < max_messages and (time.time() - start_time) < timeout:
                message_batch = consumer.poll(timeout_ms=1000, max_records=10)

                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        print(f"Consumed message: key={message.key}, "
                              f"value={message.value}, "
                              f"partition={message.partition}, "
                              f"offset={message.offset}")

                        messages_consumed += 1
                        if messages_consumed >= max_messages:
                            break

                    if messages_consumed >= max_messages:
                        break

        finally:
            consumer.close()

# Usage example
kafka_manager = KafkaManager()

# Sample messages
sample_messages = [
    {'key': 'user1', 'value': {'event': 'login', 'timestamp': '2023-01-01T10:00:00'}},
    {'key': 'user2', 'value': {'event': 'purchase', 'amount': 99.99, 'timestamp': '2023-01-01T10:01:00'}},
    {'key': 'user1', 'value': {'event': 'logout', 'timestamp': '2023-01-01T10:30:00'}},
]

# Produce messages
print("Producing messages to Kafka...")
# kafka_manager.produce_messages('user_events', sample_messages)

# Consume messages
print("\nConsuming messages from Kafka...")
# kafka_manager.consume_messages('user_events', max_messages=5)
```

### 4.2 Stream Processing with Kafka Streams

#### Real-time Analytics Pipeline
```python
from kafka import KafkaConsumer, KafkaProducer
import json
from collections import defaultdict
import time

class RealTimeAnalytics:
    """Real-time analytics using Kafka"""

    def __init__(self, input_topic: str, output_topic: str):
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.kafka_manager = KafkaManager()

        # Analytics state
        self.user_sessions = defaultdict(list)
        self.event_counts = defaultdict(int)

    def process_stream(self):
        """Process streaming data in real-time"""
        consumer = self.kafka_manager.create_consumer(self.input_topic, 'analytics_group')
        producer = self.kafka_manager.create_producer()

        print(f"Starting real-time analytics on topic: {self.input_topic}")

        try:
            for message in consumer:
                event_data = message.value

                # Process event
                analytics = self._process_event(event_data)

                if analytics:
                    # Send analytics to output topic
                    producer.send(
                        self.output_topic,
                        key=event_data.get('user_id'),
                        value=analytics
                    )

                    print(f"Processed event for user {event_data.get('user_id')}: {analytics}")

        except KeyboardInterrupt:
            print("Stopping real-time analytics...")
        finally:
            consumer.close()
            producer.close()

    def _process_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual event and compute analytics"""
        user_id = event_data.get('user_id')
        event_type = event_data.get('event_type')
        timestamp = event_data.get('timestamp')

        if not user_id or not event_type:
            return None

        # Update event counts
        self.event_counts[event_type] += 1

        # Track user session (simplified)
        self.user_sessions[user_id].append({
            'event_type': event_type,
            'timestamp': timestamp
        })

        # Keep only recent events (last 10 per user)
        if len(self.user_sessions[user_id]) > 10:
            self.user_sessions[user_id] = self.user_sessions[user_id][-10:]

        # Compute analytics
        analytics = {
            'user_id': user_id,
            'total_events': len(self.user_sessions[user_id]),
            'event_types': list(set(event['event_type'] for event in self.user_sessions[user_id])),
            'last_event': event_type,
            'last_timestamp': timestamp,
            'global_event_counts': dict(self.event_counts)
        }

        return analytics

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current analytics statistics"""
        return {
            'total_users': len(self.user_sessions),
            'total_events_processed': sum(self.event_counts.values()),
            'event_type_distribution': dict(self.event_counts),
            'active_users': len([u for u, events in self.user_sessions.items()
                               if len(events) > 0])
        }

# Usage example
# analytics = RealTimeAnalytics('user_events', 'analytics_output')
# analytics.process_stream()
```

## 5. NoSQL Databases for Big Data

### 5.1 Cassandra for High Write Throughput

#### Cassandra Data Modeling
```python
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import uuid
from datetime import datetime

class CassandraManager:
    """Cassandra database operations for big data"""

    def __init__(self, hosts: list = ['localhost'], keyspace: str = 'bigdata'):
        self.hosts = hosts
        self.keyspace = keyspace
        self.cluster = None
        self.session = None

    def connect(self):
        """Connect to Cassandra cluster"""
        try:
            self.cluster = Cluster(self.hosts)
            self.session = self.cluster.connect()

            # Create keyspace if it doesn't exist
            self.session.execute(f"""
                CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
                WITH REPLICATION = {{
                    'class': 'SimpleStrategy',
                    'replication_factor': 1
                }}
            """)

            self.session.set_keyspace(self.keyspace)
            print(f"Connected to Cassandra keyspace: {self.keyspace}")

        except Exception as e:
            print(f"Cassandra connection failed: {e}")

    def create_tables(self):
        """Create sample tables for big data analytics"""

        # User events table (time series)
        self.session.execute("""
            CREATE TABLE IF NOT EXISTS user_events (
                user_id text,
                event_time timestamp,
                event_type text,
                event_data text,
                PRIMARY KEY (user_id, event_time)
            ) WITH CLUSTERING ORDER BY (event_time DESC)
        """)

        # Product analytics table
        self.session.execute("""
            CREATE TABLE IF NOT EXISTS product_analytics (
                product_id text,
                date date,
                views counter,
                purchases counter,
                revenue counter,
                PRIMARY KEY (product_id, date)
            )
        """)

        print("Tables created successfully")

    def insert_user_event(self, user_id: str, event_type: str, event_data: dict):
        """Insert user event"""
        event_time = datetime.now()

        self.session.execute("""
            INSERT INTO user_events (user_id, event_time, event_type, event_data)
            VALUES (%s, %s, %s, %s)
        """, (user_id, event_time, event_type, json.dumps(event_data)))

    def get_user_events(self, user_id: str, limit: int = 10):
        """Retrieve user events"""
        rows = self.session.execute("""
            SELECT user_id, event_time, event_type, event_data
            FROM user_events
            WHERE user_id = %s
            LIMIT %s
        """, (user_id, limit))

        events = []
        for row in rows:
            events.append({
                'user_id': row.user_id,
                'event_time': row.event_time,
                'event_type': row.event_type,
                'event_data': json.loads(row.event_data)
            })

        return events

    def update_product_stats(self, product_id: str, date, views: int = 0,
                           purchases: int = 0, revenue: int = 0):
        """Update product analytics (using counters)"""
        self.session.execute("""
            UPDATE product_analytics
            SET views = views + %s,
                purchases = purchases + %s,
                revenue = revenue + %s
            WHERE product_id = %s AND date = %s
        """, (views, purchases, revenue, product_id, date))

    def close(self):
        """Close Cassandra connection"""
        if self.cluster:
            self.cluster.shutdown()
            print("Cassandra connection closed")

# Usage example
# cassandra_manager = CassandraManager()
# cassandra_manager.connect()
# cassandra_manager.create_tables()

# Insert sample data
# cassandra_manager.insert_user_event('user123', 'purchase',
#                                   {'product_id': 'prod456', 'amount': 99.99})
# events = cassandra_manager.get_user_events('user123')
# cassandra_manager.close()
```

### 5.2 Elasticsearch for Search and Analytics

#### Elasticsearch Operations
```python
from elasticsearch import Elasticsearch
import json

class ElasticsearchManager:
    """Elasticsearch operations for big data analytics"""

    def __init__(self, hosts: list = ['localhost:9200']):
        self.es = Elasticsearch(hosts)
        self.index_name = 'bigdata_analytics'

    def create_index(self):
        """Create Elasticsearch index with mappings"""
        mappings = {
            "mappings": {
                "properties": {
                    "user_id": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "event_type": {"type": "keyword"},
                    "event_data": {"type": "object"},
                    "location": {"type": "geo_point"},
                    "tags": {"type": "keyword"}
                }
            }
        }

        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, body=mappings)
            print(f"Created index: {self.index_name}")
        else:
            print(f"Index {self.index_name} already exists")

    def index_document(self, document: dict):
        """Index a document"""
        response = self.es.index(index=self.index_name, body=document)
        return response['_id']

    def bulk_index(self, documents: list):
        """Bulk index multiple documents"""
        actions = []
        for doc in documents:
            actions.append({"_index": self.index_name, "_source": doc})

        from elasticsearch.helpers import bulk
        success, failed = bulk(self.es, actions)
        print(f"Bulk indexed: {success} success, {failed} failed")
        return success, failed

    def search_documents(self, query: dict, size: int = 10):
        """Search documents"""
        response = self.es.search(index=self.index_name, body=query, size=size)
        return response['hits']['hits']

    def advanced_search(self, user_id: str = None, event_type: str = None,
                       date_range: dict = None, size: int = 10):
        """Perform advanced search"""

        query = {"query": {"bool": {"must": []}}}

        if user_id:
            query["query"]["bool"]["must"].append({"term": {"user_id": user_id}})

        if event_type:
            query["query"]["bool"]["must"].append({"term": {"event_type": event_type}})

        if date_range:
            query["query"]["bool"]["must"].append({
                "range": {"timestamp": date_range}
            })

        results = self.search_documents(query, size)
        return results

    def aggregate_data(self, field: str, agg_type: str = 'terms', size: int = 10):
        """Perform aggregations"""
        query = {
            "size": 0,
            "aggs": {
                f"{field}_agg": {
                    agg_type: {"field": field, "size": size}
                }
            }
        }

        response = self.es.search(index=self.index_name, body=query)
        return response['aggregations'][f'{field}_agg']

# Usage example
# es_manager = ElasticsearchManager()
# es_manager.create_index()

# Index sample documents
# sample_docs = [
#     {
#         "user_id": "user123",
#         "timestamp": "2023-01-01T10:00:00",
#         "event_type": "purchase",
#         "event_data": {"product_id": "prod456", "amount": 99.99},
#         "location": {"lat": 40.7128, "lon": -74.0060},
#         "tags": ["electronics", "premium"]
#     }
# ]
# es_manager.bulk_index(sample_docs)

# Search documents
# results = es_manager.advanced_search(user_id="user123", event_type="purchase")
# print(f"Found {len(results)} matching documents")
```

## 6. Data Lake Architecture

### 6.1 Modern Data Lake Design

#### Lambda Architecture
```python
# Conceptual Lambda Architecture implementation
class LambdaArchitecture:
    """Lambda Architecture for big data processing"""

    def __init__(self):
        self.speed_layer = SpeedLayer()      # Real-time processing
        self.batch_layer = BatchLayer()      # Batch processing
        self.serving_layer = ServingLayer()  # Query serving

    def process_data(self, data_stream):
        """Process data through all layers"""

        # Speed layer: Real-time processing
        real_time_results = self.speed_layer.process_stream(data_stream)

        # Batch layer: Batch processing
        batch_results = self.batch_layer.process_batch(data_stream)

        # Serving layer: Merge results
        final_results = self.serving_layer.merge_results(
            real_time_results, batch_results
        )

        return final_results

class SpeedLayer:
    """Real-time processing layer"""
    def __init__(self):
        self.recent_data = {}

    def process_stream(self, data_stream):
        """Process streaming data"""
        results = []

        for data_point in data_stream:
            # Real-time aggregations
            user_id = data_point.get('user_id')
            if user_id not in self.recent_data:
                self.recent_data[user_id] = []

            self.recent_data[user_id].append(data_point)

            # Keep only recent data (last hour)
            # In practice, use time windows

            # Compute real-time metrics
            user_metrics = self._compute_real_time_metrics(user_id)
            results.append(user_metrics)

        return results

    def _compute_real_time_metrics(self, user_id):
        """Compute real-time metrics for user"""
        user_data = self.recent_data.get(user_id, [])

        return {
            'user_id': user_id,
            'recent_events': len(user_data),
            'last_event_time': user_data[-1].get('timestamp') if user_data else None
        }

class BatchLayer:
    """Batch processing layer"""
    def __init__(self):
        self.batch_results = {}

    def process_batch(self, data_stream):
        """Process data in batches"""
        # In practice, this would run periodically (daily/hourly)

        # Simulate batch processing
        batch_metrics = {}

        for data_point in data_stream:
            user_id = data_point.get('user_id')
            if user_id not in batch_metrics:
                batch_metrics[user_id] = {
                    'total_events': 0,
                    'total_value': 0,
                    'first_event': None,
                    'last_event': None
                }

            metrics = batch_metrics[user_id]
            metrics['total_events'] += 1
            metrics['total_value'] += data_point.get('value', 0)

            if not metrics['first_event']:
                metrics['first_event'] = data_point.get('timestamp')
            metrics['last_event'] = data_point.get('timestamp')

        self.batch_results = batch_metrics
        return batch_metrics

class ServingLayer:
    """Query serving layer"""
    def __init__(self):
        self.merged_results = {}

    def merge_results(self, real_time_results, batch_results):
        """Merge real-time and batch results"""
        merged = {}

        # Start with batch results
        for user_id, batch_data in batch_results.items():
            merged[user_id] = batch_data.copy()

        # Overlay real-time results
        for rt_result in real_time_results:
            user_id = rt_result['user_id']
            if user_id in merged:
                merged[user_id].update(rt_result)
            else:
                merged[user_id] = rt_result

        self.merged_results = merged
        return merged

# Usage example
lambda_arch = LambdaArchitecture()

# Simulate data stream
data_stream = [
    {'user_id': 'user1', 'event': 'login', 'value': 0, 'timestamp': '2023-01-01T10:00:00'},
    {'user_id': 'user1', 'event': 'purchase', 'value': 99.99, 'timestamp': '2023-01-01T10:05:00'},
    {'user_id': 'user2', 'event': 'login', 'value': 0, 'timestamp': '2023-01-01T10:10:00'},
]

results = lambda_arch.process_data(data_stream)
print("Lambda Architecture Results:")
for user_id, metrics in results.items():
    print(f"User {user_id}: {metrics}")
```

## 7. Best Practices and Performance Optimization

### 7.1 Big Data Performance Tuning

#### Spark Optimization Techniques
```python
def optimize_spark_job(spark_df):
    """Apply Spark optimization techniques"""

    # 1. Caching frequently used DataFrames
    spark_df.cache()

    # 2. Repartitioning for better parallelism
    optimal_partitions = spark_df.rdd.getNumPartitions() * 2
    spark_df = spark_df.repartition(optimal_partitions)

    # 3. Using broadcast joins for small DataFrames
    small_df = spark_df.limit(1000)
    large_df = spark_df  # Assume this is large

    # Broadcast the small DataFrame
    from pyspark.sql.functions import broadcast
    result = large_df.join(broadcast(small_df), "join_key")

    # 4. Predicate pushdown
    filtered_df = spark_df.filter("column > 100")

    # 5. Column pruning
    selected_df = spark_df.select("col1", "col2", "col3")

    return result

# Memory management
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
```

#### Data Partitioning Strategies
```python
def implement_data_partitioning():
    """Implement effective data partitioning strategies"""

    # 1. Time-based partitioning
    time_partitioned_data = {
        'year=2023/month=01/day=01': ['data_file_1.parquet', 'data_file_2.parquet'],
        'year=2023/month=01/day=02': ['data_file_3.parquet'],
        'year=2023/month=02/day=01': ['data_file_4.parquet', 'data_file_5.parquet']
    }

    # 2. Hash-based partitioning
    def hash_partition(key, num_partitions):
        """Simple hash partitioning"""
        return hash(key) % num_partitions

    # 3. Range partitioning
    def range_partition(value, ranges):
        """Range-based partitioning"""
        for i, (min_val, max_val) in enumerate(ranges):
            if min_val <= value < max_val:
                return i
        return len(ranges) - 1  # Last partition for overflow

    # Example usage
    user_ids = ['user1', 'user2', 'user3', 'user4', 'user5']
    num_partitions = 3

    partitions = {}
    for user_id in user_ids:
        partition_id = hash_partition(user_id, num_partitions)
        if partition_id not in partitions:
            partitions[partition_id] = []
        partitions[partition_id].append(user_id)

    print("Hash-based partitioning:")
    for partition_id, users in partitions.items():
        print(f"Partition {partition_id}: {users}")

    return partitions
```

## 8. Resources and Further Reading

### Books
- "Hadoop: The Definitive Guide" by Tom White
- "Learning Spark" by Holden Karau, Andy Konwinski, Patrick Wendell, Matei Zaharia
- "Designing Data-Intensive Applications" by Martin Kleppmann

### Online Courses
- Coursera: Google Cloud Big Data and Machine Learning Fundamentals
- edX: Big Data Analytics Using Spark
- Udacity: Data Engineering Nanodegree

### Tools and Frameworks
- **Apache Hadoop**: Distributed storage and processing
- **Apache Spark**: Fast big data processing
- **Apache Kafka**: Distributed streaming platform
- **Apache Cassandra**: Highly scalable NoSQL database
- **Elasticsearch**: Distributed search and analytics engine

### Cloud Platforms
- **AWS EMR**: Managed Hadoop and Spark clusters
- **Google Dataproc**: Managed Spark and Hadoop service
- **Azure HDInsight**: Cloud-based big data analytics
- **Databricks**: Unified analytics platform

## Next Steps

Congratulations on mastering big data technologies! You now understand distributed computing, Hadoop, Spark, Kafka, and modern data architectures. In the next module, we'll explore cloud computing platforms for scalable data science.

**Ready to continue?** Proceed to [Module 11: Cloud Computing](../11_cloud_computing/)
