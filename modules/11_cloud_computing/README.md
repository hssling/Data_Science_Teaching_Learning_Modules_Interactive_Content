# Module 11: Cloud Computing for Data Science

## Overview
Cloud computing has revolutionized data science by providing scalable, on-demand computing resources and specialized services for data processing, machine learning, and analytics. This module covers major cloud platforms (AWS, Google Cloud, Azure), their data science services, deployment strategies, and cost optimization techniques.

## Learning Objectives
By the end of this module, you will be able to:
- Understand cloud computing fundamentals and service models
- Work with AWS, Google Cloud, and Azure data science services
- Deploy machine learning models in the cloud
- Implement serverless data processing pipelines
- Optimize cloud costs for data science workloads
- Choose appropriate cloud services for different use cases
- Implement security and compliance in cloud environments

## 1. Cloud Computing Fundamentals

### 1.1 Cloud Service Models

#### Infrastructure as a Service (IaaS)
- **Definition**: Virtualized computing resources over the internet
- **Examples**: EC2 (AWS), Compute Engine (GCP), Virtual Machines (Azure)
- **Use Cases**: Custom infrastructure, full control, legacy applications
- **Benefits**: Maximum flexibility, pay for what you use

#### Platform as a Service (PaaS)
- **Definition**: Platform and tools for application development
- **Examples**: Elastic Beanstalk (AWS), App Engine (GCP), App Service (Azure)
- **Use Cases**: Web applications, APIs, microservices
- **Benefits**: Faster development, managed infrastructure

#### Software as a Service (SaaS)
- **Definition**: Complete software applications delivered over the internet
- **Examples**: Salesforce, Office 365, Gmail
- **Use Cases**: Business applications, collaboration tools
- **Benefits**: No installation, automatic updates

#### Function as a Service (FaaS)/Serverless
- **Definition**: Run code in response to events without managing servers
- **Examples**: Lambda (AWS), Cloud Functions (GCP), Functions (Azure)
- **Use Cases**: Event-driven processing, APIs, scheduled tasks
- **Benefits**: Auto-scaling, pay-per-execution, zero maintenance

### 1.2 Cloud Deployment Models

#### Public Cloud
- **Definition**: Services offered by third-party providers over the internet
- **Examples**: AWS, Google Cloud, Azure
- **Benefits**: Cost-effective, scalable, globally distributed
- **Considerations**: Security, compliance, vendor lock-in

#### Private Cloud
- **Definition**: Cloud infrastructure dedicated to a single organization
- **Examples**: OpenStack, VMware Cloud
- **Benefits**: Enhanced security, customization, compliance
- **Considerations**: Higher costs, management overhead

#### Hybrid Cloud
- **Definition**: Combination of public and private cloud
- **Benefits**: Flexibility, cost optimization, gradual migration
- **Use Cases**: Sensitive data in private, scalable workloads in public

## 2. Amazon Web Services (AWS)

### 2.1 Core AWS Services for Data Science

#### Compute Services
```python
import boto3
from botocore.exceptions import ClientError

class AWSEC2Manager:
    """AWS EC2 instance management for data science"""

    def __init__(self, region: str = 'us-east-1'):
        self.ec2 = boto3.client('ec2', region_name=region)
        self.ec2_resource = boto3.resource('ec2', region_name=region)

    def create_data_science_instance(self, instance_type: str = 't3.medium',
                                   ami_id: str = 'ami-0abcdef1234567890'):
        """Create an EC2 instance optimized for data science"""

        try:
            response = self.ec2.run_instances(
                ImageId=ami_id,  # Deep Learning AMI
                MinCount=1,
                MaxCount=1,
                InstanceType=instance_type,
                KeyName='data-science-key',
                SecurityGroupIds=['sg-12345678'],
                UserData="""#!/bin/bash
                yum update -y
                yum install -y python3 pip
                pip3 install jupyter numpy pandas scikit-learn tensorflow
                """,
                TagSpecifications=[
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'Name', 'Value': 'DataScience-Instance'},
                            {'Key': 'Environment', 'Value': 'Development'}
                        ]
                    }
                ]
            )

            instance_id = response['Instances'][0]['InstanceId']
            print(f"Created EC2 instance: {instance_id}")

            return instance_id

        except ClientError as e:
            print(f"Error creating instance: {e}")
            return None

    def list_instances(self):
        """List all EC2 instances"""
        try:
            response = self.ec2.describe_instances()
            instances = []

            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    instance_info = {
                        'InstanceId': instance['InstanceId'],
                        'State': instance['State']['Name'],
                        'InstanceType': instance['InstanceType'],
                        'LaunchTime': instance['LaunchTime']
                    }
                    instances.append(instance_info)

            return instances

        except ClientError as e:
            print(f"Error listing instances: {e}")
            return []

# Usage
# ec2_manager = AWSEC2Manager()
# instance_id = ec2_manager.create_data_science_instance()
# instances = ec2_manager.list_instances()
```

#### Storage Services
```python
class AWSStorageManager:
    """AWS storage services for data science"""

    def __init__(self, region: str = 'us-east-1'):
        self.s3 = boto3.client('s3', region_name=region)
        self.glacier = boto3.client('glacier', region_name=region)

    def create_s3_bucket(self, bucket_name: str):
        """Create an S3 bucket for data storage"""

        try:
            # Create bucket
            if region == 'us-east-1':
                self.s3.create_bucket(Bucket=bucket_name)
            else:
                self.s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': region}
                )

            # Enable versioning
            self.s3.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )

            print(f"Created S3 bucket: {bucket_name}")
            return True

        except ClientError as e:
            print(f"Error creating bucket: {e}")
            return False

    def upload_dataset(self, bucket_name: str, file_path: str, s3_key: str):
        """Upload dataset to S3"""

        try:
            self.s3.upload_file(file_path, bucket_name, s3_key)
            print(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")

            # Generate presigned URL for sharing
            presigned_url = self.s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': s3_key},
                ExpiresIn=3600  # 1 hour
            )

            return presigned_url

        except ClientError as e:
            print(f"Error uploading file: {e}")
            return None

    def archive_old_data(self, vault_name: str, file_path: str):
        """Archive old data to Glacier"""

        try:
            with open(file_path, 'rb') as f:
                archive_response = self.glacier.upload_archive(
                    vaultName=vault_name,
                    body=f
                )

            archive_id = archive_response['archiveId']
            print(f"Archived {file_path} to Glacier vault {vault_name}")
            print(f"Archive ID: {archive_id}")

            return archive_id

        except ClientError as e:
            print(f"Error archiving to Glacier: {e}")
            return None

# Usage
# storage_manager = AWSStorageManager()
# storage_manager.create_s3_bucket('my-data-science-bucket')
# presigned_url = storage_manager.upload_dataset('my-bucket', 'data.csv', 'datasets/data.csv')
```

### 2.2 AWS Machine Learning Services

#### SageMaker for Model Development
```python
import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator
import boto3

class SageMakerManager:
    """AWS SageMaker operations for machine learning"""

    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.role = get_execution_role()
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)

    def create_training_job(self, training_data_s3: str, output_s3: str,
                          instance_type: str = 'ml.m5.large', instance_count: int = 1):
        """Create a SageMaker training job"""

        # Define algorithm (using built-in XGBoost)
        algorithm_specification = {
            'TrainingImage': sagemaker.image_uris.retrieve('xgboost', self.region, '1.5-1'),
            'TrainingInputMode': 'File'
        }

        # Define input data
        input_data_config = [
            {
                'ChannelName': 'train',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': training_data_s3,
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                }
            }
        ]

        # Define output data
        output_data_config = {
            'S3OutputPath': output_s3
        }

        # Define resource configuration
        resource_config = {
            'InstanceType': instance_type,
            'InstanceCount': instance_count,
            'VolumeSizeInGB': 10
        }

        # Create training job
        training_job_name = f'xgb-training-{int(time.time())}'

        self.sagemaker_client.create_training_job(
            TrainingJobName=training_job_name,
            AlgorithmSpecification=algorithm_specification,
            RoleArn=self.role,
            InputDataConfig=input_data_config,
            OutputDataConfig=output_data_config,
            ResourceConfig=resource_config,
            StoppingCondition={'MaxRuntimeInSeconds': 3600}
        )

        print(f"Created training job: {training_job_name}")
        return training_job_name

    def deploy_model(self, model_name: str, training_job_name: str,
                    instance_type: str = 'ml.t2.medium', initial_instance_count: int = 1):
        """Deploy a trained model as an endpoint"""

        # Create model
        model_response = self.sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': sagemaker.image_uris.retrieve('xgboost', self.region, '1.5-1'),
                'ModelDataUrl': f's3://my-bucket/models/{training_job_name}/output/model.tar.gz'
            },
            ExecutionRoleArn=self.role
        )

        # Create endpoint configuration
        endpoint_config_name = f'{model_name}-config'
        self.sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'primary',
                    'ModelName': model_name,
                    'InitialInstanceCount': initial_instance_count,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1
                }
            ]
        )

        # Create endpoint
        endpoint_name = f'{model_name}-endpoint'
        self.sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )

        print(f"Deployed model as endpoint: {endpoint_name}")
        return endpoint_name

    def predict_with_endpoint(self, endpoint_name: str, input_data):
        """Make predictions using deployed endpoint"""

        runtime = boto3.client('sagemaker-runtime', region_name=self.region)

        # Convert input to CSV format (for XGBoost)
        import io
        csv_buffer = io.StringIO()
        pd.DataFrame(input_data).to_csv(csv_buffer, index=False, header=False)
        csv_data = csv_buffer.getvalue()

        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/csv',
            Body=csv_data
        )

        predictions = response['Body'].read().decode('utf-8')
        return predictions

# Usage
# sagemaker_manager = SageMakerManager()
# training_job = sagemaker_manager.create_training_job(
#     's3://my-bucket/data/train/', 's3://my-bucket/models/')
# endpoint = sagemaker_manager.deploy_model('my-model', training_job)
```

#### AWS Lambda for Serverless Computing
```python
import boto3
import zipfile
import json
from pathlib import Path

class LambdaManager:
    """AWS Lambda functions for serverless data processing"""

    def __init__(self, region: str = 'us-east-1'):
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.iam_client = boto3.client('iam', region_name=region)

    def create_lambda_role(self, role_name: str = 'lambda-data-processing-role'):
        """Create IAM role for Lambda function"""

        assume_role_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }

        try:
            # Create role
            role_response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy),
                Description='Role for Lambda data processing functions'
            )

            # Attach basic execution role
            self.iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
            )

            # Attach S3 access policy
            self.iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess'
            )

            print(f"Created IAM role: {role_name}")
            return role_response['Role']['Arn']

        except self.iam_client.exceptions.EntityAlreadyExistsException:
            # Role already exists, get its ARN
            role_response = self.iam_client.get_role(RoleName=role_name)
            return role_response['Role']['Arn']

    def create_lambda_function(self, function_name: str, handler_code: str,
                             role_arn: str, timeout: int = 300):
        """Create a Lambda function for data processing"""

        # Create deployment package
        zip_buffer = self._create_deployment_package(handler_code)

        try:
            response = self.lambda_client.create_function(
                FunctionName=function_name,
                Runtime='python3.9',
                Role=role_arn,
                Handler='lambda_function.lambda_handler',
                Code={'ZipFile': zip_buffer.getvalue()},
                Description='Data processing Lambda function',
                Timeout=timeout,
                MemorySize=1024,  # 1GB
                Environment={
                    'Variables': {
                        'ENVIRONMENT': 'production',
                        'LOG_LEVEL': 'INFO'
                    }
                }
            )

            print(f"Created Lambda function: {function_name}")
            return response['FunctionArn']

        except self.lambda_client.exceptions.ResourceConflictException:
            print(f"Function {function_name} already exists")
            return None

    def _create_deployment_package(self, handler_code: str):
        """Create ZIP deployment package"""

        import io

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add handler code
            zip_file.writestr('lambda_function.py', handler_code)

            # Add requirements (if any)
            requirements = """
pandas==1.5.0
numpy==1.21.0
boto3==1.26.0
"""
            zip_file.writestr('requirements.txt', requirements)

        zip_buffer.seek(0)
        return zip_buffer

    def invoke_lambda_function(self, function_name: str, payload: dict):
        """Invoke Lambda function"""

        try:
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )

            # Parse response
            response_payload = json.loads(response['Payload'].read())
            return response_payload

        except Exception as e:
            print(f"Error invoking Lambda function: {e}")
            return None

# Example Lambda handler code
lambda_handler_code = '''
import json
import boto3
import pandas as pd
from io import StringIO

def lambda_handler(event, context):
    """Lambda function for data processing"""

    try:
        # Get S3 bucket and key from event
        bucket = event.get('bucket')
        key = event.get('key')

        if not bucket or not key:
            return {
                'statusCode': 400,
                'body': json.dumps('Missing bucket or key parameters')
            }

        # Download data from S3
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = pd.read_csv(obj['Body'])

        # Perform data processing
        processed_data = process_data(data)

        # Save processed data back to S3
        output_key = key.replace('.csv', '_processed.csv')
        csv_buffer = StringIO()
        processed_data.to_csv(csv_buffer, index=False)

        s3.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=csv_buffer.getvalue()
        )

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Data processed successfully',
                'input_file': key,
                'output_file': output_key,
                'rows_processed': len(processed_data)
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }

def process_data(df):
    """Process the data"""
    # Example processing: clean data, add features
    df = df.dropna()
    df['processed_at'] = pd.Timestamp.now()
    df['total'] = df.select_dtypes(include=[np.number]).sum(axis=1)

    return df
'''

# Usage
# lambda_manager = LambdaManager()
# role_arn = lambda_manager.create_lambda_role()
# lambda_manager.create_lambda_function('data-processor', lambda_handler_code, role_arn)

# Invoke function
# payload = {'bucket': 'my-data-bucket', 'key': 'input/data.csv'}
# result = lambda_manager.invoke_lambda_function('data-processor', payload)
```

## 3. Google Cloud Platform (GCP)

### 3.1 GCP Data Science Services

#### Vertex AI for ML Development
```python
from google.cloud import aiplatform
from google.cloud import storage
import pandas as pd

class VertexAIManager:
    """Google Cloud Vertex AI operations"""

    def __init__(self, project_id: str, location: str = 'us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)

    def upload_dataset_to_bigquery(self, dataset_name: str, table_name: str, df: pd.DataFrame):
        """Upload dataset to BigQuery"""

        from google.cloud import bigquery

        client = bigquery.Client()

        # Create dataset if it doesn't exist
        dataset_ref = client.dataset(dataset_name)
        try:
            client.get_dataset(dataset_ref)
        except:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = 'US'
            client.create_dataset(dataset)

        # Upload DataFrame to BigQuery
        table_ref = dataset_ref.table(table_name)
        job = client.load_table_from_dataframe(df, table_ref)

        job.result()  # Wait for job to complete
        print(f"Uploaded {len(df)} rows to {dataset_name}.{table_name}")

        return f"{self.project_id}.{dataset_name}.{table_name}"

    def train_automl_model(self, dataset_bq_uri: str, target_column: str,
                          model_name: str = 'automl_model'):
        """Train an AutoML model"""

        # Create dataset
        dataset = aiplatform.TabularDataset.create(
            display_name=f"{model_name}_dataset",
            bq_source=dataset_bq_uri
        )

        # Train model
        job = aiplatform.AutoMLTabularTrainingJob(
            display_name=f"{model_name}_training",
            optimization_prediction_type="regression" if target_column == 'numeric' else "classification"
        )

        model = job.run(
            dataset=dataset,
            target_column=target_column,
            training_fraction_split=0.8,
            validation_fraction_split=0.1,
            test_fraction_split=0.1,
            budget_milli_node_hours=1000,
            model_display_name=model_name
        )

        print(f"Trained AutoML model: {model.resource_name}")
        return model

    def deploy_model(self, model, endpoint_name: str = 'model_endpoint'):
        """Deploy model to endpoint"""

        endpoint = model.deploy(
            deployed_model_display_name=endpoint_name,
            traffic_split={"0": 100},
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=3
        )

        print(f"Deployed model to endpoint: {endpoint.resource_name}")
        return endpoint

    def predict_with_endpoint(self, endpoint, instances: list):
        """Make predictions using deployed endpoint"""

        predictions = endpoint.predict(instances=instances)
        return predictions

# Usage
# vertex_manager = VertexAIManager('my-gcp-project')
# bq_uri = vertex_manager.upload_dataset_to_bigquery('ml_datasets', 'customer_data', df)
# model = vertex_manager.train_automl_model(bq_uri, 'churn_probability')
# endpoint = vertex_manager.deploy_model(model)
```

#### Cloud Storage and Dataflow
```python
from google.cloud import storage
from google.cloud import dataflow

class GCPStorageManager:
    """Google Cloud Storage operations"""

    def __init__(self, project_id: str):
        self.storage_client = storage.Client(project=project_id)
        self.project_id = project_id

    def create_bucket(self, bucket_name: str, location: str = 'US'):
        """Create a GCS bucket"""

        try:
            bucket = self.storage_client.create_bucket(bucket_name, location=location)

            # Enable versioning
            bucket.versioning_enabled = True
            bucket.patch()

            print(f"Created GCS bucket: {bucket_name}")
            return bucket

        except Exception as e:
            print(f"Error creating bucket: {e}")
            return None

    def upload_file(self, bucket_name: str, source_file: str, destination_blob: str):
        """Upload file to GCS"""

        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob)

            blob.upload_from_filename(source_file)

            # Make public (optional)
            # blob.make_public()

            print(f"Uploaded {source_file} to gs://{bucket_name}/{destination_blob}")
            return f"gs://{bucket_name}/{destination_blob}"

        except Exception as e:
            print(f"Error uploading file: {e}")
            return None

    def download_file(self, bucket_name: str, source_blob: str, destination_file: str):
        """Download file from GCS"""

        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(source_blob)

            blob.download_to_filename(destination_file)

            print(f"Downloaded gs://{bucket_name}/{source_blob} to {destination_file}")
            return True

        except Exception as e:
            print(f"Error downloading file: {e}")
            return False

# Usage
# gcs_manager = GCPStorageManager('my-gcp-project')
# gcs_manager.create_bucket('my-data-bucket')
# gcs_manager.upload_file('my-data-bucket', 'local_file.csv', 'data/file.csv')
```

## 4. Microsoft Azure

### 4.1 Azure Machine Learning Services

#### Azure ML SDK for Model Training
```python
from azureml.core import Workspace, Experiment, Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.train.automl import AutoMLConfig
from azureml.core.dataset import Dataset
import pandas as pd

class AzureMLManager:
    """Azure Machine Learning operations"""

    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str):
        self.workspace = Workspace.get(
            name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group
        )

    def create_compute_cluster(self, cluster_name: str = 'cpu-cluster',
                             vm_size: str = 'STANDARD_DS3_V2', max_nodes: int = 4):
        """Create Azure ML compute cluster"""

        try:
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=vm_size,
                max_nodes=max_nodes
            )

            compute_target = ComputeTarget.create(
                self.workspace,
                cluster_name,
                compute_config
            )

            compute_target.wait_for_completion(show_output=True)
            print(f"Created compute cluster: {cluster_name}")

            return compute_target

        except Exception as e:
            print(f"Error creating compute cluster: {e}")
            return None

    def upload_dataset(self, data_path: str, dataset_name: str):
        """Upload dataset to Azure ML"""

        try:
            datastore = self.workspace.get_default_datastore()

            dataset = Dataset.Tabular.from_delimited_files(
                path=data_path,
                datastore=datastore
            )

            dataset = dataset.register(
                workspace=self.workspace,
                name=dataset_name,
                description='Dataset for ML training'
            )

            print(f"Registered dataset: {dataset_name}")
            return dataset

        except Exception as e:
            print(f"Error uploading dataset: {e}")
            return None

    def run_automl_experiment(self, dataset, target_column: str,
                            experiment_name: str = 'automl_experiment'):
        """Run AutoML experiment"""

        # Configure AutoML
        automl_config = AutoMLConfig(
            task='classification',  # or 'regression'
            training_data=dataset,
            label_column_name=target_column,
            primary_metric='accuracy',
            experiment_timeout_minutes=30,
            max_concurrent_iterations=4,
            n_cross_validations=5
        )

        # Create experiment
        experiment = Experiment(self.workspace, experiment_name)

        # Run experiment
        run = experiment.submit(automl_config)
        run.wait_for_completion(show_output=True)

        # Get best model
        best_run, best_model = run.get_output()

        print(f"Best model accuracy: {best_run.metrics['accuracy']}")
        return best_model, best_run

    def deploy_model(self, model, model_name: str = 'ml_model'):
        """Deploy model as web service"""

        from azureml.core.model import Model
        from azureml.core.webservice import AciWebservice, Webservice

        # Register model
        registered_model = Model.register(
            workspace=self.workspace,
            model_path=model.model_path,
            model_name=model_name
        )

        # Create inference config
        from azureml.core.model import InferenceConfig
        from azureml.core.environment import Environment

        env = Environment.get(self.workspace, "AzureML-sklearn-0.24-ubuntu18.04-py37-cpu")
        inference_config = InferenceConfig(entry_script="score.py", environment=env)

        # Deploy to ACI
        deployment_config = AciWebservice.deploy_configuration(
            cpu_cores=1,
            memory_gb=1,
            auth_enabled=True
        )

        service = Model.deploy(
            self.workspace,
            model_name + '_service',
            [registered_model],
            inference_config,
            deployment_config
        )

        service.wait_for_deployment(show_output=True)
        print(f"Deployed model as web service: {service.scoring_uri}")

        return service

# Usage
# azure_ml = AzureMLManager('sub-id', 'resource-group', 'workspace-name')
# compute = azure_ml.create_compute_cluster()
# dataset = azure_ml.upload_dataset('data/train.csv', 'customer_data')
# model, run = azure_ml.run_automl_experiment(dataset, 'target_column')
# service = azure_ml.deploy_model(model)
```

## 5. Cloud Cost Optimization

### 5.1 Cost Monitoring and Optimization

#### AWS Cost Optimization
```python
import boto3
from datetime import datetime, timedelta

class AWSCostOptimizer:
    """AWS cost monitoring and optimization"""

    def __init__(self, region: str = 'us-east-1'):
        self.ce_client = boto3.client('ce', region_name=region)
        self.ec2_client = boto3.client('ec2', region_name=region)

    def get_cost_and_usage(self, days: int = 30):
        """Get AWS cost and usage data"""

        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)

        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.isoformat(),
                    'End': end_date.isoformat()
                },
                Granularity='DAILY',
                Metrics=['BlendedCost'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                    {'Type': 'DIMENSION', 'Key': 'AZ'}
                ]
            )

            total_cost = 0
            service_costs = {}

            for result in response['ResultsByTime']:
                for group in result['Groups']:
                    service = group['Keys'][0]
                    cost = float(group['Metrics']['BlendedCost']['Amount'])
                    total_cost += cost

                    if service not in service_costs:
                        service_costs[service] = 0
                    service_costs[service] += cost

            print(f"Total AWS cost for last {days} days: ${total_cost:.2f}")

            # Top 5 services by cost
            top_services = sorted(service_costs.items(), key=lambda x: x[1], reverse=True)[:5]
            print("\nTop 5 services by cost:")
            for service, cost in top_services:
                print(f"  {service}: ${cost:.2f}")

            return {
                'total_cost': total_cost,
                'service_costs': service_costs,
                'top_services': top_services
            }

        except Exception as e:
            print(f"Error getting cost data: {e}")
            return None

    def identify_unused_resources(self):
        """Identify unused AWS resources"""

        unused_resources = {}

        try:
            # Find stopped EC2 instances
            response = self.ec2_client.describe_instances(
                Filters=[
                    {'Name': 'instance-state-name', 'Values': ['stopped']}
                ]
            )

            stopped_instances = []
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    stopped_instances.append({
                        'instance_id': instance['InstanceId'],
                        'instance_type': instance['InstanceType'],
                        'launch_time': instance['LaunchTime']
                    })

            unused_resources['stopped_ec2_instances'] = stopped_instances

            print(f"Found {len(stopped_instances)} stopped EC2 instances")

            # Check for unattached EBS volumes
            ebs_response = self.ec2_client.describe_volumes(
                Filters=[
                    {'Name': 'status', 'Values': ['available']}
                ]
            )

            unattached_volumes = []
            for volume in ebs_response['Volumes']:
                unattached_volumes.append({
                    'volume_id': volume['VolumeId'],
                    'size_gb': volume['Size'],
                    'volume_type': volume['VolumeType']
                })

            unused_resources['unattached_ebs_volumes'] = unattached_volumes

            print(f"Found {len(unattached_volumes)} unattached EBS volumes")

            return unused_resources

        except Exception as e:
            print(f"Error identifying unused resources: {e}")
            return None

    def recommend_savings(self, cost_data: dict):
        """Provide cost optimization recommendations"""

        recommendations = []

        # Check for high-cost services
        if cost_data and 'service_costs' in cost_data:
            total_cost = cost_data['total_cost']

            for service, cost in cost_data['service_costs'].items():
                percentage = (cost / total_cost) * 100

                if percentage > 20:  # Service costs more than 20% of total
                    recommendations.append({
                        'type': 'high_cost_service',
                        'service': service,
                        'cost': cost,
                        'percentage': percentage,
                        'recommendation': f"Review usage of {service} - it accounts for {percentage:.1f}% of costs"
                    })

        # Instance rightsizing recommendations
        recommendations.append({
            'type': 'rightsizing',
            'recommendation': 'Consider using AWS Compute Optimizer to identify over-provisioned instances'
        })

        # Reserved instances
        recommendations.append({
            'type': 'reserved_instances',
            'recommendation': 'Evaluate purchasing Reserved Instances for steady-state workloads'
        })

        # Storage optimization
        recommendations.append({
            'type': 'storage_optimization',
            'recommendation': 'Use S3 Intelligent Tiering and Glacier for cost-effective storage'
        })

        return recommendations

# Usage
# cost_optimizer = AWSCostOptimizer()
# cost_data = cost_optimizer.get_cost_and_usage(days=30)
# unused_resources = cost_optimizer.identify_unused_resources()
# recommendations = cost_optimizer.recommend_savings(cost_data)
```

### 5.2 Auto-scaling and Spot Instances

#### AWS Auto Scaling Configuration
```python
class AWSAutoScalingManager:
    """AWS Auto Scaling for cost optimization"""

    def __init__(self, region: str = 'us-east-1'):
        self.autoscaling = boto3.client('autoscaling', region_name=region)
        self.ec2 = boto3.client('ec2', region_name=region)

    def create_auto_scaling_group(self, asg_name: str, launch_template_id: str,
                                min_size: int = 1, max_size: int = 10,
                                desired_capacity: int = 2):
        """Create Auto Scaling Group"""

        try:
            response = self.autoscaling.create_auto_scaling_group(
                AutoScalingGroupName=asg_name,
                LaunchTemplate={
                    'LaunchTemplateId': launch_template_id,
                    'Version': '$Latest'
                },
                MinSize=min_size,
                MaxSize=max_size,
                DesiredCapacity=desired_capacity,
                AvailabilityZones=['us-east-1a', 'us-east-1b', 'us-east-1c'],
                HealthCheckType='EC2',
                HealthCheckGracePeriod=300
            )

            print(f"Created Auto Scaling Group: {asg_name}")
            return asg_name

        except Exception as e:
            print(f"Error creating ASG: {e}")
            return None

    def configure_scaling_policies(self, asg_name: str):
        """Configure scaling policies"""

        # Scale out policy (increase capacity)
        self.autoscaling.put_scaling_policy(
            AutoScalingGroupName=asg_name,
            PolicyName='scale-out',
            PolicyType='TargetTrackingScaling',
            TargetTrackingConfiguration={
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'ASGAverageCPUUtilization'
                },
                'TargetValue': 70.0
            }
        )

        # Scale in policy (decrease capacity)
        self.autoscaling.put_scaling_policy(
            AutoScalingGroupName=asg_name,
            PolicyName='scale-in',
            PolicyType='TargetTrackingScaling',
            TargetTrackingConfiguration={
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'ASGAverageCPUUtilization'
                },
                'TargetValue': 30.0
            }
        )

        print(f"Configured scaling policies for {asg_name}")

    def use_spot_instances(self, launch_template_id: str):
        """Configure launch template for spot instances"""

        try:
            # Get current launch template
            lt_response = self.ec2.describe_launch_templates(
                LaunchTemplateIds=[launch_template_id]
            )

            current_data = lt_response['LaunchTemplates'][0]['LaunchTemplateData']

            # Update with spot options
            self.ec2.modify_launch_template(
                LaunchTemplateId=launch_template_id,
                LaunchTemplateData={
                    **current_data,
                    'InstanceMarketOptions': {
                        'MarketType': 'spot',
                        'SpotOptions': {
                            'MaxPrice': '0.10',  # Max spot price ($0.10/hour)
                            'SpotInstanceType': 'one-time',
                            'InstanceInterruptionBehavior': 'terminate'
                        }
                    }
                }
            )

            print(f"Configured spot instances for launch template: {launch_template_id}")

        except Exception as e:
            print(f"Error configuring spot instances: {e}")

# Usage
# scaling_manager = AWSAutoScalingManager()
# asg_name = scaling_manager.create_auto_scaling_group('data-science-asg', 'lt-12345')
# scaling_manager.configure_scaling_policies(asg_name)
# scaling_manager.use_spot_instances('lt-12345')
```

## 6. Security and Compliance in the Cloud

### 6.1 Cloud Security Best Practices

#### Identity and Access Management (IAM)
```python
class CloudSecurityManager:
    """Cloud security and compliance management"""

    def __init__(self, cloud_provider: str = 'aws'):
        self.cloud_provider = cloud_provider
        if cloud_provider == 'aws':
            self.iam = boto3.client('iam')
        elif cloud_provider == 'gcp':
            from google.cloud import iam
            self.iam = iam.Client()
        elif cloud_provider == 'azure':
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.authorization import AuthorizationManagementClient
            credential = DefaultAzureCredential()
            self.iam = AuthorizationManagementClient(credential, 'subscription-id')

    def create_least_privilege_role(self, role_name: str, permissions: list):
        """Create IAM role with least privilege principle"""

        if self.cloud_provider == 'aws':
            # AWS IAM policy
            policy_document = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": permissions,
                        "Resource": "*"
                    }
                ]
            }

            try:
                self.iam.create_role(
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps({
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {"Service": "ec2.amazonaws.com"},
                                "Action": "sts:AssumeRole"
                            }
                        ]
                    }),
                    Path='/data-science/'
                )

                # Attach custom policy
                self.iam.put_role_policy(
                    RoleName=role_name,
                    PolicyName=f'{role_name}_policy',
                    PolicyDocument=json.dumps(policy_document)
                )

                print(f"Created AWS IAM role: {role_name}")
                return role_name

            except Exception as e:
                print(f"Error creating IAM role: {e}")
                return None

    def enable_encryption(self, resource_type: str, resource_id: str):
        """Enable encryption for cloud resources"""

        if self.cloud_provider == 'aws':
            if resource_type == 's3':
                # Enable S3 bucket encryption
                s3 = boto3.client('s3')
                s3.put_bucket_encryption(
                    Bucket=resource_id,
                    ServerSideEncryptionConfiguration={
                        'Rules': [
                            {
                                'ApplyServerSideEncryptionByDefault': {
                                    'SSEAlgorithm': 'AES256'
                                }
                            }
                        ]
                    }
                )
                print(f"Enabled encryption for S3 bucket: {resource_id}")

            elif resource_type == 'rds':
                # Enable RDS encryption
                rds = boto3.client('rds')
                rds.modify_db_instance(
                    DBInstanceIdentifier=resource_id,
                    StorageEncrypted=True,
                    ApplyImmediately=True
                )
                print(f"Enabled encryption for RDS instance: {resource_id}")

    def setup_monitoring_and_alerting(self):
        """Setup monitoring and alerting for security events"""

        if self.cloud_provider == 'aws':
            # Create CloudWatch alarms for security events
            cloudwatch = boto3.client('cloudwatch')

            # Alarm for unauthorized API calls
            cloudwatch.put_metric_alarm(
                AlarmName='UnauthorizedAPICalls',
                AlarmDescription='Alert on unauthorized API calls',
                MetricName='UnauthorizedAttemptCount',
                Namespace='AWS/Security',
                Statistic='Sum',
                ComparisonOperator='GreaterThanThreshold',
                Threshold=0,
                Period=300,
                EvaluationPeriods=1
            )

            print("Setup CloudWatch alarms for security monitoring")

    def implement_data_governance(self, dataset_name: str, classification: str):
        """Implement data governance and classification"""

        governance_policies = {
            'public': {
                'encryption': 'standard',
                'access_control': 'open',
                'retention_days': 365
            },
            'internal': {
                'encryption': 'enhanced',
                'access_control': 'role_based',
                'retention_days': 2555  # 7 years
            },
            'confidential': {
                'encryption': 'maximum',
                'access_control': 'strict',
                'retention_days': 2555
            },
            'restricted': {
                'encryption': 'military_grade',
                'access_control': 'need_to_know',
                'retention_days': 3650  # 10 years
            }
        }

        if classification in governance_policies:
            policy = governance_policies[classification]

            print(f"Applied {classification} governance policy to {dataset_name}:")
            print(f"  Encryption: {policy['encryption']}")
            print(f"  Access Control: {policy['access_control']}")
            print(f"  Retention: {policy['retention_days']} days")

            return policy
        else:
            print(f"Unknown classification: {classification}")
            return None

# Usage
# security_manager = CloudSecurityManager('aws')
# role_name = security_manager.create_least_privilege_role('data-scientist-role',
#     ['s3:GetObject', 's3:PutObject', 'sagemaker:CreateTrainingJob'])
# security_manager.enable_encryption('s3', 'my-data-bucket')
# security_manager.setup_monitoring_and_alerting()
# policy = security_manager.implement_data_governance('customer_data', 'confidential')
```

## 7. Choosing the Right Cloud Platform

### 7.1 Platform Comparison

#### Decision Framework
```python
def compare_cloud_platforms(requirements: dict):
    """Compare cloud platforms based on requirements"""

    platforms = {
        'aws': {
            'strengths': ['Most services', 'Global infrastructure', 'Enterprise focus'],
            'weaknesses': ['Complex pricing', 'Steep learning curve'],
            'best_for': ['Large enterprises', 'Complex architectures', 'Global scale'],
            'data_science_services': ['SageMaker', 'EMR', 'Kinesis', 'Redshift']
        },
        'gcp': {
            'strengths': ['ML/AI focus', 'Big data analytics', 'Simple pricing'],
            'weaknesses': ['Smaller ecosystem', 'Regional coverage'],
            'best_for': ['Data science teams', 'ML-focused projects', 'Startups'],
            'data_science_services': ['Vertex AI', 'BigQuery ML', 'Dataflow', 'Dataproc']
        },
        'azure': {
            'strengths': ['Enterprise integration', '.NET ecosystem', 'Hybrid cloud'],
            'weaknesses': ['Complex licensing', 'Regional coverage'],
            'best_for': ['Microsoft shops', 'Enterprise IT', 'Hybrid deployments'],
            'data_science_services': ['Azure ML', 'Synapse Analytics', 'Databricks', 'HDInsight']
        }
    }

    scores = {}

    for platform, details in platforms.items():
        score = 0

        # Score based on requirements
        if requirements.get('ml_focus', False) and 'ML/AI focus' in details['strengths']:
            score += 3
        if requirements.get('enterprise', False) and 'Enterprise' in details.get('best_for', []):
            score += 3
        if requirements.get('global_scale', False) and 'Global' in details.get('strengths', []):
            score += 2
        if requirements.get('hybrid_cloud', False) and 'Hybrid' in details.get('strengths', []):
            score += 2
        if requirements.get('simple_pricing', False) and 'Simple pricing' in details.get('strengths', []):
            score += 1

        scores[platform] = score

    # Sort by score
    ranked_platforms = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print("Cloud Platform Recommendation:")
    print("=" * 40)

    for platform, score in ranked_platforms:
        details = platforms[platform]
        print(f"\n{platform.upper()} (Score: {score})")
        print(f"Best for: {', '.join(details['best_for'])}")
        print(f"Key services: {', '.join(details['data_science_services'])}")
        print(f"Strengths: {', '.join(details['strengths'])}")

    return ranked_platforms[0][0]  # Return top recommendation

# Usage
requirements = {
    'ml_focus': True,
    'enterprise': False,
    'global_scale': True,
    'hybrid_cloud': False,
    'simple_pricing': True
}

recommended_platform = compare_cloud_platforms(requirements)
print(f"\nRecommended platform: {recommended_platform.upper()}")
```

## 8. Resources and Further Reading

### Books
- "Cloud Computing for Data Analysis" by Dan McCreary and Ann Kelly
- "AWS Certified Machine Learning Specialty" preparation guides
- "Google Cloud AI Platform" documentation

### Online Courses
- Coursera: AWS Machine Learning Engineer
- Google Cloud: Machine Learning with TensorFlow on Google Cloud
- Microsoft Learn: Azure AI Fundamentals

### Certifications
- AWS Certified Machine Learning - Specialty
- Google Cloud Professional Machine Learning Engineer
- Azure AI Engineer Associate

### Tools and Frameworks
- **AWS**: SageMaker, EMR, Kinesis, Lambda
- **Google Cloud**: Vertex AI, BigQuery, Dataflow, Cloud Functions
- **Azure**: Azure ML, Synapse Analytics, Databricks, Functions

## Next Steps

Congratulations on mastering cloud computing for data science! You now understand cloud platforms, deployment strategies, and cost optimization. In the next module, we'll explore ethics and best practices in data science.

**Ready to continue?** Proceed to [Module 12: Ethics and Best Practices](../12_ethics_best_practices/)
