"""Train an image classification model."""

import os
import time
import sys
import json

import boto3
import wget


START = time.time()
ROLE = sys.argv[1]
BUCKET = sys.argv[2]
STACK_NAME = sys.argv[3]
COMMIT_ID = sys.argv[4][0:7]
TIMESTAMP = time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())
TRAINING_IMAGE = (
    '811284229777.dkr.ecr.us-east-1.amazonaws.com/image-classification:latest'
)


def download(url):
    """Download URL.

    :param url: URL to download.
    """
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        wget.download(url, filename)


def upload_to_s3(channel, file):
    """Upload file to S3 bucket.

    :param channel: Training channel (directory).
    :param file: File to upload.
    """
    s3_resource = boto3.resource('s3')
    data = open(file, "rb")
    key = channel + '/' + file
    s3_resource.Bucket(BUCKET).put_object(Key=key, Body=data)


def write_config_file(config_dict: dict, env: str):
    """Write config info to JSON file.

    :param config_dict: Dictionary of config values to write.
    :param env: Environment name for which to write config values.
    """
    file_path = f'./CloudFormation/configuration_{env}.json'
    config_dict['Parameters']['Environment'] = env
    with open(file_path, 'w') as config_fh:
        config_fh.write(json.dumps(config_dict))


def collect_data(url: str, channel: str, file_name: str):
    """Collect data to use for training and testing.

    :param url: URL to download.
    :param channel: Training channel (directory).
    :param file: File to upload.
    """
    print(f"Downloadng data : {channel}")
    download(url)
    upload_to_s3(channel, file_name)
    print(f"Finished downloadng data : {channel}")

# caltech-256
collect_data(
    'http://data.mxnet.io/data/caltech-256/caltech-256-60-train.rec',
    'train',
    'caltech-256-60-train.rec'
)
collect_data(
    'http://data.mxnet.io/data/caltech-256/caltech-256-60-val.rec',
    'validation',
    'caltech-256-60-val.rec'
)

print("Setting Algorithm Settings")

# The algorithm supports multiple network
# depth (number of layers).
# They are 18, 34, 50, 101, 152 and 200
# For this training, we will use 18 layers
NUM_LAYERS = "18"

# we need to specify the input image shape
# for the training data
IMAGE_SHAPE = "3,224,224"

# we also need to specify the number of
# training samples in the training set
# for caltech it is 15420
NUM_TRAINING_SAMPLES = "15420"

# specify the number of output classes
NUM_CLASSES = "257"

# batch size for training
MINI_BATCH_SIZE = "64"

# number of epochs
EPOCHS = "15"

# learning rate
LEARNING_RATE = "0.01"

# create unique job name
JOB_NAME = f'{STACK_NAME}-{COMMIT_ID}-{TIMESTAMP}'

TRAINING_PARAMS = {
    # specify the training docker image
    "AlgorithmSpecification": {
        "TrainingImage": TRAINING_IMAGE,
        "TrainingInputMode": "File"
    },
    "RoleArn": ROLE,
    "OutputDataConfig": {"S3OutputPath": f's3://{BUCKET}/'},
    "ResourceConfig": {
        "InstanceCount": 1,
        "InstanceType": "ml.p2.xlarge",
        "VolumeSizeInGB": 50
    },
    "TrainingJobName": JOB_NAME,
    "HyperParameters": {
        "image_shape": IMAGE_SHAPE,
        "num_layers": str(NUM_LAYERS),
        "num_training_samples": str(NUM_TRAINING_SAMPLES),
        "num_classes": str(NUM_CLASSES),
        "mini_batch_size": str(MINI_BATCH_SIZE),
        "epochs": str(EPOCHS),
        "learning_rate": str(LEARNING_RATE)
    },
    "StoppingCondition": {
        # Maximum time to run training for model
        "MaxRuntimeInSeconds": 3600,
        # Maximum time to wait for spot instances to train model
        "MaxWaitTimeInSeconds": 3600
    },
    "EnableManagedSpotTraining": True,

    # Training data should be inside a subdirectory called "train"
    # Validation data should be inside a subdirectory called "validation"
    # The algorithm currently only supports fully replicated model (where data
    # is copied onto each machine)
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f's3://{BUCKET}/train/',
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "application/x-recordio",
            "CompressionType": "None"
        },
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f's3://{BUCKET}/validation/',
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "application/x-recordio",
            "CompressionType": "None"
        }
    ]
}
print(f'Training job name: {JOB_NAME}')
print(
    "\nInput Data Location: " +
    f"{TRAINING_PARAMS['InputDataConfig'][0]['DataSource']['S3DataSource']}"
)

# create the Amazon SageMaker training job
SAGEMAKER_CLIENT = boto3.client(service_name='sagemaker')
SAGEMAKER_CLIENT.create_training_job(**TRAINING_PARAMS)

# confirm that the training job has started
STATUS = SAGEMAKER_CLIENT.describe_training_job(TrainingJobName=JOB_NAME)[
    'TrainingJobStatus'
]
print(f'Training job current status: {STATUS}')

try:
    # wait for the job to finish and report the ending status
    SAGEMAKER_CLIENT.get_waiter('training_job_completed_or_stopped').wait(
        TrainingJobName=JOB_NAME
    )
    TRAINING_INFO = SAGEMAKER_CLIENT.describe_training_job(
        TrainingJobName=JOB_NAME
    )
    STATUS = TRAINING_INFO['TrainingJobStatus']
    print(f'Training job ended with status: {STATUS}')
except SAGEMAKER_CLIENT.exceptions.ResourceNotFound:
    print('Training failed to start')
    # if exception is raised, that means it has failed
    MESSAGE = SAGEMAKER_CLIENT.describe_training_job(TrainingJobName=JOB_NAME)[
        'FailureReason'
    ]
    print(f'Training failed with the following error: {MESSAGE}')


# Creating configuration files so we can pass parameters to our Sagemaker
# Endpoint CloudFormation
CONFIG_DATA = {
    "Parameters": {
        "BucketName": BUCKET,
        "CommitID": COMMIT_ID,
        "Environment": None,
        "ParentStackName": STACK_NAME,
        "SageMakerRole": ROLE,
        "Timestamp": TIMESTAMP
    }
}
write_config_file(CONFIG_DATA, 'qa')
write_config_file(CONFIG_DATA, 'prod')

END = time.time()
print(END - START)
