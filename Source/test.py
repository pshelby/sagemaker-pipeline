"""Test SageMaker model by calling its endpoint."""

import json
import os
import sys
import time
from random import randint

import boto3
import wget

import numpy as np


START = time.time()
IMG_URL_PREFIX = 'http://www.vision.caltech.edu/Image_Datasets/Caltech256/images'

ENDPOINT_NAME = sys.argv[1]
CONFIGURATION_FILE = sys.argv[2]

with open(CONFIGURATION_FILE) as f:
    DATA = json.load(f)

COMMIT_ID = DATA["Parameters"]["CommitID"]
TIMESTAMP = DATA["Parameters"]["Timestamp"]
ENDPOINT_NAME = f'{ENDPOINT_NAME}-{COMMIT_ID}-{TIMESTAMP}'

with open(f'{os.path.dirname(os.path.realpath(__file__))}/object_categories.json') as label_f:
    OBJECT_CATEGORIES = json.load(label_f)

SAGEMAKER_RUNTIME = boto3.client('runtime.sagemaker')

IMG_CATEGORY_IDS = [str(randint(1, 257)).rjust(3, '0')]
for category_id in IMG_CATEGORY_IDS:
    IMG_START = time.time()

    IMG_CATEGORY_NAME = OBJECT_CATEGORIES[int(category_id) - 1]
    IMG_SUFFIX = str(randint(1, 200)).rjust(4, '0')
    IMG_SHORTNAME = f'{IMG_CATEGORY_NAME}_{IMG_SUFFIX}.jpg'
    IMG_URL = f'{IMG_URL_PREFIX}/{category_id}.{IMG_CATEGORY_NAME}/{category_id}_{IMG_SUFFIX}.jpg'
    print(f'Test Image: {IMG_URL}')

    wget.download(
        IMG_URL,
        IMG_SHORTNAME,
        bar=None
    )
    with open(IMG_SHORTNAME, 'rb') as f:
        PAYLOAD = f.read()
        PAYLOAD = bytearray(PAYLOAD)

    RESPONSE = SAGEMAKER_RUNTIME.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                                 ContentType='application/x-image',
                                                 Body=PAYLOAD)
    RESULT = RESPONSE['Body'].read()

    # result will be in json format and convert it to ndarray
    RESULT = json.loads(RESULT.decode('utf-8'))

    # the result will output the probabilities for all classes
    # find the class with maximum probability and print the class index
    INDEX = np.argmax(RESULT)
    print(f"Expected: label - {IMG_CATEGORY_NAME}")
    print(f"Result: label - {OBJECT_CATEGORIES[INDEX]}, probability - {str(RESULT[INDEX])}")

    IMG_END = time.time()
    IMG_SECONDS = IMG_END - IMG_START
    print(f'Time: {IMG_SECONDS}\n')

END = time.time()
SECONDS = END - START
print(f'Total Time: {SECONDS}s')
