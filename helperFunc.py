import numpy as np
import pandas as pd
from scipy.stats import norm
import json
import gzip
import boto3
import toml

creds = toml.load(".streamlit/secrets.toml")
bucketName = "product-data-dagny099"

# -----------------------------------------------------
# Load the gzip dataset downloaded
def load_product_data(file, keepUs):
    data_list  = [] 
    print(f'\nLoading file {file}')
    # List to store dictionaries with only desired keys
    with gzip.open(file, 'r') as f:
        for line in f:
            json_obj = json.loads(line)
            data_list.append({key: json_obj[key] for key in keepUs if key in json_obj})  ## ***
    print('Done')
    return pd.DataFrame(data_list)



# -----------------------------------------------------
# Create continuous data
def create_sample_df(n_samples, real_correlation=0.5):
    '''Make some sample data in a dataframe for plotting & code practice '''
    x = np.random.rand(n_samples)
    y = real_correlation * x + 0.1 * np.random.randn(n_samples)  # Correlation with some noise 

    # Create categorical data with three categories
    categories = ['A', 'B', 'C']
    category = np.random.choice(categories, size=n_samples, p=[0.4, 0.3, 0.3])

    # Introduce interaction with the categorical variable
    for i in range(n_samples):
        if category[i] == 'A':
            y[i] += 0.2
        elif category[i] == 'B':
            y[i] -= 0.2
        # 'C' remains neutral

    # Return dataframe
    return pd.DataFrame({'x': x, 'y': y, 'category': category})

# -----------------------------------------------------
def find_CI(r, n_samples, confidence_level=0.90):
    '''Find the confidence interval for a given confidence level for a given number of data samples'''
    # Set the desired confidence level
    alpha = 1 - confidence_level
    z_value = norm.ppf(1 - alpha/2)  # Get the Z value for the desired confidence level

    # Use Fisher's Z=transformation to find the 95% confidence interval
    z = 0.5 * np.log((1 + r) / (1 - r))
    std_error_z = 1 / np.sqrt(n_samples - 3)
    z_low = z - z_value * std_error_z
    z_high = z + z_value * std_error_z

    r_low = (np.exp(2 * z_low) - 1) / (np.exp(2 * z_low) + 1)
    r_high = (np.exp(2 * z_high) - 1) / (np.exp(2 * z_high) + 1)
    return r_low, r_high


# -----------------------------------------------------
from textblob import TextBlob
def get_sentiment_1(text):
    '''Compute sentiment based on '''
    return TextBlob(text).sentiment.polarity


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
def get_sentiment_2(text):

    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

# -----------------------------------------------------
def upload_to_s3(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    if object_name is None:
        object_name = file_name

    session = boto3.Session(
        aws_access_key_id=creds["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=creds["AWS_SECRET_ACCESS_KEY"],
    )
    s3 = session.resource('s3')
    try:
        s3.meta.client.upload_file(Filename=file_name, Bucket=bucket, Key=object_name)
    except Exception as e:
        print(f"Upload failed with error: {e}")
        return False

# -----------------------------------------------------

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# # Sample preprocessing function
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Convert text string to tokens
    tokens = word_tokenize(text)
    # Lemmatize/stem the tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    # Return a string of tokens
    return ' '.join(tokens)
