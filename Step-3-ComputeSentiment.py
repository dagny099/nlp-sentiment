#run this script at the command line after the review test is pre-processed
import sys
from helperFunc import *

def main():

    if len(sys.argv)-1<1:
        print(f'Call this script with a product category name, e.g. python Step-3-ComputeSentiment.py Digital_Music)')
    else:
        PRODUCT_CATEGORY = sys.argv[1]
        path = f'data/{PRODUCT_CATEGORY}'

    print(f'\n\n---STEP 3) Compute Sentiment\n---\nThis script loads preprocessed reviews and runs two algorithms to compute sentiment per review\n\n')
    # Load dataframe with preprocessed reviews:
    try:
        df = pd.read_parquet(path+'/processed_reviews.parquet')
        print(f'Loaded {path}/processed reviews ')
    except:
        print(f'File not found: ')

    # -----------------------
    # Compute TextBlob scores
    if 1:
        print('Computing textblob scores')
        df['sentiment_v1'] = df['reviewText_processed'].apply(get_sentiment_1)

        # Store scores
        print('Saving locally...')
        df[['asin','overall','sentiment_v1']].to_parquet(f'{path}/sentiment_scores_1.parquet', engine = 'pyarrow', compression = 'gzip')
        print('Saved textblob scores as local parquet file')

        #upload_to_s3(f'{PRODUCT_CATEGORY}_sentiment_scores_1.parquet', "run-explorer-files")
        #print('Uploaded parquet to S3')

    # -----------------------
    if 1:
        # Compute Vader scores
        print('Computing VADER scores')
        df['sentiment_v2'] = df['reviewText_processed'].apply(get_sentiment_2)

        # ETL ... convert the list column -> 4 numeric columns
        print('ETL of the list column')
        df[['neu', 'neg', 'pos','compound']] = df['sentiment_v2'].apply(pd.Series)

        # Store scores
        print('Saving locally...')
        df[['asin','overall','neu','neg','pos','compound']].to_parquet(f'{path}/sentiment_scores_2.parquet', engine = 'pyarrow', compression = 'gzip')
        print('Saved VADER scores as local parquet file')

        #upload_to_s3(f'{PRODUCT_CATEGORY}_sentiment_scores_2.parquet', "run-explorer-files")
        #print('Uploaded parquet to S3')



if __name__=="__main__":
    main()

