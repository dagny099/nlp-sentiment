import sys
from helperFunc import *

def return_polarity(senti_score):
    if senti_score < -0.05:
        return -1  #Negative sentiment
    elif senti_score > 0.05:
        return 1   #Positive sentiment
    else:
        return 0


def main():
          
    if len(sys.argv)-1<1:
        print(f'Call this script with a product category name, e.g. python Step-4-CombineScores.py Digital_Music)')
    else:
        PRODUCT_CATEGORY = sys.argv[1]
        path = f'data/{PRODUCT_CATEGORY}'

    print(f'\n\n---\nSTEP 4) Combine Scores\n---\nThis script loads the saves sentiment scores and assigns a polarity, -1 or 0 or 1, and saves a single dataframe with those features.\n\n')

    METADATA_FILE = f'data/{PRODUCT_CATEGORY}/metadata_for_reviews.parquet'
    REVIEW_FILE = f'data/{PRODUCT_CATEGORY}/processed_reviews.parquet'
    SENTI_1 = f'data/{PRODUCT_CATEGORY}/sentiment_scores_1.parquet'
    SENTI_2 = f'data/{PRODUCT_CATEGORY}/sentiment_scores_2.parquet'

    # LOAD REVIEWS FILE
    print('Loading metadata, reviews, and sentiment scores...')
    metaDF = pd.read_parquet(METADATA_FILE)
    revDF = pd.read_parquet(REVIEW_FILE)
    s1DF = pd.read_parquet(SENTI_1)
    s2DF = pd.read_parquet(SENTI_2)

    # Assign a polarity based on TextBlob scores
    print('Assigning polarity based on sentiment scores...')
    s1DF['polarity_v1'] = s1DF['sentiment_v1'].apply(return_polarity)

    # Assign a polarity based on the most likely sentiment based on VADER sub-scores
    s2DF['polarity_v2'] = s2DF[['neu', 'neg', 'pos']].idxmax(axis=1)
    replace_map = {    'neu': 0,    'neg': -1,    'pos': 1 } # Replace the categories with numbers
    s2DF['polarity_v2'] = s2DF['polarity_v2'].replace(replace_map)

    # Assign a polarity based on the VADER Compound sub-score
    s2DF['polarity_v2_c'] = s2DF['compound'].apply(return_polarity)

    # Join all info into one dataframe:
    print('Joining all info together')
    TMP = pd.concat([s1DF, s2DF.drop(columns=['asin', 'overall'])], axis=1)
    TMP = pd.concat([TMP, revDF.drop(columns=['asin', 'overall'])], axis=1)
    productDF = TMP.merge(metaDF.drop(['description','price','category'], axis=1), on='asin', how='inner')

    # Store scores
    print('Saving locally...')
    productDF.to_parquet(f'{path}/SCORED_REVIEWS.parquet', engine = 'pyarrow', compression = 'gzip')
    print('Saved "productDF" as local parquet file')

    #upload_to_s3(f'{PRODUCT_CATEGORY}_SCORED_REVIEWS.parquet', bucketName)
    #print('Uploaded parquet to S3')


if __name__=="__main__":
    main()
