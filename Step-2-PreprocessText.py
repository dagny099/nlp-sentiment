
import sys
from helperFunc import *

def main():
          
    if len(sys.argv)-1<1:
        print(f'Call this script with a product category name, e.g. python Step-2-PreprocessText.py Digital_Music)')
    else:
        PRODUCT_CATEGORY = sys.argv[1]
        # PRODUCT_CATEGORY = 'Movies_and_TV'# 'Movies_and_TV' # 'Books' or 'Digital_Music'

    print(f'\n\n---\nSTEP 2) Preprocess Text\n---\nThis script processes a dataframe with the column "reviewText" and saves a parquet file with the preprocessed reviews\n\n')

    REVIEW_FILE = f'data/{PRODUCT_CATEGORY}/{PRODUCT_CATEGORY}_5.json.gz'
    SAVE_PROCESSED = f'data/{PRODUCT_CATEGORY}/processed_reviews.parquet'

    # LOAD REVIEWS FILE
    # ------------------------------------------------
    reviewsDF = load_product_data(REVIEW_FILE, ['asin', 'overall', 'reviewText'])

    # Replace any missing reviews with '' instead of NaN
    reviewsDF['reviewText'].fillna('', inplace=True)

    # Preprocess the reviews:
    # (For Movies_and_TV => This took 12min on my mac mini)
    print(f'\n1- Preprocessing text')
    reviewsDF['reviewText_processed'] = reviewsDF['reviewText'].apply(preprocess)

    print(f'\n2- Saving preprocessed reviews')
    reviewsDF.to_parquet(SAVE_PROCESSED, engine = 'pyarrow', compression = 'gzip')

    print(f'\n3- Uploading to S3')
    object_name = PRODUCT_CATEGORY + '_' + SAVE_PROCESSED.split('/')[-1]
    upload_to_s3(SAVE_PROCESSED, bucketName, object_name)
    print(f'Done')
    print(reviewsDF.shape)
    


if __name__=="__main__":
    main()

