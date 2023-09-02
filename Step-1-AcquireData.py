import sys
import os
import pandas as pd
import numpy as np
from helperFunc import *


def main():

    if len(sys.argv)-1<1:
        print(f'Call this script with a product category name, e.g. python Step-1-AcquireData.py Digital_Music)')
    else:
        PRODUCT_CATEGORY = sys.argv[1]
        # PRODUCT_CATEGORY = 'Movies_and_TV'# 'Movies_and_TV' # 'Books' or 'Digital_Music'

    print(f'\n\n---\nSTEP 1) Aquire data\n---\nThis script loads the downloaded zip data and saves a parquet file with only the de-duped metadata rowss\n\n')
    print(f'\nKNOW THIS!!! The "/data" folder should have a folder for {PRODUCT_CATEGORY} with files: \n\t"meta_{PRODUCT_CATEGORY}.json.gz" and \t"{PRODUCT_CATEGORY}_5.json.gz"')

    METADATA_FILE = f'data/{PRODUCT_CATEGORY}/meta_{PRODUCT_CATEGORY}.json.gz'
    REVIEW_FILE = f'data/{PRODUCT_CATEGORY}/{PRODUCT_CATEGORY}_5.json.gz'
    SAVE_MERGED = f'data/{PRODUCT_CATEGORY}/metadata_for_reviews.parquet'

    # LOAD METADATA FILE 
    # ------------------------------------------------
    file = METADATA_FILE
    metaDF = load_product_data(file, ["title","asin","description","price","category","main_cat"])
    nprod_orig = metaDF.shape[0]
    
    # LOAD REVIEWS FILE
    # ------------------------------------------------
    file = REVIEW_FILE
    reviewsDF = load_product_data(file, ['asin', 'overall', 'reviewText'])


    # PRUNE METADATA & SAVE 
    # ------------------------------------------------
    # Prune duplicates from metadataDF:
    print(f'\2- Dropping duplicate metadata items...')
    metaDF = metaDF.loc[metaDF.drop(['description','category'], axis=1).drop_duplicates().index]

    # Join with df to AGGREGATE number of reviews per item:
    print(f'\n3- Merging reviews-data with metadata ...')
    review_counts = reviewsDF.groupby('asin').size().reset_index(name='num_reviews')
    metaDF = metaDF.merge(review_counts, on='asin', how='left').fillna(0)
    # Drop products without reviews and sort by num_reviews ????

    print(f'\n4- Saving metadata file with review counts: {SAVE_MERGED}')
    metaDF.to_parquet(SAVE_MERGED, engine = 'pyarrow', compression = 'gzip')

    print(f'\n5- Uploading to S3')
    object_name = PRODUCT_CATEGORY + '_' + SAVE_MERGED.split('/')[-1]
    upload_to_s3(SAVE_MERGED, bucketName, object_name)
    print(f'Done')
    print(metaDF.shape)
    
    
    # DISPLAY STATISTICS
    # ------------------------------------------------
    nItems = metaDF.drop(['description','category'],axis=1).drop_duplicates().shape[0]
    print(f'\n------\nNumber of {PRODUCT_CATEGORY} products with metadata: {nprod_orig}')
    print(f'Number of products after duplicates are removed: {nItems}')
    
    nprod = len(reviewsDF['asin'].unique())
    print(f'\nNumber of PRODUCTS with reviews: {nprod}')
    print(f'Number of REVIEWS to analyze: {reviewsDF.shape[0]}\n')

    

if __name__=="__main__":
    main()

