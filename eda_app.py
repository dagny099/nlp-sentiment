import streamlit as st
from st_keyup import st_keyup
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import gzip
import json
from helperFunc import *

# -----------------------------------------------------
# Custom variables - later move to params.py
# n_samples = 50
# PRODUCT_CATEGORY = 'Movies_and_TV'

# -----------------------------------------------------
# Setup session
st.set_page_config(
    page_title="Explore Sentiment NLP",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.google.com/document/d/1UYg9J-Sd0-ojnYoqFasAB-OjrpsnGcJvfgkgpJsSJco/edit?usp=sharing',
        'Report a bug': "http://www.barbhs.com",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

if 'openai_key'not in st.session_state:
    st.session_state.openai_key = st.secrets["openai_key"]

# conn = st.experimental_connection('s3', type=FilesConnection)

@st.cache_data
def load_metadata(PRODUCT_CATEGORY):
    # keepUs, data_list, metaSm  = ["title","asin","description","price","category","main_cat"], [] , []
    # with gzip.open(f'data/{PRODUCT_CATEGORY}/meta_{PRODUCT_CATEGORY}.json.gz', 'r') as f:
    #     for line in f:
    #         json_obj = json.loads(line)
    #         data_list.append({key: json_obj[key] for key in keepUs if key in json_obj})  ## ***
    # # df.drop(['description','category'],axis=1).drop_duplicates()
    # return pd.DataFrame(data_list)
    return pd.read_parquet(f'data/{PRODUCT_CATEGORY}/metadata_for_reviews.parquet')


@st.cache_data
def load_reviews(PRODUCT_CATEGORY):
    data_list, keys, ctr = [], ['asin', 'overall', 'reviewText'], 0
    with gzip.open(f'data/{PRODUCT_CATEGORY}/{PRODUCT_CATEGORY}_5.json.gz', 'r') as f:
        for line in f:
            # if ctr<LIMIT:
                json_obj = json.loads(line)
                data_list.append({key: json_obj[key] for key in keys if key in json_obj})
                ctr+=1

    return pd.DataFrame(data_list)

def plot_ratings_hist(tmpDf):
    # Plotting the histogram
    plt.figure(figsize=(8, 5))
    tmpDf['overall'].hist(bins=[1, 2, 3, 4, 5, 6], edgecolor='black', align='left')
    # Setting the title and labels
    plt.title('Ratings Histogram')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.xticks([1, 2, 3, 4, 5])
    return plt

# -----------------------------------------------------
st.title('Explore Textual Feelings')
# -----------------------------------------------------

st.sidebar.header('Amazon product reviews by category')
PRODUCT_CATEGORY = st.sidebar.selectbox('Pick a category:', overviewDF['Product Category'], index= 4).replace(' ','_')

idx = [i for i, n in enumerate(overviewDF['Product Category']) if n == PRODUCT_CATEGORY.replace('_',' ')]
idx = idx[0]

if 'title_filter_sec1' not in st.session_state:
     user_input = overviewDF['Example_Product_Title'].iloc[idx]
else:
     user_input = st.session_state.title_filter_sec1

col1,  col2 = st.columns([1, 1])

my_slot1, my_slot2, my_slot3 = col1.empty(), col1.empty(), col1.empty()
my_slot4, my_slot5, my_slot6 = col2.empty(), col2.empty(), col2.empty()
                               
# LOAD DATA
my_slot1.text(f'Loading data from {PRODUCT_CATEGORY}...')   
df = load_metadata(PRODUCT_CATEGORY)   # df = create_sample_df(n_samples) 
df = df[df['num_reviews']>0].sort_values(by='num_reviews',ascending=False)

my_slot1.subheader(f"{PRODUCT_CATEGORY.replace('_',' ')} Product Reviews")

with my_slot2:    
    user_input = st_keyup("Enter a title below to limit results (or leave blank to show all)", value=user_input, key="title_filter_sec1")
    filtered_df = df[df['title'].str.startswith(user_input)]
    my_slot3.dataframe(filtered_df.drop(['asin'],axis=1), hide_index=True)
    #TODO- sort the columms to show the number of reviews in the second column , maybe a boxplot of the reviews?

# st.markdown("""---\n<h4>Reviews</h4>""", unsafe_allow_html=True)
st.subheader("Explore Language Used")
with st.expander(label="", expanded=False):
    st.pyplot(plot_ratings_hist(reviewsDF[reviewsDF['asin']==x]))

with st.expander(label="Toggle List of Reviews", expanded=True):
    revtext = st.empty()
    revtext2 = st.empty()
    revtext.text(f'Loading data from {PRODUCT_CATEGORY}...')   
    reviewsDF = load_reviews(PRODUCT_CATEGORY) 


    #Choose the entry at the top of the dataframe to show reviews:
    x=filtered_df.iloc[0]['asin'] 
    xTitle = df[df['asin']==x].iloc[0]['title']
    nReviews = reviewsDF[reviewsDF['asin']==x].shape[0]
    revtext.write(f'Now showing {nReviews} reviews for: {xTitle}\n')
    revtext2.dataframe(reviewsDF[reviewsDF['asin']==x].drop(['asin'],axis=1), hide_index=True, use_container_width=True)
    #TODO- For more engagement and color-consistency, change the 5pt integers to a  color scale? or stars?     


# RIGHT COLUMN CONTENT
with my_slot6.expander(label="Word Cloud", expanded=False):
    st.write('Placeholder')
    

with my_slot5.expander(label="List of most reviewed products", expanded=False):
    # st.pyplot(plot_ratings_hist(reviewsDF[reviewsDF['asin']==x]))
    st.write('The visualization below is filler for now')
    sampleDF = create_sample_df(100)
    # Seaborn Barplot
    plt.figure(figsize=(8, 6))
    sns.barplot(data=sampleDF, x='x', y='y')
    st.pyplot(plt)


st.empty()

st.markdown("""---""")

sec2_L,  sec2_M, sec2_R = st.columns([1,1,1])

sec2_L.image("mei.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
sec2_M.image("mei.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
sec2_R.image("mei.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")


#     st.write("- Insight 1: This is a sample insight")
#     st.write("- Insight 2: The bar graph (Seaborn) represents column B values")
#     st.write("- Insight 3: Add more insights based on your data")

# # Third subheading with collapsible content
# with st.expander("Subheading 3: Scatter Plot & Filter"):
#     x_col = st.selectbox('Select x-axis column:', df.columns)
#     y_col = st.selectbox('Select y-axis column:', df.columns)

#     # Plotly Scatter Plot
#     fig = px.scatter(df, x=x_col, y=y_col)
#     st.plotly_chart(fig)

