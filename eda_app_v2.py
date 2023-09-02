import streamlit as st
from st_keyup import st_keyup
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import gzip
import json
from helperFunc import *
from wordcloud import WordCloud
import plotly.figure_factory as ff

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
def load_data(PRODUCT_CATEGORY, data_type='meta'):  #'meta' or 'reviews'
    if data_type=='meta':
        TMP = pd.read_parquet(f'data/{PRODUCT_CATEGORY}/metadata_for_reviews.parquet')
        TMP = TMP[['title','num_reviews','description','price','main_cat']].sort_values(by='num_reviews', ascending=False)
        TMP = TMP[TMP['num_reviews']>0]
    else:
        TMP = pd.read_parquet(f'data/{PRODUCT_CATEGORY}/SCORED_REVIEWS.parquet')
    TMP['num_reviews'] = TMP['num_reviews'].astype(int)
    return TMP

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

def resetStuff():
    del st.session_state['title_filter_sec1']

# -----------------------------------------------------
# Initialize state stuff
if 'pick_cat' not in st.session_state:
    PRODUCT_CATEGORY = 'Software'
else:
    PRODUCT_CATEGORY = st.session_state.pick_cat
    # PRODUCT_CATEGORY = PRODUCT_CATEGORY.replace(' ','_')
    # st.write(st.session_state)

PRODUCT_CATEGORY = PRODUCT_CATEGORY.replace(" ","_")
init_category = overviewDF[overviewDF['Product Category']==PRODUCT_CATEGORY.replace("_"," ")].index.to_list()[0]
df = load_data(PRODUCT_CATEGORY,'meta')   # df = create_sample_df(n_samples) 
reviewsDF = load_data(PRODUCT_CATEGORY,'reviews') 

# -----------------------------------------------------
st.title('Explore Textual Feelings')
with st.expander(label="Project Motivation and Assumptions"):
    st.write('This project uses a dataset of product reviews as a "supervised learning" exploration of sentiment in review text')
    st.write('Assumption 1a: If a person felt good about the product, the rating will probably be a 4 or 5 ')
    st.write('Assumption 1b: If a person felt bad about the product, the rating will probably be a 1 or 2')
    st.write('Assumption 1c: Generally, if a person felt neutrally about the item, they might not write much and/or be less likely to use emotional words')

    st.write('This project will investigate those assumptions and, critically, provide transparency about how sentiment models work.')
    st.write("Let's begin by investigating what people write about Amazon from a particular category...")

# --------------------------------------------------------
# SIDEBAR - Select Product Cateogry and Display Statistics
st.sidebar.header('Amazon Product Reviews')
# Pick a category of product from the selectbox
PRODUCT_CATEGORY = st.sidebar.selectbox('Pick a category:', overviewDF['Product Category'], 
                     index= init_category, key="pick_cat", on_change=resetStuff)

# st.sidebar.write(f'Number of reviews to analyze: {reviewsDF.shape[0]}')  # ‣ 

# IDENTIFY CATEGORY FILTER:
colors = overviewDF.shape[0]*['blue']
colors[init_category]='red'

# Bar chart shows metric across categories 
fig = px.bar(overviewDF,x="Num_Products_Reviewed", y="Product Category", color=colors, text_auto='.0f', orientation="h")
fig.update_layout(height=275, title='Number of Products Reviewed', 
                  xaxis_title='', yaxis={"categoryorder":"total ascending"}, 
                  margin=dict(l=20, r=20, t=50, b=0), showlegend=False)
st.sidebar.plotly_chart(fig, theme="streamlit", use_container_width=True)

# Show table with top most-rated items
st.sidebar.markdown(f"""<h5>Top Rated Products in {PRODUCT_CATEGORY}""", unsafe_allow_html=True)
# st.sidebar.dataframe(df.sort_values(by='num_reviews', ascending=False)['title'].head(5))
for i in range(0,3):
    st.sidebar.write(f"‣ {df.sort_values(by='num_reviews', ascending=False).iloc[i]['title']}, {int(df.sort_values(by='num_reviews', ascending=False).iloc[i]['num_reviews'])} reviews")
# DISPLAY STATISTICS
# st.sidebar.markdown(f"""<hr>""", unsafe_allow_html=True)
rating_counts = reviewsDF['overall'].value_counts().reset_index()
fig2 = px.pie(rating_counts, names='overall', values='count', title='Frequency of Ratings in Category')
st.sidebar.plotly_chart(fig2, theme="streamlit", use_container_width=True)
# --------------------------------------------------------

#Initialize user-input for title filter
if 'title_filter_sec1' not in st.session_state:
   #user_input = overviewDF['Example_Product_Title'].iloc[init_category]
    user_input = 'A'
else:
    user_input = st.session_state.title_filter_sec1

col1,  col2 = st.columns([1, 1])

my_slot1, my_slot2, my_slot3 = col1.empty(), col1.empty(), col1.empty()
my_slot4, my_slot5, my_slot6, my_slot7 = col2.empty(), col2.empty(), col2.empty(), col2.empty()
                               
# LOAD DATA
my_slot1.text(f'Loading data from {PRODUCT_CATEGORY}...')   
my_slot1.subheader(f"{PRODUCT_CATEGORY.replace('_',' ')} Product Reviews")
# st.write(user_input)

with my_slot2:    
    user_input = st_keyup("Enter a title below to limit results (or leave blank to show all)", value=user_input, key="title_filter_sec1") #, on_change=resetStuff)
    filtered_df = df[df['title'].str.startswith(user_input)]
    my_slot3.dataframe(filtered_df, hide_index=True)


st.subheader("Explore Language Used")
with st.expander(label="Toggle List of Reviews", expanded=True):
    revtext = st.empty()
    revtext2 = st.empty()
    revtext.text(f'Loading data from {PRODUCT_CATEGORY}...')   


    #Choose the entry at the top of the dataframe to show reviews:
    # x=filtered_df.iloc[0]['asin'] 
    xTitle = filtered_df.iloc[0]['title']
    filtered_df2 = reviewsDF[reviewsDF['title']==xTitle]
    nReviews = filtered_df2.shape[0]
    revtext.write(f'Now showing {nReviews} reviews for: {xTitle}\n')
    # !!!!!
    revtext2.dataframe(filtered_df2[['title','overall','reviewText']], hide_index=True, use_container_width=True)    
    

with my_slot4:
    st.markdown(f"""<br><br><h5>Visualizations to Help Understand the Universe of Products Reviewed</h5>""", unsafe_allow_html=True)
  
# RIGHT COLUMN CONTENT
with my_slot5:
    with st.expander(label="WordCloud", expanded=True):
        sal = st.radio('Include text from reviews that were rated...', ['All (1-5)','Only Rating 4-5', 'Only Rating 1-2', 'Only Neutral'], horizontal=True )
        if sal=='All (1-5)':
            text = ' '.join(review for review in filtered_df2['reviewText_processed'])
        elif sal=='Only Rating 4-5':
            text = ' '.join(review for review in filtered_df2[(filtered_df2.overall==4)|(filtered_df2.overall==5)]['reviewText_processed'])
        elif sal=='Only Rating 1-2':
            text = ' '.join(review for review in filtered_df2[(filtered_df2.overall==1)|(filtered_df2.overall==2)]['reviewText_processed'])
        else:
            text = ' '.join(review for review in filtered_df2[(filtered_df2.overall==3)]['reviewText_processed'])

        text = ' '.join(review for review in filtered_df2['reviewText_processed'])
        # Generate the word cloud
        wordcloud = WordCloud(background_color='white', max_words=100, contour_width=3, contour_color='steelblue').generate(text)
        plt.figure(figsize=(10,7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

with my_slot6.expander(label=f"Overall Ratings {xTitle}", expanded=False):
        st.pyplot(plot_ratings_hist(filtered_df2))


with my_slot7.expander(label="List of most reviewed products", expanded=False):
    sampleDF = create_sample_df(100)
    # Seaborn Barplot
    plt.figure(figsize=(8, 6))
    sns.barplot(data=sampleDF, x='x', y='y')
    st.pyplot(plt)

# st.markdown("""---\n<h4>Reviews</h4>""", unsafe_allow_html=True)
st.subheader("Framing as a Machine Learning Problem")
st.write('Here, explain how the 5-scale rating will be collapsed into 3 classes: Pos, Neg, Neutral')
st.write('We will develop a machine learning model to predict the class based on a sentiment analysis of the review text')
with st.expander(label="Features Used for Classification"):
    st.write("Sentiment analysis is performed on each review's text and a prediction is made based on the score.")
    st.write("We will explore the accuracy of the prediction based on a series of gradual steps...")
    st.write("First, just throw the processed Review Text into a sentiment analysis model, TextBlob (or start w unprocessed?)")
    st.write("... How well does it perform? Where does it make errors? ")
    
with st.expander(label="Algorithms Used for Classification"):
    st.write("RandomForestClassifier has been used with these parameters...")
    st.write("Explain why to start with a Random Forest classifier...")


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

