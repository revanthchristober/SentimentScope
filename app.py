import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import base64
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit.components.v1 as components
from langdetect import detect
from googletrans import Translator
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import re
import requests
from bs4 import BeautifulSoup
import emoji
from PIL import Image
import matplotlib as mpl

app_icon = Image.open('icon.png')

# Set the page title to a custom text
# st.set_page_config(page_title='SentimentScope', layout='wide')

# Set the page config with a custom theme
st.set_page_config(
    page_title="SentimentScope",
    page_icon=app_icon,
    layout="wide",
)



# Apply the custom theme
st.markdown(
    """
    <style>
    .reportview-container {
            margin-top: -2em;
    }
    #MainMenu {visibility: hidden; }
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    #stDecoration {display:none;}
    """,
    unsafe_allow_html=True
)

# Custom CSS to inject
hide_cursor_style = """
<style>
select:active, select:hover {
    outline: none;
}
select {
    -moz-appearance: none;
    -webkit-appearance: none;
    appearance: none;
}
</style>
"""

# Inject custom CSS with markdown
st.markdown(hide_cursor_style, unsafe_allow_html=True)

# Load the English NER model
nlp = spacy.load("en_core_web_sm")

translator = Translator()

# Function to load the model and vectorizer (with caching)
@st.cache_data()
def load_model_and_vectorizer():
    model = joblib.load('model/random_forest_bow.pkl')
    vectorizer = joblib.load('model/badminton_vectorizer.pkl')
    return model, vectorizer

# Preprocessing function
def preprocess_reviews(reviews):
    processed_reviews = []
    for review in reviews:
        # Apply preprocessing steps (e.g., lowercasing, removing special characters and emojis)
        processed_review = review.lower()
        processed_review = re.sub(r'[^a-zA-Z\s]', '', processed_review)
        
        # Remove emojis
        processed_review = ''.join(char for char in processed_review if char not in emoji.UNICODE_EMOJI)
        
        processed_reviews.append(processed_review)
    return processed_reviews

loaded_model, loaded_vectorizer = load_model_and_vectorizer()

# Function for Named Entity Recognition
def ner_extraction(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function for topic modeling
def topic_modeling(text_data):
    vectorizer = CountVectorizer(stop_words='english')
    dtm = vectorizer.fit_transform(text_data)
    lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_model.fit(dtm)
    return lda_model, vectorizer

def listToString(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele
    # return string
    return str1

# Function to perform aspect-based sentiment analysis
def aspect_sentiment_analysis(text):
    sentiments = []
    for aspect in listToString(aspect_keywords):
        if aspect.text.lower() in text.lower():
            sentiment = predict(text)
            sentiments.append((aspect, sentiment))
    return sentiments

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return "Unknown"

# Title of the app
st.title('SentimentScope: The Flipkart Review Insight Engine')

def plot_histogram(df, column_name):
    st.subheader('Review Length Distribution')
    mpl.rcParams.update({'text.color' : "white",
                     'axes.labelcolor' : "white",
                     'xtick.color' : 'white',
                     'ytick.color' : 'white'})
    fig, ax = plt.subplots(facecolor='#00172B')
    sns.histplot(df[column_name].str.len(), bins=20, ax=ax)
    st.pyplot(fig)

def plot_scatter(df, text_column, sentiment_column):
    st.subheader('Sentiment Score vs Review Length')
    mpl.rcParams.update({'text.color' : "white",
                     'axes.labelcolor' : "white",
                     'xtick.color' : 'white',
                     'ytick.color' : 'white'})
    fig, ax = plt.subplots(facecolor='#00172B')
    sentiment_score = df[sentiment_column].apply(lambda x: 1 if x == 'Positive' else 0)
    sns.scatterplot(x=df[text_column].str.len(), y=sentiment_score, ax=ax)
    st.pyplot(fig)

def plot_box(df, column_name):
    st.subheader('Review Length Spread')
    mpl.rcParams.update({'text.color' : "white",
                     'axes.labelcolor' : "white",
                     'xtick.color' : 'white',
                     'ytick.color' : 'white'})
    fig, ax = plt.subplots(facecolor='#00172B')
    sns.boxplot(x=df[column_name].str.len(), ax=ax)
    st.pyplot(fig)

def plot_heatmap(df):
    st.subheader('Feature Correlation Heatmap')
    mpl.rcParams.update({'text.color' : "white",
                     'axes.labelcolor' : "white",
                     'xtick.color' : 'white',
                     'ytick.color' : 'white'})
    fig, ax = plt.subplots(facecolor='#00172B')
    sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax)
    st.pyplot(fig)

def plot_time_series(df, date_column, sentiment_column):
    st.subheader('Sentiment Over Time')
    time_data = df.groupby([date_column, sentiment_column]).size().unstack(fill_value=0)
    time_data.plot(kind='line', figsize=(10, 5))
    st.pyplot()

def plot_violin(df, sentiment_column):
    st.subheader('Sentiment Score Distribution')
    fig, ax = plt.subplots(facecolor='#00172B')
    mpl.rcParams.update({'text.color' : "white",
                     'axes.labelcolor' : "white"})
    
    sns.violinplot(x=df[sentiment_column].apply(lambda x: 1 if x == 'Positive' else 0), ax=ax)
    st.pyplot(fig)

# Function to remove custom stopwords
def remove_custom_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in custom_stopwords]
    return ' '.join(filtered_words)

# Function to predict sentiment
def predict(input_text):
    # Transform the input text
    text_vector = loaded_vectorizer.transform([input_text])
    
    # Make prediction
    sentiment = loaded_model.predict(text_vector)
    
    return sentiment[0]

# Function to create visualizations
# def create_visualizations(sentiment_counts, text_data=None):
#     # Create a bar chart of sentiment distribution
#     st.title('Bar Chart')
#     st.bar_chart(sentiment_counts)

#     mpl.rcParams.update({'text.color' : "white",
#                      'axes.labelcolor' : "white"})
    
#     # Create a pie chart of sentiment distribution
#     fig, ax = plt.subplots(facecolor='#00172B')
#     st.title('Pie Chart')
#     ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
#     ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#     ax.set_facecolor('#00172B')
#     st.pyplot(fig)

#     # Generate a word cloud if text data is provided
#     if text_data is not None:
#         wordcloud = WordCloud(background_color='#00172B').generate(" ".join(text_data))
#         st.title('WordCloud Plot')
#         plt.imshow(wordcloud, interpolation='bilinear')
#         plt.axis('off')
#         st.pyplot(plt)

def create_visualizations(sentiment_counts, text_data=None):
    # Create a bar chart of sentiment distribution
    st.title('Bar Chart')
    st.bar_chart(sentiment_counts)

    mpl.rcParams.update({'text.color': "white",
                         'axes.labelcolor': "white"})

    # Create a pie chart of sentiment distribution
    fig, ax = plt.subplots(facecolor='#00172B')
    st.title('Pie Chart')
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_facecolor('#00172B')
    st.pyplot(fig)

    # Generate a word cloud if text data is provided and it's not empty
    if text_data is not None and any(text_data):
        # Apply preprocessing to text data
        processed_text = preprocess_reviews(text_data)
        # Concatenate all processed reviews into a single string
        all_text = " ".join(processed_text)
        # Check if there are enough words for word cloud
        if all_text.strip():  # Check if text is not empty after preprocessing
            wordcloud = WordCloud(background_color='#00172B').generate(all_text)
            st.title('WordCloud Plot')
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
        else:
            st.warning("No words available to generate a word cloud.")



# Function to download CSV file
def download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_analysis.csv">Download CSV file</a>'
    st.markdown(href, unsafe_allow_html=True)

# Function to display interactive data tables
def interactive_table(df):
    components.html(df.to_html(escape=False, index=False), height=300)

# Function to analyze sentiment intensity
def sentiment_intensity(df, text_column):
    analyzer = SentimentIntensityAnalyzer()
    df['Intensity'] = df[text_column].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    st.write(df[['Sentiment', 'Intensity']])

# Define the options for input
input_options = ['Enter a Review Text (default)', 'Upload CSV/TEXT/XLSX/JSON', 'Enter a Flipkart Product URL']

# Create the dropdown menu for input selection
selected_input = st.selectbox('Select Input Type:', input_options)

# Conditional logic based on the user's selection
if selected_input == 'Enter a Review Text (default)':
    # user_text = st.text_area('Enter your text here:')
    # Add logic to process the text
    # Text input for user review
    user_review = st.text_area("Enter a review text:")

    selected_language = st.selectbox('Filter reviews by language (optional)', ['Select a Language', 'English (default)', 'French', 'Spanish', 'German'])

    # Assuming aspect keywords are provided by the user
    aspect_keywords = st.text_input("Enter aspect keywords (comma-separated)", "")
    aspect_keywords = aspect_keywords.split(',')

    custom_stopwords = st.text_input("Enter custom stopwords (comma-separated)", "")
    custom_stopwords = custom_stopwords.split(',')

    # Apply custom stopwords removal
    if custom_stopwords:
        user_review = remove_custom_stopwords(user_review)

    # Apply language detection to reviews
    if selected_language != 'English (default)' and selected_language != 'Select a Language':
        review_language = detect_language(user_review)
        user_review2 = user_review[review_language == selected_language]
        st.write(f'Filtered reviews for language: {selected_language}')
        user_review_og = user_review
        user_review = translator.translate(user_review, dest='en').text

    # Predict button
    if st.button('**Predict**'):
        with st.spinner('**Predicting**...'):
            if user_review:
                # Display sentiment and visualization
                sentiment = predict(user_review)
                if sentiment == 'Positive':
                    st.success(f'The sentiment of the review is: {sentiment}')
                else:
                    st.error(f'The sentiment of the review is: {sentiment}')
                
                # Apply aspect-based sentiment analysis
                if aspect_keywords:
                    # df['Aspect Sentiments'] = df[selected_column].apply(aspect_sentiment_analysis)
                    # st.write(df[['Review', 'Aspect Sentiments']])
                    if detect_language(aspect_keywords) != 'English':
                        aspect_keywords = translator.translate(aspect_keywords, dest='en')
                        # dir(aspect_keywords)
                        
                        st.success(f"Review: **{user_review}**")
                        # st.success(f"Aspect Sentiments: {type(aspect_keywords)}")
                        st.success(f"Aspect Sentiments: **{[aspect_keyword.text for aspect_keyword in aspect_keywords]}**")

                    elif detect_language(aspect_keywords) == 'English':
                        aspect_sentiments = aspect_sentiment_analysis(user_review)
                        st.write(f"Review: **{user_review}**")
                        # st.success(f"Aspect Sentiments: {type(aspect_keywords)}")
                        st.success(f"Aspect Sentiments: **{aspect_sentiments}**")

                review_entity = ner_extraction(user_review)
                # st.success(f"Extracted Entities: {review_entity}")

                lda_model, vectorizer = topic_modeling(np.array([user_review]))
                st.write("### **Top 5 topics**:")
                topics_df = pd.DataFrame()
                for i, topic in enumerate(lda_model.components_):
                    top_features = [vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-10:]]
                    topic_df = pd.DataFrame(top_features, columns=[f'Topic {i+1}'])
                    topics_df = pd.concat([topics_df, topic_df], axis=1).drop_duplicates()

                st.dataframe(topics_df)
                
                # Create visualizations for a single review
                st.write('## **Visualizations**:')
                try:
                    if user_review_og: create_visualizations(pd.Series([sentiment]).value_counts(), [user_review_og])
                except: create_visualizations(pd.Series([sentiment]).value_counts(), [user_review])
                user_review = None
            else:
                st.warning('Please enter a review text or upload a file to predict sentiment.')

elif selected_input == 'Upload CSV/TEXT/XLSX/JSON':
    # File uploader in the center
    uploaded_file = st.file_uploader("Upload your input file", type=["csv", "txt", "json", "xlsx"])
    # If a file is uploaded, process the file
    if uploaded_file is not None:
        try:
            # Read the file based on its type
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.type == "application/json":
                df = pd.read_json(uploaded_file)
            elif uploaded_file.type == "text/plain":
                content = uploaded_file.getvalue().decode("utf-8")
                # Split the content by newline characters ("\n")
                lines = content.split("\n")
                # Create a DataFrame from the lines
                df = pd.DataFrame(lines, columns=['Review text'])
            else:
                st.error("Unsupported file type")
                st.stop()

            # Language selection dropdown
            selected_language = st.selectbox('Filter by language (optional)', ['Select Language', 'English', 'French', 'Spanish', 'German'])

            # Allow the user to select a column for sentiment analysis
            selected_column = st.selectbox('Select the column to analyze', df.columns)
            original_text_data = df[selected_column].tolist() 
            
            custom_stopwords = st.text_input("Enter custom stopwords (comma-separated)", "")
            custom_stopwords = custom_stopwords.split(',')
            if st.button('**Predict**'):
                with st.spinner('**Please Wait**...'):
                    # Apply custom stopwords removal
                    if custom_stopwords:
                        # user_review = remove_custom_stopwords(user_review)
                        df[selected_column] = df[selected_column].apply(remove_custom_stopwords)

                    # Apply language detection to reviews
                    if selected_language != 'English' and selected_language!='Select Language':
                        df['Language'] = df[selected_column].apply(detect_language)
                        df = df[df['Language'] == selected_language]
                        st.write(f'Filtered reviews for language: {selected_language}')

                    df['Sentiment'] = df[selected_column].apply(predict)
                    st.write('### **Predictions**:')
                    st.dataframe(df['Sentiment'])

                    df['Entities'] = df[selected_column].apply(ner_extraction)
                    st.write('### **Extracted Entities**:')
                    st.dataframe(df[[selected_column, 'Entities']])

                    # Analyze sentiment intensity
                    st.write('### **Analysed Sentiment Intensities**:')
                    sentiment_intensity(df, selected_column)

                    # Apply topic modeling
                    # if df[selected_column]:
                    lda_model, vectorizer = topic_modeling(df[selected_column])
                    st.write("### **Top 5 topics**:")
                    # for i, topic in enumerate(lda_model.components_):
                    #     st.write(f"Topic {i+1}:")
                    #     st.write([vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-10:]])
                    topics_df = pd.DataFrame()
                    for i, topic in enumerate(lda_model.components_):
                        top_features = [vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-10:]]
                        topic_df = pd.DataFrame(top_features, columns=[f'Topic {i+1}'])
                        topics_df = pd.concat([topics_df, topic_df], axis=1).drop_duplicates()

                    st.dataframe(topics_df.drop_duplicates())

                    st.write('## Combined Data:')
                    st.dataframe(df)

                    # Create visualizations based on the predicted sentiments
                    st.write('## **Visualizations**:')
                    left_col, center_col, right_col = st.columns(3)
                    with left_col:
                        create_visualizations(df['Sentiment'].value_counts(), original_text_data)
                    with right_col:
                    # Call visualization functions
                        plot_histogram(df, selected_column)
                        plot_scatter(df, selected_column, 'Sentiment')
                        plot_box(df, selected_column)

                    # Display interactive data tables
                    # interactive_table(df)
                    
                    
                    # Provide a download link for the results
                    # download_link(df)

                    uploaded_file = None
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Add logic to process the file

elif selected_input == 'Enter a Flipkart Product URL':
    product_url = st.text_input('#### **Enter the Flipkart product URL here**:')
    # Language selection dropdown
    selected_language = st.selectbox('Filter by language (optional)', ['Select Language', 'English', 'French', 'Spanish', 'German'])

    aspect_keywords = st.text_input("Enter aspect keywords (comma-separated)", "")
    aspect_keywords = aspect_keywords.split(',')

    # Add logic to scrape and process the reviews from the URL
    # User-Agent and Accept-Language headers
    headers = {
        'User-Agent': 'Use your own user agent',
        'Accept-Language': 'en-us,en;q=0.5'
    }

    customer_names = []
    review_title = []
    ratings = []
    review_text = []

    if st.button('**Analyse** & **Predict**') and product_url:
        with st.spinner('**Please Wait**...'):
            for i in range(1, 44):
                # Construct the URL for the current page
                url = product_url
                # Send a GET request to the page
                page = requests.get(url, headers=headers)

                # Parse the HTML content
                soup = BeautifulSoup(page.content, 'html.parser')

                # Extract customer names
                names = soup.find_all('p', class_='_2sc7ZR _2V5EHH')
                for name in names:
                    customer_names.append(name.get_text())

                # Extract review titles
                title = soup.find_all('p', class_='_2-N8zT')
                for t in title:
                    review_title.append(t.get_text())

                # Extract ratings
                rat = soup.find_all('div', class_='_3LWZlK _1BLPMq')
                for r in rat:
                    rating = r.get_text()
                    if rating:
                        ratings.append(rating)
                    else:
                        ratings.append('0')  # Replace null ratings with 0

                # Extract reviews
                cmt = soup.find_all('div', class_='t-ZTKy')
                for c in cmt:
                    comment_text = c.div.div.get_text(strip=True)
                    review_text.append(comment_text)

            # Ensure all lists have the same length
            min_length = min(len(customer_names), len(review_title), len(ratings), len(review_text))
            customer_names = customer_names[:min_length]
            review_title = review_title[:min_length]
            ratings = ratings[:min_length]
            review_text = review_text[:min_length]

            # Create a DataFrame from the collected data
            data = {
                'Customer Name': customer_names,
                'Review Title': review_title,
                'Rating': ratings,
                'Review text': review_text
            }

            
            df = pd.DataFrame(data).drop_duplicates()
            st.write('## **Extracted Reviews**:')
            st.dataframe(df)

            # Applying the preprocessing
            preprocessed_reviews = preprocess_reviews(df['Review text'].tolist())

            # Predicting the sentiment for each review from the preprocessed review data.
            predicted_sentiments = [predict(review) for review in preprocessed_reviews]

            # Converting the predicted sentiments to a DataFrame.
            reviews_df = pd.DataFrame({'Review': df['Review text'].tolist(), 'Sentiment': predicted_sentiments})

            reviews_df = reviews_df.drop_duplicates()

            # Apply language detection to reviews
            if selected_language != 'English' and selected_language != 'Select Language':
                df['Language'] = df['Review text'].apply(detect_language)
                df = df[df['Language'] == selected_language]
                st.write(f'Filtered reviews for language: {selected_language}')

            
            # custom_stopwords = st.text_input("Enter custom stopwords (comma-separated) (optional)", "")
            # custom_stopwords = custom_stopwords.split(',')

            # # Apply custom stopwords removal
            # if custom_stopwords:
            #     # user_review = remove_custom_stopwords(user_review)
            #     df['Review text'] = df[selected_column].apply(remove_custom_stopwords)
                    
            # Render an interactive data table
            st.write('### **Predictions**:')
            st.dataframe(reviews_df)
            # Performing analyse sentiment intensity
            st.write('### **Analysed Sentiment Intensities**:')
            sentiment_intensity(reviews_df, 'Review')

            reviews_df['Entities'] = reviews_df['Review'].apply(ner_extraction)
            st.write('### **Extracted Entities**:')
            st.write(reviews_df[['Review', 'Entities']])
        
            # Apply topic modeling
            lda_model, vectorizer = topic_modeling(reviews_df['Review'])
            st.write("### **Top 5 topics**:")
            # for i, topic in enumerate(lda_model.components_):
            #     st.write(f"Topic {i+1}:")
            #     st.write([vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-10:]])
            topics_df = pd.DataFrame()
            for i, topic in enumerate(lda_model.components_):
                top_features = [vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-10:]]
                topic_df = pd.DataFrame(top_features, columns=[f'Topic {i+1}'])
                topics_df = pd.concat([topics_df, topic_df], axis=1).drop_duplicates()

            st.dataframe(topics_df)
            
            # Creating Visualizations on them.
            st.write('## **Visualizations**:')
            left_col, center_col, right_col = st.columns(3)
            with left_col:
                create_visualizations(reviews_df['Sentiment'].value_counts(), df['Review text'].tolist())
            with right_col:
                plot_histogram(reviews_df, 'Review')
                plot_scatter(reviews_df, 'Review', 'Sentiment')
                plot_box(reviews_df, 'Review')


