"""
Simple Streamlit webserver application for serving developed classification models.

Author: Explore Data Science Academy.

Note:
---------------------------------------------------------------------
Please follow the instructions provided within the README.md file
located within this directory for guidance on how to use this script
correctly.
---------------------------------------------------------------------

Description: This file is used to launch a minimal streamlit web
application. You are expected to extend the functionality of this script
as part of your predict project.

For further help with the Streamlit framework, see:
https://docs.streamlit.io/en/latest/
"""

# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image

# Data dependencies
import pandas as pd
import numpy as np

with open("README.md", "r") as file:
    markdown_text = file.read()

# Team member 1

# Team member 2
# st.subheader("Chikezie Ezenna")
# st.image("c.jpg", caption="Data Scientist", use_column_width=True)
# st.write("Chikezie is a data scientist with expertise in sentiment analysis and text mining. She has a keen interest in social media analytics.")



# Vectorizer
news_vectorizer = open("resources/vectorizer.pkl", "rb")
tweet_cv = joblib.load(news_vectorizer)  # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")


# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit"""

    # Creates a main title and subheader on your page -
    # these are static across all pages
    head, s,a,im= st.columns(4)
    head.title("Sigmoid Data Services")
    im.image("resources/Sigmoid.png", caption="Data talks", width=100)

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["EDGE 1.0", "Exploratory Data Analysis", "Home", "About Us"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Home":
        st.title("Welcome to Twitter Sentiment Classification")
        st.subheader("EDGE 1.0.")
        st.write("This application performs sentiment analysis on Twitter data.")
        st.write("Please navigate through the pages using the sidebar.")
        st.info("EDSA has tasked us with creating a Machine Learning model to classify individuals' belief in climate change based on their tweets. This solution helps companies gauge public sentiment, informing their marketing strategies for environmentally friendly products and services. By accurately classifying climate change beliefs, businesses gain valuable insights into consumer perceptions across demographics and geographic regions. This empowers them to address environmental concerns and reduce carbon footprints while aligning with their values. Ultimately, the model aids market research efforts, enhancing companies' understanding of customer sentiments and facilitating effective targeting and messaging.")

    if selection == "About Us":
        st.title("About Us")
        st.write("We are a team of data scientists passionate about natural language processing and sentiment analysis.")
        st.write("Our goal is to provide accurate sentiment classification for Twitter data to help businesses gain insights from social media.")

        st.header("Team Members")
        # Create three columns
        col1, col2, col3 = st.columns(3)
        desc1, desc2, desc3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        desc4,desc5,desc6 = st.columns(3)

        # Place the images in separate columns
        col1.image("resources/a.jpeg", caption="Chief Data Scientist", width=200)
        col3.image("resources/i.jpg", caption="Senior Developer", width=200)
        col2.image("resources/c.jpg", caption="Chief Technical Officer", width=200)

        desc1.subheader("Ayomide Aladetuyi")
        desc1.write("Ayomide is an experienced data scientist specializing in NLP. He has a strong background in machine learning and deep learning.")

        desc2.subheader("Chikezie Ezenna")
        desc2.write("Chikezie is an experienced data analyst specializing in drawing insights from data. He has a strong background in Finance.")

        desc3.subheader("Ifeoluwa Ayodele")
        desc3.write("Ife is an experienced data expert specializing in modelling. He has a strong background in Quality Assesment.")

        col4.image("resources/s.jpg", caption="Chief Administrative Officer", width=200)
        col5.image("resources/l.jpg", caption="Market Researcher", width=200)
        

        desc4.subheader("Siphosethu Matomela")
        desc4.write("Siphosethu is an experienced administrative specialist specialized in team alignment and management.")

        desc5.subheader("Lazarus Chukwueke")
        desc5.write("Lazarus is our Market researcher. He has a strong background in Market research.")


    if selection == "Exploratory Data Analysis":
        st.info("General Information")
        st.subheader("Climate Change Sentiment Analysis - Exploratory Data Analysis")
        st.write("This project aims to understand the public perception of climate change and its perceived threat level by analyzing individuals' beliefs based on their tweet data. The objective is to develop a machine learning model that can classify individuals' sentiments towards climate change, providing valuable insights for market research and future marketing strategies.")

        st.subheader("The dataset consists of tweet data with the following columns")
        st.write("- `sentiment`: Class label representing the belief in climate change (0: Neutral, 1: Pro, -1: Anti, 2: News)")
        st.write("- `message`: Tweet body")
        st.write("`tweetid`: Twitter unique ID")
        st.write("The training dataset contains 15,819 entries, while the test dataset contains 10,546 entries. The sentiment column contains four unique values, representing different classes of beliefs. The tweetid column is a case of high cardinality and will be dropped during the preprocessing phase.")
        
        st.subheader("Class Imbalance")
        st.write("The training dataset exhibits a class imbalance, with a majority of tweets falling under the 'Pro' sentiment category, indicating strong support for the belief in man-made climate change. This imbalance could lead to a model that performs well at categorizing certain sentiment categories but performs poorly on others.")
        st.image("resources/imgs/twitter_sentiment.png")
        
        st.subheader("Word Analysis")
        st.write("The unprocessed training dataset contains approximately 280,000 total words and 48,000 unique words. On average, each tweet contains around 3.158 unique words. This analysis provides insights into the language usage within the tweets.")
        
        st.subheader("Mentions and Hashtags Analysis")
        st.write("The analysis reveals that the 'Anti' and 'Pro' sentiments have the most mentions per tweet, with most tweets containing at least one mention. The top hashtags used in the tweets include 'Climate,' 'climate change,''Trump,' and 'Before the flood.' The hashtag 'Before the flood' is popular among pro-climate change tweets and refers to a 2016 documentary highlighting the dangers of climate change. The hashtag 'MAGA' (Make America Great Again) is popular in anti-climate change tweets, associated with support for Donald Trump.")
        st.image("resources/imgs/Hashtag.png")
        
        st.subheader("Popular Handles")
        st.write("The most popular news handles are actual news stations' handles, while the most popular pro handles consist of celebrity accounts and news station handles. Donald Trump is prominently mentioned in the most popular anti and neutral tweets.")
        st.image("resources/imgs/Handle.png")

        st.subheader("Tweet Preprocessing")
        st.write("The dataset contains various elements such as punctuations, URLs, emojis, hashtags, mentions, and retweets. A preprocessing function has been implemented to clean the tweet data, removing these elements to ensure better analysis and model performance.")
        
        st.subheader("Data Cleaning")
        st.write("-**Dropping TweetId Feature**: The `tweetId` feature is dropped from the dataset since it contains unique values and doesn't contribute significantly to the model's accuracy. Removing this feature helps reduce computational cost.")
        st.write("- **Stop Words**: Stop words, which are common words that don't carry significant meaning in a language, are typically removed from text analysis. However, it is not explicitly mentioned in the provided content whether stop words were removed or not.")
        st.write("- **Lemmatization**: Lemmatization is performed to reduce words to their base or root form (lemma). This process helps to standardize the text and reduce inflected forms. Examples of lemmatization include reducing 'running' to 'run' and 'cars' to 'car.'")
        st.write("- **Splitting Train Dataset**: The training dataset is split into a training set and a validation set. The training set is used to train the model, while the validation set is used to evaluate the model's performance on unseen data.")
        st.write("- **Feature Selection**: The dataset contains a large number of features (105,864), but not all of them contribute positively to the model's performance. Feature selection using the KBest method is employed to select the most relevant features for the model.")
        st.write("- **Correcting Imbalance with Oversampling**: Class imbalance is addressed using the Synthetic Minority Oversampling Technique (SMOTE). SMOTE generates synthetic data points to increase the representation of minority classes, improving the model's ability to predict all classes effectively.")
        st.write("- **Model Evaluation and Performance**: Classification models are developed, trained, and evaluated using accuracy, precision, and recall metrics. The Random Forest Classifier, Ridge Classifier, Support Vector Machine, Logistic Regression, Naive Bayes, and K-Nearest Neighbors classifiers are evaluated.")
        st.write("- **Random Forest Classifier Model**: The Random Forest Classifier is an ensemble learning algorithm that constructs multiple decision trees using random subsets of the training data and features. The model's accuracy is evaluated using a confusion matrix, indicating that it predicts the correct sentiment 71% of the time.")
        st.write("- **Ridge Classifier**: The Ridge Classifier, a linear model, is considered due to the high-dimensional nature of the problem. It shows high accuracy, precision, recall, and F1 scores.")
        st.write("- **Nearest Neighbor Classifier**: The K-Nearest Neighbor classifier, a nonparametric classification method, is considered to explore a different approach. It uses the plurality of votes from its neighbors for classification.")
        st.image("resources/imgs/Accuracy_score_green.png")
        st.write("- **Outcome**: The F1 scores for both the training and test datasets vary across the classifiers, with the highest scores observed for the Ridge Classifier (F1 Train: 0.993915, F1 Test: 0.738938).")





    
    # Building out the prediction page
    if selection == "EDGE 1.0":
        st.subheader("EDGE 1.0.")
        st.info("Prediction with ML Models.")

        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")
        st.info("2: News,    1: Pro,     0: Neutral,    -1: Anti")
        if st.button("Ridge Classification"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("resources/ridge_model.pkl"), "rb"))
            prediction = predictor.predict(vect_text)
            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))

        if st.button("Logistic Classification"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("resources/logistic_reg.pkl"), "rb"))
            prediction = predictor.predict(vect_text)
            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))

        if st.button("RF Classification"):
			# Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("resources/random_forest.pkl"), "rb"))
            prediction = predictor.predict(vect_text)
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))

        if st.button("Naives Bayes Classification"):
			# Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("resources/Naive_Bayes.pkl"), "rb"))
            prediction = predictor.predict(vect_text)
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))








# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
