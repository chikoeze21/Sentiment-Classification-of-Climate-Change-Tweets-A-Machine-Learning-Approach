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
    options = ["Prediction", "Information", "Home", "About Us"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Home":
        st.title("Welcome to Twitter Sentiment Classification")
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
        col1.image("resources/a.jpeg", caption="Data Scientist", width=200)
        col3.image("resources/i.jpg", caption="Data Scientist", width=200)
        col2.image("resources/c.jpg", caption="Data Scientist", width=200)

        desc1.subheader("Ayomide Aladetuyi")
        desc1.write("Ayomide is an experienced data scientist specializing in NLP. He has a strong background in machine learning and deep learning.")

        desc2.subheader("Chikezie Ezenna")
        desc2.write("Chikezie is an experienced data analyst specializing in drawing insights from data. He has a strong background in Finance.")

        desc3.subheader("Ifeoluwa Ayodele")
        desc3.write("Ife is an experienced data expert specializing in modelling. He has a strong background in Quality Assesment.")

        col4.image("resources/s.jpg", caption="Data Scientist", width=200)
        col5.image("resources/l.jpg", caption="Data Scientist", width=200)
        col6.image("resources/s.jpg", caption="Data Scientist", width=200)

        desc4.subheader("Siphosethu Matomela")
        desc4.write("Siphosethu is an experienced administrative specialist specialized in team alignment and management.")

        desc5.subheader("Lazarus Chukwueke")
        desc5.write("Lazarus is our Market researcher. He has a strong background in Market research.")

        desc6.subheader("Umphile Molaeng")
        desc6.write("Umphile is Data analyst specialized various technologies such as Power BI, Python and Python Libraries for Data Science.")


    if selection == "Information":
        st.info("General Information")
        with open("README.md", "r") as file:
          markdown_text = file.read()

    # Display the Markdown content
        st.markdown(markdown_text)
        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']])  # will write the df to the page


    # Building out the prediction page
    if selection == "Prediction":
        st.subheader("EDGE 1.0.")
        st.info("Prediction with ML Models.")

        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")

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
