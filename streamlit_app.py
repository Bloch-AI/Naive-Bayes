import streamlit as st
import pandas as pd
import string
from collections import Counter
import matplotlib.pyplot as plt
import nltk

# Download NLTK resources (only needed on first run)
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# =============================================================================
# Custom Pre-processing Functions (Tokenisation, Stop word Removal, Stemming)
# =============================================================================
def custom_tokenizer(text):
    """
    Convert text to lowercase, remove punctuation, tokenise, remove stop words,
    and apply stemming using a SnowballStemmer (for UK English, we use the English stemmer).
    """
    # Lowercase the text (UK English convention)
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenise the text
    tokens = word_tokenize(text)
    # Remove stop words (using NLTK's English stopwords)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Apply stemming (using SnowballStemmer for English)
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

# =============================================================================
# Training Data (Extended Sample Data in UK English)
# =============================================================================
def get_training_data():
    data = {
        "review": [
            "The food was absolutely delicious and the service was excellent.",
            "I loved the wonderful ambience and tasty meals.",
            "Fantastic dining experience with great flavours.",
            "The restaurant had a cosy atmosphere and superb cuisine.",
            "A delightful meal with outstanding service.",
            "The food was disappointing and bland.",
            "I had a mediocre experience with slow service.",
            "The restaurant was awful and the food was terrible.",
            "Poor quality food and unfriendly staff.",
            "A subpar dining experience overall.",
            "Exquisite plating and exceptional taste.",
            "An average meal with no distinctive flavour.",
            "The waiters were friendly but the food was underwhelming.",
            "Superb desserts and a pleasant dining environment.",
            "The ambience was dull and the meal was unsatisfactory.",
            "A brilliant culinary experience with delightful presentation.",
            "Mediocre service and poor food presentation."
        ],
        "sentiment": [
            "Positive", "Positive", "Positive", "Positive", "Positive",
            "Negative", "Negative", "Negative", "Negative", "Negative",
            "Positive", "Negative", "Negative", "Positive", "Negative", "Positive", "Negative"
        ]
    }
    return pd.DataFrame(data)

# =============================================================================
# Train Model with Caching
# =============================================================================
@st.cache_resource
def train_model(nb_variant):
    """
    Trains a Naive Bayes classifier using the chosen variant.  
    The TfidfVectorizer utilises a custom tokenizer that implements our pre-processing steps.
    """
    df = get_training_data()
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Create a TF‑IDF vectoriser with our custom tokenizer.
    # We disable automatic lowercasing since our custom_tokenizer already does that.
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, lowercase=False)
    X = vectorizer.fit_transform(df['review'])
    y = df['sentiment']

    # Train the chosen Naive Bayes model
    if nb_variant == "Multinomial":
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB()
        model.fit(X, y)
    elif nb_variant == "Bernoulli":
        from sklearn.naive_bayes import BernoulliNB
        model = BernoulliNB()
        model.fit(X, y)
    elif nb_variant == "Gaussian":
        from sklearn.naive_bayes import GaussianNB
        # GaussianNB requires a dense array
        model = GaussianNB()
        model.fit(X.toarray(), y)
    else:
        st.error("Unsupported Naive Bayes variant selected.")
        return None, None

    return model, vectorizer

# =============================================================================
# Streamlit App Layout
# =============================================================================
st.title("Naive Bayes Simulator for Restaurant Reviews (UK English)")
st.markdown("""
Naive Bayes is a simple yet remarkably effective machine learning algorithm.  
This simulator demonstrates how a Naive Bayes classifier predicts the sentiment of a restaurant review.  
In this demo, we have expanded the training data and explicitly show key data pre‑processing steps such as tokenisation,  
stop word removal and stemming. Enjoy exploring the process and the results!
""")

# -----------------------------
# Sidebar Configuration
# -----------------------------
st.sidebar.header("Configuration")
nb_variant = st.sidebar.selectbox("Select Naive Bayes Variant", options=["Multinomial", "Bernoulli", "Gaussian"])
show_preprocessing = st.sidebar.checkbox("Show Data Pre-processing Visualisations", value=True)

# -----------------------------
# Train Model
# -----------------------------
model, vectorizer = train_model(nb_variant)

# =============================================================================
# Data Pre-processing Visualisations
# =============================================================================
if show_preprocessing:
    st.subheader("Data Pre-processing Visualisation")
    df = get_training_data()
    
    # Sample review to demonstrate processing steps
    sample_review = st.text_area(
        "Enter a sample review to view the pre‑processing steps:",
        "The restaurant had an exceptional ambience, and the flavours were outstanding!"
    )
    st.markdown("**Original Review:**")
    st.write(sample_review)
    
    # Display tokenised and stemmed version of the sample review
    tokens = custom_tokenizer(sample_review)
    st.markdown("**Tokenised & Stemmed Version:**")
    st.write(tokens)
    
    # Compute the frequency distribution of tokens in the entire training dataset
    all_tokens = []
    for review in df['review']:
        all_tokens.extend(custom_tokenizer(review))
    word_freq = Counter(all_tokens)
    top_words = word_freq.most_common(10)
    
    if top_words:
        words, freqs = zip(*top_words)
        freq_df = pd.DataFrame({"Word": words, "Frequency": freqs})
        st.markdown("**Top 10 Most Frequent Tokens in the Training Data:**")
        st.bar_chart(freq_df.set_index("Word"))
    else:
        st.write("No tokens found.")

# =============================================================================
# User Input and Prediction
# =============================================================================
st.subheader("Enter a Restaurant Review")
default_review = "I loved the delicious food and friendly service."
user_review = st.text_area("Your Review:", default_review)

if st.button("Predict Sentiment"):
    if not user_review:
        st.error("Please enter a review to analyse.")
    else:
        # Transform the input review using the same vectoriser.
        X_new = vectorizer.transform([user_review])
        if nb_variant == "Gaussian":
            X_new = X_new.toarray()
        
        # Predict sentiment and class probabilities.
        prediction = model.predict(X_new)[0]
        try:
            prediction_proba = model.predict_proba(X_new)[0]
            proba_df = pd.DataFrame([prediction_proba], columns=model.classes_)
        except AttributeError:
            proba_df = pd.DataFrame({"Info": ["Probability scores not available."]})
        
        st.subheader("Prediction")
        st.write(f"**Sentiment:** {prediction}")
        st.subheader("Class Probabilities")
        st.write(proba_df)

st.markdown("""
---
### How Does Naive Bayes Work?
Naive Bayes calculates the probability that a document (or review) belongs to a given class (e.g. Positive or Negative)  
by combining the probabilities of each token (word) appearing in documents of that class.  
Despite the ‘naive’ assumption that each token is independent of the others, the algorithm works remarkably well in practice.  
This demo also illustrates how pre‑processing steps—such as tokenisation, stop word removal, and stemming—prepare the text for classification.
""")
