import streamlit as st
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt

# Import NLTK components (only for stopwords and stemming)
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# =============================================================================
# Custom Tokenizer (using regex, no dependency on NLTK's "punkt")
# =============================================================================
def custom_tokenizer(text):
    """
    Converts text to lowercase, extracts words using a regular expression,
    removes stop words, and applies stemming.
    This approach avoids using nltk.tokenize.word_tokenize (which requires the "punkt" data).
    """
    # Lowercase the text
    text = text.lower()
    # Extract words using regex (matches sequences of word characters)
    tokens = re.findall(r'\b\w+\b', text)
    
    # Load stopwords if available; otherwise use an empty set
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        stop_words = set()
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    # Apply stemming using SnowballStemmer
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens

# =============================================================================
# Sample Training Data (UK English)
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
# Train Model (with caching)
# =============================================================================
@st.cache_resource
def train_model(nb_variant):
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    df = get_training_data()
    
    # Use a TfidfVectorizer with our custom tokenizer.
    # We disable automatic lowercasing since our tokenizer already lowercases.
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
# Train the Model
# -----------------------------
model, vectorizer = train_model(nb_variant)

# =============================================================================
# Data Pre-processing Visualisations
# =============================================================================
if show_preprocessing:
    st.subheader("Data Pre-processing Visualisation")
    df = get_training_data()
    
    # Allow the user to enter a sample review to visualise pre‑processing
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
    
    # Compute frequency distribution of tokens in the training data
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
        
        # Predict sentiment and display class probabilities.
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
