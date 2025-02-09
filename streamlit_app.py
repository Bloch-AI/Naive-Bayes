import streamlit as st
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt

# Import NLTK components (only for stopwords and stemming)
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# =============================================================================
# Custom Tokenizer (using regex; avoids dependency on NLTK's "punkt")
# =============================================================================
def custom_tokenizer(text):
    """
    Converts text to lowercase, extracts words using a regular expression,
    removes stop words, and applies stemming.
    """
    # Lowercase the text
    text = text.lower()
    # Extract words using regex (matches sequences of word characters)
    tokens = re.findall(r'\b\w+\b', text)
    
    # Load stopwords if available; otherwise, use an empty set
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
    # Create a TfidfVectorizer that uses our custom tokenizer (no automatic lowercasing)
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, lowercase=False)
    X = vectorizer.fit_transform(df['review'])
    y = df['sentiment']
    
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
        model = GaussianNB()
        # GaussianNB requires a dense array
        model.fit(X.toarray(), y)
    else:
        st.error("Unsupported Naive Bayes variant selected.")
        return None, None
    
    return model, vectorizer

# =============================================================================
# Function to Compute Token-Level Sentiment Association
# (Only works for discrete NB models that provide feature_log_prob_)
# =============================================================================
def get_token_sentiments(tokens, model, vectorizer):
    token_sentiments = []
    classes = model.classes_
    # Check if we have the expected class labels
    if "Positive" in classes and "Negative" in classes:
        pos_index = list(classes).index("Positive")
        neg_index = list(classes).index("Negative")
        # Process unique tokens to avoid duplicates
        unique_tokens = sorted(set(tokens))
        for token in unique_tokens:
            if token in vectorizer.vocabulary_:
                col_index = vectorizer.vocabulary_[token]
                lp_token_pos = model.feature_log_prob_[pos_index, col_index]
                lp_token_neg = model.feature_log_prob_[neg_index, col_index]
                diff = lp_token_pos - lp_token_neg
                if diff > 0:
                    assoc = "Positive"
                elif diff < 0:
                    assoc = "Negative"
                else:
                    assoc = "Neutral"
                token_sentiments.append({
                    "Token": token,
                    "Association": assoc,
                    "Score": round(diff, 3)
                })
            else:
                token_sentiments.append({
                    "Token": token,
                    "Association": "Unknown",
                    "Score": None
                })
        return pd.DataFrame(token_sentiments)
    else:
        return pd.DataFrame()

# =============================================================================
# Streamlit App Layout
# =============================================================================
st.title("Naive Bayes Simulator for Restaurant Reviews (UK English)")

st.markdown("""
Naive Bayes is a simple yet remarkably effective machine learning algorithm.  
This simulator demonstrates how a Naive Bayes classifier predicts the sentiment of a restaurant review.  
**Note:** Due to a small training dataset, the default review may be classified as negative.
""")

# -----------------------------
# Sidebar Configuration
# -----------------------------
st.sidebar.header("Configuration")
nb_variant = st.sidebar.selectbox("Select Naive Bayes Variant", options=["Multinomial", "Bernoulli", "Gaussian"])

# -----------------------------
# Train the Model
# -----------------------------
model, vectorizer = train_model(nb_variant)

# -----------------------------
# User Input and Prediction
# -----------------------------
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
        
        # Predict overall sentiment.
        prediction = model.predict(X_new)[0]
        st.subheader("Prediction")
        st.write(f"**Sentiment:** {prediction}")
        
        # For discrete NB models, display token-level sentiment associations.
        if nb_variant in ["Multinomial", "Bernoulli"]:
            tokens = custom_tokenizer(user_review)
            token_df = get_token_sentiments(tokens, model, vectorizer)
            if not token_df.empty:
                st.subheader("Token-Level Sentiment Association")
                st.write("""
                Each token’s score represents the difference between its log probability for the positive class 
                and the negative class. A positive score indicates a positive association, while a negative 
                score indicates a negative association.
                """)
                st.dataframe(token_df)
            else:
                st.write("No token-level sentiment data available.")
        else:
            st.write("Token-level sentiment association is not available for the selected model.")

st.markdown("""
---
### How Does Naive Bayes Work?
Naive Bayes calculates the probability that a document (or review) belongs to a given class (e.g. Positive or Negative)  
by combining the probabilities of each token (word) appearing in documents of that class.  
Despite the ‘naive’ assumption that each token is independent, the algorithm works remarkably well in practice.  
This demo illustrates how pre‑processing steps — such as tokenisation, stop word removal, and stemming — prepare the text for classification.
""")


