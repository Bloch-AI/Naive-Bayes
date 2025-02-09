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
    removes both standard stop words and domain-specific tokens, and applies stemming.
    """
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    
    # Load NLTK stopwords (if available)
    try:
        nltk_stopwords = set(stopwords.words('english'))
    except LookupError:
        nltk_stopwords = set()
    
    # Domain-specific tokens to remove
    domain_stopwords = {"food", "service", "restaurant", "meal", "dining"}
    
    # Combine both sets of stopwords
    all_stopwords = nltk_stopwords.union(domain_stopwords)
    
    # Filter out tokens that are in the combined stopword list
    tokens = [token for token in tokens if token not in all_stopwords]
    
    # Apply stemming using SnowballStemmer
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens

# =============================================================================
# Expanded Training Data (UK English)
# =============================================================================
def get_training_data():
    positive_reviews = [
        "The food was absolutely delicious and the service was excellent.",
        "I loved the wonderful ambience and tasty meals.",
        "Fantastic dining experience with great flavours.",
        "The restaurant had a cosy atmosphere and superb cuisine.",
        "A delightful meal with outstanding service.",
        "Exquisite plating and exceptional taste.",
        "The meal was perfect and the service was friendly.",
        "I had a marvelous time; the food was exquisite.",
        "The dishes were creative, and the flavours were divine.",
        "A truly memorable dining experience.",
        "Outstanding service and delicious food.",
        "The presentation was beautiful and the taste was exceptional.",
        "The staff was friendly, and the dishes were delightful.",
        "A perfect blend of taste and atmosphere.",
        "The food was amazing and the ambience was enchanting.",
        "I thoroughly enjoyed the meal and the service was impeccable.",
        "A top-notch experience with exquisite flavours.",
        "The chef did a fantastic job, and every dish was a delight.",
        "Incredible taste and a warm, welcoming atmosphere.",
        "Absolutely superb dining experience with delectable dishes."
    ]
    
    negative_reviews = [
        "The food was disappointing and bland.",
        "I had a mediocre experience with slow service.",
        "The restaurant was awful and the food was terrible.",
        "Poor quality food and unfriendly staff.",
        "A subpar dining experience overall.",
        "The waiters were rude and the food was underwhelming.",
        "The meal was cold and lacking in flavour.",
        "I was not impressed with the service or the food.",
        "The ambience was dull and the dishes were poorly prepared.",
        "An unpleasant experience; the food was tasteless.",
        "The restaurant was noisy, and the food was mediocre at best.",
        "I found the food greasy and the service unresponsive.",
        "Not worth the price, the quality was very low.",
        "The portions were small and the flavours were disappointing.",
        "I regret dining here, as the food was substandard.",
        "The experience was very poor with bland dishes.",
        "The food was overcooked and the service was slow.",
        "I did not enjoy the meal, and the staff was indifferent.",
        "The presentation was messy and the taste was off.",
        "Overall, it was an unsatisfactory dining experience."
    ]
    
    reviews = positive_reviews + negative_reviews
    sentiments = ["Positive"] * len(positive_reviews) + ["Negative"] * len(negative_reviews)
    
    return pd.DataFrame({"review": reviews, "sentiment": sentiments})

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
# Compute Token-Level Sentiment Association
# (Works for discrete NB models that provide feature_log_prob_)
# =============================================================================
def get_token_sentiments(tokens, model, vectorizer):
    token_sentiments = []
    classes = model.classes_
    # Only compute if both positive and negative classes are present
    if "Positive" in classes and "Negative" in classes:
        pos_index = list(classes).index("Positive")
        neg_index = list(classes).index("Negative")
        unique_tokens = sorted(set(tokens))
        for token in unique_tokens:
            if token in vectorizer.vocabulary_:
                col_index = vectorizer.vocabulary_[token]
                lp_token_pos = model.feature_log_prob_[pos_index, col_index]
                lp_token_neg = model.feature_log_prob_[neg_index, col_index]
                diff = lp_token_pos - lp_token_neg
                token_sentiments.append({"Token": token, "Score": diff})
            else:
                token_sentiments.append({"Token": token, "Score": 0})
        df = pd.DataFrame(token_sentiments)
        return df.sort_values("Score")
    else:
        return pd.DataFrame()

# =============================================================================
# Plot Token-Level Sentiment Association
# =============================================================================
def plot_token_sentiments(token_df):
    fig, ax = plt.subplots(figsize=(8, 4))
    tokens = token_df["Token"]
    scores = token_df["Score"]
    # Assign colours: green for positive scores, red for negative, gray for 0.
    colors = ['green' if score > 0 else 'red' if score < 0 else 'gray' for score in scores]
    ax.bar(tokens, scores, color=colors)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Token")
    ax.set_ylabel("Score")
    ax.set_title("Token-Level Sentiment Association")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

# =============================================================================
# Streamlit App Layout
# =============================================================================
st.title("Naive Bayes Simulator for Restaurant Reviews (UK English)")

st.markdown("""
Naive Bayes is a simple yet remarkably effective machine learning algorithm.  
This simulator demonstrates how a Naive Bayes classifier predicts the sentiment of a restaurant review.  
With an expanded training dataset and refined tokenisation (removing common domain words), the classifier should now provide more accurate predictions.
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
    # First, obtain tokens using the custom tokenizer.
    tokens = custom_tokenizer(user_review)
    if not tokens:
        st.subheader("Prediction")
        st.write("**Sentiment:** Neutral (Not enough training data)")
    else:
        X_new = vectorizer.transform([user_review])
        if nb_variant == "Gaussian":
            X_new = X_new.toarray()
        
        # Predict overall sentiment.
        prediction = model.predict(X_new)[0]
        st.subheader("Prediction")
        st.write(f"**Sentiment:** {prediction}")
        
        # For discrete NB models, display token-level sentiment associations.
        if nb_variant in ["Multinomial", "Bernoulli"]:
            token_df = get_token_sentiments(tokens, model, vectorizer)
            if not token_df.empty:
                st.subheader("Token-Level Sentiment Association")
                st.write("""
                The following table shows the tokens found in your review along with their sentiment association scores.
                A positive score (green) indicates a positive association, while a negative score (red) indicates a negative association.
                """)
                st.dataframe(token_df)
                plot_token_sentiments(token_df)
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
