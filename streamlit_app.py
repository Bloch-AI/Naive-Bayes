import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB

# ----------------------------
# Helper function to train the classifier
# ----------------------------
@st.cache_resource
def train_model(nb_variant):
    # A small sample dataset of restaurant reviews
    data = {
        "review": [
            "The food was absolutely delicious and the service was excellent.",
            "I loved the wonderful ambiance and tasty meals.",
            "Fantastic dining experience with great flavors.",
            "The restaurant had a cozy atmosphere and superb cuisine.",
            "A delightful meal with outstanding service.",
            "The food was disappointing and bland.",
            "I had a mediocre experience with slow service.",
            "The restaurant was awful and the food was terrible.",
            "Poor quality food and unfriendly staff.",
            "A subpar dining experience overall."
        ],
        "sentiment": [
            "Positive",
            "Positive",
            "Positive",
            "Positive",
            "Positive",
            "Negative",
            "Negative",
            "Negative",
            "Negative",
            "Negative"
        ]
    }
    df = pd.DataFrame(data)

    # Use TfidfVectorizer to convert text reviews into numerical features.
    # Stop words are removed to focus on the most meaningful words.
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['review'])
    y = df['sentiment']

    # Train the selected Naive Bayes model
    if nb_variant == "Multinomial":
        model = MultinomialNB()
        model.fit(X, y)
    elif nb_variant == "Bernoulli":
        model = BernoulliNB()
        model.fit(X, y)
    elif nb_variant == "Gaussian":
        # GaussianNB requires a dense array.
        model = GaussianNB()
        model.fit(X.toarray(), y)
    else:
        st.error("Unsupported Naive Bayes variant selected.")
        return None, None

    return model, vectorizer

# ----------------------------
# Streamlit App Layout
# ----------------------------

st.title("Naive Bayes Simulator for Restaurant Reviews")

st.markdown("""
Naive Bayes is a simple yet surprisingly effective machine learning algorithm.  
This simulator demonstrates how a Naive Bayes classifier works by predicting the sentiment 
of a restaurant review. You can choose between three variants:
- **Multinomial Naive Bayes:** Counts word frequencies.
- **Bernoulli Naive Bayes:** Checks for the presence or absence of words.
- **Gaussian Naive Bayes:** Assumes continuous features (here applied to TF‑IDF scores).

Enter a review below and see how the classifier makes its prediction!
""")

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("Configuration")
nb_variant = st.sidebar.selectbox("Select Naive Bayes Variant", options=["Multinomial", "Bernoulli", "Gaussian"])

# Train the model based on the selected variant.
model, vectorizer = train_model(nb_variant)

# ----------------------------
# User Input and Prediction
# ----------------------------
st.subheader("Enter a Restaurant Review")
default_review = "I loved the delicious food and friendly service."
user_review = st.text_area("Your Review:", default_review)

if st.button("Predict Sentiment"):
    if not user_review:
        st.error("Please enter a review to analyze.")
    else:
        # Transform the input review using the same vectorizer
        X_new = vectorizer.transform([user_review])
        if nb_variant == "Gaussian":
            # GaussianNB needs a dense array
            X_new = X_new.toarray()
        
        # Predict sentiment and class probabilities
        prediction = model.predict(X_new)[0]
        try:
            prediction_proba = model.predict_proba(X_new)[0]
            proba_df = pd.DataFrame([prediction_proba], columns=model.classes_)
        except AttributeError:
            # Some classifiers may not support predict_proba.
            proba_df = pd.DataFrame({"Info": ["Probability scores not available."]})
        
        st.subheader("Prediction")
        st.write(f"**Sentiment:** {prediction}")

        st.subheader("Class Probabilities")
        st.write(proba_df)

# ----------------------------
# Additional Explanation
# ----------------------------
st.markdown("""
---
### How Does Naive Bayes Work?
Naive Bayes calculates the probability that a document (or review) belongs to each class (e.g. Positive or Negative) 
by combining the probabilities of each word appearing in documents of that class. Despite the “naive” assumption that 
each word is independent of the others, the algorithm works remarkably well in practice. This is especially true 
in text classification tasks, such as filtering spam emails or analyzing customer reviews.
""")
