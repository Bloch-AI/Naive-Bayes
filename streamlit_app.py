#**********************************************
# Naive Bayes Demo App
# Version 1.0
# 5th September 2024
# Your Name
# your.email@example.com
#**********************************************
# This Python code creates a web-based application using Streamlit to demonstrate
# the use of Naive Bayes classifiers for sentiment analysis of text reviews.
# The application allows users to enter a restaurant review and select one of three Naive Bayes variants:
# - Multinomial NB
# - Bernoulli NB
# - Gaussian NB
#
# The code performs the following key steps:
# 1. Preprocesses text (tokenization, lemmatization, and removal of stopwords, including domain-specific words).
# 2. Trains the classifier on a predefined dataset of positive and negative reviews.
# 3. Provides an interactive input for sentiment analysis.
# 4. Displays visualizations and a worked numerical calculation explaining the model's decision.
#
# For discrete NB models (Multinomial and Bernoulli), token-level sentiment associations are calculated and shown.
# For Gaussian NB, a pie chart of class probability distribution and overall probability calculations are displayed.
#**********************************************

import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk

# Ensure necessary NLTK data is downloaded for lemmatization
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Import NLTK components for stopwords and lemmatization
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =============================================================================
# Custom Tokenizer Function
# =============================================================================
def custom_tokenizer(text):
    """
    Preprocess the input text by:
    - Converting to lowercase
    - Extracting words using a regular expression
    - Removing standard stopwords and domain-specific tokens (e.g. "food", "service")
    - Applying lemmatization to preserve the natural form (e.g. "friendly" remains "friendly")
    Returns a list of processed tokens.
    """
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    
    try:
        nltk_stopwords = set(stopwords.words('english'))
    except LookupError:
        nltk_stopwords = set()
    
    domain_stopwords = {"food", "service", "restaurant", "meal", "dining"}
    all_stopwords = nltk_stopwords.union(domain_stopwords)
    tokens = [token for token in tokens if token not in all_stopwords]
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for token in tokens:
        lemma = lemmatizer.lemmatize(token, pos='v')
        if lemma == token:
            lemma = lemmatizer.lemmatize(token, pos='n')
        lemmatized_tokens.append(lemma)
    
    return lemmatized_tokens

# =============================================================================
# Predefined Training Data (UK English Reviews)
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
# Model Training Function (with caching)
# =============================================================================
@st.cache_resource
def train_model(nb_variant):
    """
    Trains a Naive Bayes classifier based on the selected variant.
    Uses a TfidfVectorizer with the custom_tokenizer on a predefined dataset.
    Returns the trained model and the vectorizer.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    df = get_training_data()
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
        model.fit(X.toarray(), y)
    else:
        st.error("Unsupported Naive Bayes variant selected.")
        return None, None
    
    return model, vectorizer

# =============================================================================
# Token-Level Sentiment Association (for discrete NB models)
# =============================================================================
def get_token_sentiments(tokens, model, vectorizer):
    """
    For Multinomial and Bernoulli NB models, compute the difference in log-probabilities
    for each unique token between the Positive and Negative classes.
    Returns a DataFrame sorted by the token's contribution.
    """
    token_sentiments = []
    classes = model.classes_
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
# Visualization Function for Token-Level Sentiment Association
# =============================================================================
def plot_token_sentiments(token_df):
    """
    Creates a bar chart of token-level sentiment associations.
    Tokens with positive scores are shown in green; those with negative scores in red.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    tokens = token_df["Token"]
    scores = token_df["Score"]
    colors = ['green' if score > 0 else 'red' if score < 0 else 'gray' for score in scores]
    ax.bar(tokens, scores, color=colors)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Token")
    ax.set_ylabel("Score")
    ax.set_title("Token-Level Sentiment Association")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

# =============================================================================
# Main App Layout and Functionality
# =============================================================================
st.title("Naive Bayes Demo App for Sentiment Analysis")
st.markdown("""
This application demonstrates how a Naive Bayes classifier can be used for sentiment analysis.
Enter a restaurant review and choose a Naive Bayes variant (Multinomial, Bernoulli, or Gaussian) to see the prediction,
detailed token-level analysis (for discrete models), and a worked numerical calculation that explains the overall decision.
""")

# Sidebar for model selection
st.sidebar.header("Model Configuration")
nb_variant = st.sidebar.selectbox("Select Naive Bayes Variant", options=["Multinomial", "Bernoulli", "Gaussian"])

# Train the selected model
model, vectorizer = train_model(nb_variant)

# Input section for user review
st.subheader("Enter a Restaurant Review")
default_review = "I loved the delicious food and friendly service."
user_review = st.text_area("Your Review:", default_review)

if st.button("Predict Sentiment"):
    # Preprocess the review text
    tokens = custom_tokenizer(user_review)
    if not tokens:
        st.subheader("Prediction")
        st.write("**Sentiment:** Neutral (Not enough training data)")
        st.markdown("### Result Explanation")
        st.write("The review did not yield any tokens after pre-processing. This may be due to the removal of common words. In such cases, there isn’t enough information for the classifier, so the sentiment is treated as Neutral.")
    else:
        # Transform the review into features
        X_new = vectorizer.transform([user_review])
        if nb_variant == "Gaussian":
            X_new = X_new.toarray()
        
        # Obtain probability estimates and determine overall sentiment
        proba = model.predict_proba(X_new)[0]
        pos_index = list(model.classes_).index("Positive")
        neg_index = list(model.classes_).index("Negative")
        if abs(proba[pos_index] - proba[neg_index]) < 0.1:
            overall_sentiment = "Neutral"
        else:
            overall_sentiment = model.predict(X_new)[0]
        
        st.subheader("Prediction")
        st.write(f"**Sentiment:** {overall_sentiment}")
        
        # Narrative explanation of the result
        st.markdown("### Result Explanation")
        if overall_sentiment == "Positive":
            st.write("The review is classified as **Positive** because the tokens extracted (e.g. 'love', 'delicious', 'friendly') show strong positive associations in the model, contributing to a higher probability for the positive class.")
        elif overall_sentiment == "Negative":
            st.write("The review is classified as **Negative** because some tokens have strong negative associations. Please review the detailed token analysis below for more information.")
        else:
            st.write("The review is classified as **Neutral** because the difference between positive and negative probabilities is small, indicating insufficient evidence for a strong classification.")
        
        # Provide algorithm-specific explanation
        if nb_variant == "Bernoulli":
            st.write("**Note for Bernoulli NB:** In this model, tokens are treated as binary features (present/absent). This means that token frequency is not considered; each token's contribution is based solely on whether it is present in the review.")
        elif nb_variant == "Multinomial":
            st.write("**Note for Multinomial NB:** In this model, token frequency is taken into account; tokens that appear more often in positive reviews contribute more strongly to a positive classification.")
        
        # For discrete NB models, display token-level sentiment analysis and a worked calculation.
        if nb_variant in ["Multinomial", "Bernoulli"]:
            token_df = get_token_sentiments(tokens, model, vectorizer)
            if not token_df.empty:
                st.subheader("Token-Level Sentiment Association")
                st.write("""
                The table below shows the tokens extracted from your review along with their sentiment association scores.
                A positive score indicates a positive association, while a negative score indicates a negative association.
                """)
                html_table = token_df.reset_index(drop=True).to_html(index=False)
                st.markdown(html_table, unsafe_allow_html=True)
                plot_token_sentiments(token_df)
                
                # --- Worked Numerical Calculation for Discrete NB ---
                st.markdown("### Worked Numerical Calculation (Discrete NB)")
                log_prior_diff = model.class_log_prior_[pos_index] - model.class_log_prior_[neg_index]
                token_sum = token_df["Score"].sum()
                overall_log_diff = log_prior_diff + token_sum
                # Force neutrality if the overall difference is close to zero (threshold 0.1)
                if abs(overall_log_diff) < 0.1:
                    overall_log_diff = 0
                    overall_sentiment = "Neutral"
                calc_str = (
                    f"**Log prior difference (Positive - Negative):** {log_prior_diff:.4f}\n\n"
                    "**Token Contributions:**\n"
                )
                for _, row in token_df.iterrows():
                    calc_str += f"- **{row['Token']}**: {row['Score']:.4f}\n"
                calc_str += (
                    f"\n**Sum of token contributions:** {token_sum:.4f}\n\n"
                    f"**Overall log probability difference:** {overall_log_diff:.4f}\n\n"
                )
                if overall_log_diff > 0:
                    calc_str += "Since the overall log probability difference is positive, the review is classified as **Positive**."
                elif overall_log_diff < 0:
                    calc_str += "Since the overall log probability difference is negative, the review is classified as **Negative**."
                else:
                    calc_str += "Since the overall log probability difference is 0 (or nearly 0), the review is classified as **Neutral**."
                st.markdown(calc_str)
            else:
                st.write("No token-level sentiment data available.")
        
        # For Gaussian NB, display a pie chart and overall probability calculation.
        elif nb_variant == "Gaussian":
            st.subheader("Gaussian NB: Class Probability Distribution")
            st.write("Gaussian NB uses continuous features. Below is the distribution of class probabilities:")
            # Dynamically assign colors: Positive=green, Negative=red.
            colors = [("green" if cls == "Positive" else "red" if cls == "Negative" else "gray") for cls in model.classes_]
            fig, ax = plt.subplots()
            ax.pie(proba, labels=model.classes_, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.set_title("Class Probability Distribution")
            st.pyplot(fig)
            
            st.markdown("### Worked Numerical Calculation (Gaussian NB)")
            st.write(f"**Probability for Positive:** {proba[pos_index]:.4f}")
            st.write(f"**Probability for Negative:** {proba[neg_index]:.4f}")
            diff = proba[pos_index] - proba[neg_index]
            st.write(f"**Difference (Positive - Negative):** {diff:.4f}")
            if diff > 0:
                st.write("Since the difference is positive, the review is classified as **Positive**.")
            elif diff < 0:
                st.write("Since the difference is negative, the review is classified as **Negative**.")
            else:
                st.write("Since the difference is 0, the review is classified as **Neutral**.")
        else:
            st.write("Token-level sentiment association is not available for the selected model.")

# =============================================================================
# Footer Section
# =============================================================================
footer = st.container()
footer.markdown(
    '''
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: black;
        color: white;
        text-align: center;
        padding: 10px 0;
    }
    </style>
    <div class="footer">
        <p>© 2025 Bloch AI LTD - All Rights Reserved. <a href="https://www.bloch.ai" style="color: white;">www.bloch.ai</a></p>
    </div>
    ''',
    unsafe_allow_html=True
)

#**********************************************
# End of Naive Bayes Demo App
#**********************************************
