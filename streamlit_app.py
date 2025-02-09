#**********************************************
# Naive Bayes Demo App
# Version 1.1
# 5th September 2024
# Your Name
# your.email@example.com
#**********************************************
# This Python code creates a web-based application using Streamlit to demonstrate
# the use of Naive Bayes classifiers for sentiment analysis of text reviews.
# Users can enter a restaurant review and select one of three Naive Bayes variants:
# - Multinomial NB
# - Bernoulli NB
# - Gaussian NB
#
# Key steps include:
# 1. Preprocessing the text (tokenisation, lemmatisation and removal of stopwords, including domain-specific ones).
# 2. Training a classifier on a predefined dataset of positive and negative reviews.
# 3. Allowing interactive sentiment analysis.
# 4. Displaying visualisations and a worked numerical calculation to explain the model's decision.
#
# For discrete NB models (Multinomial and Bernoulli), the app calculates token-level
# contributions and sums these with the model's log priors. For Gaussian NB, a pie chart
# of class probabilities is displayed alongside a simple calculation based on overall probabilities.
#
# An interactive slider allows the user to adjust the "Neutrality Threshold" – if the difference
# between the Positive and Negative scores is very small (below this threshold), the result will be shown as Neutral.
#**********************************************

import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk

# Ensure necessary NLTK data is downloaded for lemmatisation
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =============================================================================
# Custom Tokeniser Function
# =============================================================================
def custom_tokenizer(text):
    """
    Preprocess the input text by:
    - Converting it to lowercase.
    - Extracting words using a regular expression.
    - Removing standard stopwords and domain-specific tokens (e.g. "food", "service").
    - Applying lemmatisation to preserve the natural form of words (e.g. "friendly" remains "friendly").
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
    lemmatised_tokens = []
    for token in tokens:
        lemma = lemmatizer.lemmatize(token, pos='v')
        if lemma == token:
            lemma = lemmatizer.lemmatize(token, pos='n')
        lemmatised_tokens.append(lemma)
    
    return lemmatised_tokens

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
        "I had a marvellous time; the food was exquisite.",
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
    Uses a TfidfVectoriser with the custom_tokenizer on the predefined dataset.
    Returns the trained model and the vectoriser.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    df = get_training_data()
    vectoriser = TfidfVectorizer(tokenizer=custom_tokenizer, lowercase=False)
    X = vectoriser.fit_transform(df['review'])
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
    
    return model, vectoriser

# =============================================================================
# Token-Level Sentiment Association (for discrete NB models)
# =============================================================================
def get_token_sentiments(tokens, model, vectoriser):
    """
    For Multinomial and Bernoulli NB models, calculates the log-probability difference
    for each unique token between the Positive and Negative classes.
    Returns a DataFrame sorted by token contribution.
    """
    token_sentiments = []
    classes = model.classes_
    if "Positive" in classes and "Negative" in classes:
        pos_index = list(classes).index("Positive")
        neg_index = list(classes).index("Negative")
        unique_tokens = sorted(set(tokens))
        for token in unique_tokens:
            if token in vectoriser.vocabulary_:
                col_index = vectoriser.vocabulary_[token]
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
# Visualisation Function for Token-Level Sentiment Association
# =============================================================================
def plot_token_sentiments(token_df):
    """
    Creates a bar chart displaying the sentiment contribution of each token.
    Tokens with positive scores are shown in green, and those with negative scores in red.
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
# Theory and Background Section (Expandable)
# =============================================================================
with st.expander("Learn About Naive Bayes"):
    st.markdown(r"""
    **Naive Bayes Explained Simply:**
    
    Naive Bayes is a type of classifier that uses probability to decide which category a text belongs to.
    It is based on Bayes' Theorem, which is written as:
    
    \[
    P(C|X) = \frac{P(X|C) \times P(C)}{P(X)}
    \]
    
    - **\(P(C|X)\)**: The probability that the review belongs to class \(C\) (e.g. Positive or Negative) given the words \(X\) in the review.
    - **\(P(X|C)\)**: The likelihood of observing the words \(X\) if the review is of class \(C\).
    - **\(P(C)\)**: The prior probability of class \(C\) – how common the class is in the training data.
    - **\(P(X)\)**: A normalising constant that ensures all probabilities sum to 1.
    
    In simple terms, the classifier starts with a baseline belief about each class (the prior), looks at the words in the review (the likelihood), and updates its belief (the posterior). The class with the highest posterior probability is chosen.
    """, unsafe_allow_html=True)

# =============================================================================
# Interactive Slider Explanation
# =============================================================================
st.sidebar.header("Model Configuration")
nb_variant = st.sidebar.selectbox("Select Naive Bayes Variant", options=["Multinomial", "Bernoulli", "Gaussian"])
st.sidebar.markdown("""
**Neutrality Threshold:**  
Use the slider below to set the minimum difference between the log probabilities for the Positive and Negative classes.
If the difference is less than this threshold, the review is considered **Neutral**.
This helps to identify borderline cases.
""")
neutral_threshold = st.sidebar.slider("Neutrality Threshold (log difference)", 0.0, 1.0, 0.1, step=0.01)

# =============================================================================
# Main App Layout and Functionality
# =============================================================================
st.title("Naive Bayes Demo App for Sentiment Analysis")
st.markdown("""
Enter a restaurant review and choose a Naive Bayes variant (Multinomial, Bernoulli, or Gaussian)
to see the prediction, detailed token-level analysis (for discrete models), and a worked numerical calculation 
that explains the overall decision.
""")

# Train the model
model, vectoriser = train_model(nb_variant)

# Input section for user review
st.subheader("Enter a Restaurant Review")
default_review = "delicious food but very slow"
user_review = st.text_area("Your Review:", default_review)

if st.button("Predict Sentiment"):
    # Preprocess the review text
    tokens = custom_tokenizer(user_review)
    if not tokens:
        st.subheader("Prediction")
        st.write("**Sentiment:** Neutral (Not enough training data)")
        st.markdown("### Result Explanation")
        st.write("No tokens were obtained after pre-processing. This might be due to common words being removed. In such cases, the review is shown as Neutral.")
    else:
        X_new = vectoriser.transform([user_review])
        if nb_variant == "Gaussian":
            X_new = X_new.toarray()
        
        # Obtain overall class probabilities
        proba = model.predict_proba(X_new)[0]
        pos_index = list(model.classes_).index("Positive")
        neg_index = list(model.classes_).index("Negative")
        
        # Decide on overall sentiment based on the neutrality threshold
        if abs(proba[pos_index] - proba[neg_index]) < neutral_threshold:
            overall_sentiment = "Neutral"
        else:
            overall_sentiment = model.predict(X_new)[0]
        
        st.subheader("Prediction")
        st.write(f"**Sentiment:** {overall_sentiment}")
        
        # Simplified narrative explanation
        st.markdown("### Result Explanation")
        if overall_sentiment == "Positive":
            st.write("The review is classified as **Positive** because the evidence (from the words) favours the positive class.")
        elif overall_sentiment == "Negative":
            st.write("The review is classified as **Negative** because the evidence (from the words) favours the negative class.")
        else:
            st.write("The review is classified as **Neutral** because the difference between the positive and negative evidence is very small.")
        
        # Explain model variant differences
        if nb_variant == "Bernoulli":
            st.write("**Note for Bernoulli NB:** In this model, words are treated as binary features (present or not), so each word contributes its full effect, regardless of how many times it appears.")
        elif nb_variant == "Multinomial":
            st.write("**Note for Multinomial NB:** In this model, the frequency of words is considered; words that appear more often have a larger effect on the classification.")
        
        # For discrete NB models, display token-level analysis and a worked numerical calculation.
        if nb_variant in ["Multinomial", "Bernoulli"]:
            token_df = get_token_sentiments(tokens, model, vectoriser)
            if not token_df.empty:
                st.subheader("Token-Level Sentiment Association")
                st.write("""
                The table below shows the words extracted from your review along with their individual contribution scores.
                A positive score means the word supports a Positive classification, while a negative score supports a Negative classification.
                """)
                html_table = token_df.reset_index(drop=True).to_html(index=False)
                st.markdown(html_table, unsafe_allow_html=True)
                plot_token_sentiments(token_df)
                
                # --- Worked Numerical Calculation for Discrete NB ---
                st.markdown("### Worked Numerical Calculation (Discrete NB)")
                # The log prior difference reflects the inherent bias from the training data.
                log_prior_diff = model.class_log_prior_[pos_index] - model.class_log_prior_[neg_index]
                # The sum of the word contributions shows the overall evidence from the review.
                token_sum = token_df["Score"].sum()
                overall_log_diff = log_prior_diff + token_sum
                # Force neutrality if the overall difference is very small.
                if abs(overall_log_diff) < neutral_threshold:
                    overall_log_diff = 0
                    overall_sentiment = "Neutral"
                calc_str = (
                    f"**Log prior difference (Positive - Negative):** {log_prior_diff:.4f}\n\n"
                    "**Word Contributions:**\n"
                )
                for _, row in token_df.iterrows():
                    calc_str += f"- **{row['Token']}**: {row['Score']:.4f}\n"
                calc_str += (
                    f"\n**Sum of word contributions:** {token_sum:.4f}\n\n"
                    f"**Overall log probability difference:** {overall_log_diff:.4f}\n\n"
                    "This overall difference is the sum of the model's initial bias (log priors) and the evidence from the review.\n"
                )
                if overall_log_diff > 0:
                    calc_str += "Because the overall log difference is positive, the review is classified as **Positive**."
                elif overall_log_diff < 0:
                    calc_str += "Because the overall log difference is negative, the review is classified as **Negative**."
                else:
                    calc_str += "Because the overall log difference is zero (or nearly zero), the review is classified as **Neutral**."
                st.markdown(calc_str)
            else:
                st.write("No token-level sentiment data available.")
        
        # For Gaussian NB, display a pie chart and overall probability calculation.
        elif nb_variant == "Gaussian":
            st.subheader("Gaussian NB: Class Probability Distribution")
            st.write("Gaussian NB uses continuous features. Below is the distribution of class probabilities:")
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
            st.write("This difference shows which class is more likely. A positive difference indicates a Positive review; a negative difference indicates a Negative review. If the difference is very small, the review is considered Neutral.")
        else:
            st.write("Token-level sentiment analysis is not available for the selected model.")

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
        <p>© 2024 Bloch AI LTD - All Rights Reserved. <a href="https://www.bloch.ai" style="color: white;">www.bloch.ai</a></p>
    </div>
    ''',
    unsafe_allow_html=True
)

#**********************************************
# End of Naive Bayes Demo App
#**********************************************
