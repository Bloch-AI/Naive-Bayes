#**********************************************
# Naive Bayes Demo App
# Version 1.1
# 5th September 2024
# Your Name
# your.email@example.com
#**********************************************
# This Streamlit app shows how Naive Bayes can be used to decide if a restaurant review
# is positive or negative. You can try out three types of models:
# - Multinomial NB
# - Bernoulli NB
# - Gaussian NB
#
# The app does the following:
# 1. Cleans and prepares the text (it lowers the case, removes common words, and simplifies words).
# 2. Trains the model using a small set of example positive and negative reviews.
# 3. Lets you enter your own review to see what the model thinks.
# 4. Shows simple charts and numbers to explain why the model made its choice.
#
# For the Multinomial and Bernoulli models, the app looks at each word in your review and
# adds up how much each word pushes the decision toward positive or negative.
# If this total is very small (within a chosen limit), the review is labeled as Neutral.
#
# For the Gaussian model, a pie chart shows the percentage chance for each class.
#
# Use the slider to adjust the “Neutrality Threshold.” If the difference between the
# positive and negative scores is small enough, the review is marked as Neutral.
#**********************************************

import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk

# Download NLTK data needed for simplifying words
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =============================================================================
# Simple Text Cleaner Function
# =============================================================================
def custom_tokenizer(text):
    """
    Clean the text by:
    - Changing everything to lowercase.
    - Picking out words.
    - Removing common words and some extra words like "food" or "service".
    - Simplifying words to their basic form.
    Returns a list of clean words.
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
# Example Reviews for Training
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
# Train the Model (with caching)
# =============================================================================
@st.cache_resource
def train_model(nb_variant):
    """
    Teach the model using the example reviews.
    We use a tool that converts text into numbers.
    Based on your choice, we train one of three models.
    Returns the trained model and the tool that turned text into numbers.
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
# Word-Level Influence (for Multinomial and Bernoulli models)
# =============================================================================
def get_token_sentiments(tokens, model, vectoriser):
    """
    For the Multinomial and Bernoulli models, this function checks each word
    to see how much it pushes the review towards positive or negative.
    It returns a list of words with a simple score.
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
# Simple Bar Chart for Word Influence
# =============================================================================
def plot_token_sentiments(token_df):
    """
    Creates a bar chart that shows how each word in the review pushes the decision.
    Green means the word pushes the review toward positive.
    Red means it pushes toward negative.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    tokens = token_df["Token"]
    scores = token_df["Score"]
    colors = ['green' if score > 0 else 'red' if score < 0 else 'gray' for score in scores]
    ax.bar(tokens, scores, color=colors)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Word")
    ax.set_ylabel("Score")
    ax.set_title("How Each Word Affects the Sentiment")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

# =============================================================================
# Learn About Naive Bayes (Simple Explanation)
# =============================================================================
with st.expander("Learn About Naive Bayes"):
    st.markdown(r"""
    **What is Naive Bayes?**
    
    Naive Bayes is a simple way to decide if a review is good or bad.
    It looks at the words in the review and uses them to make a guess.
    
    **How does it work?**
    - It starts with a basic guess.
    - Then, it checks each word to see if it makes the review seem good or bad.
    - Finally, it adds everything up to decide if the review is positive or negative.
    """, unsafe_allow_html=True)

# =============================================================================
# Sidebar: Model Settings
# =============================================================================
st.sidebar.header("Model Settings")
nb_variant = st.sidebar.selectbox("Choose a Naive Bayes Model", options=["Multinomial", "Bernoulli", "Gaussian"])
st.sidebar.markdown("""
**Neutrality Threshold:**  
Use this slider to decide when a review is too balanced.
If the difference between the positive and negative scores is very small, the review will be marked as Neutral.
""")
neutral_threshold = st.sidebar.slider("Neutrality Threshold (small number)", 0.0, 1.0, 0.1, step=0.01)

# =============================================================================
# Main App Layout
# =============================================================================
st.title("Naive Bayes Demo App for Sentiment Analysis")
st.markdown("""
Type a restaurant review below and choose a model to see:
- The final decision (Positive, Negative, or Neutral).
- How each word in your review affected the decision.
- A simple number breakdown showing why the decision was made.
""")

# Train the model using the chosen variant
model, vectoriser = train_model(nb_variant)

# Input area for your review
st.subheader("Enter a Restaurant Review")
default_review = "delicious food but very slow"
user_review = st.text_area("Your Review:", default_review)

if st.button("Predict Sentiment"):
    # Clean the review text
    tokens = custom_tokenizer(user_review)
    if not tokens:
        st.subheader("Prediction")
        st.write("**Sentiment:** Neutral (Not enough words)")
        st.markdown("### Explanation")
        st.write("No useful words were found in your review. This might happen if all words were too common. In this case, the review is marked as Neutral.")
    else:
        X_new = vectoriser.transform([user_review])
        if nb_variant == "Gaussian":
            X_new = X_new.toarray()
        
        # Find the positions for positive and negative in the model
        classes = model.classes_.tolist()
        pos_index = classes.index("Positive")
        neg_index = classes.index("Negative")
        
        # Get the final decision based on the model type
        if nb_variant == "Gaussian":
            proba = model.predict_proba(X_new)[0]
            pos_prob = proba[pos_index]
            neg_prob = proba[neg_index]
            diff = pos_prob - neg_prob
            if abs(diff) < neutral_threshold:
                overall_sentiment = "Neutral"
            else:
                overall_sentiment = "Positive" if diff > 0 else "Negative"
        else:
            # For Multinomial and Bernoulli models, look at each word's influence.
            token_df = get_token_sentiments(tokens, model, vectoriser)
            log_prior_diff = model.class_log_prior_[pos_index] - model.class_log_prior_[neg_index]
            token_sum = token_df["Score"].sum() if not token_df.empty else 0
            overall_log_diff = log_prior_diff + token_sum
            
            # If the total effect is very small, mark as Neutral.
            if abs(overall_log_diff) <= neutral_threshold:
                overall_sentiment = "Neutral"
            else:
                overall_sentiment = "Positive" if overall_log_diff > 0 else "Negative"

        # Show the final prediction
        st.subheader("Prediction")
        st.write(f"**Sentiment:** {overall_sentiment}")
        
        # Simple explanation of the result
        st.markdown("### Explanation")
        if overall_sentiment == "Positive":
            st.write("The review is seen as **Positive** because the words mostly point to a good experience.")
        elif overall_sentiment == "Negative":
            st.write("The review is seen as **Negative** because the words mostly point to a bad experience.")
        else:
            st.write("The review is marked as **Neutral** because it is very balanced.")
        
        # Additional notes for each model type
        if nb_variant == "Bernoulli":
            st.write("**Note for Bernoulli NB:** Each word counts once, no matter how many times it appears.")
        elif nb_variant == "Multinomial":
            st.write("**Note for Multinomial NB:** Words that appear more often have a bigger effect.")
        
        # For Multinomial and Bernoulli models, show the word-by-word breakdown.
        if nb_variant in ["Multinomial", "Bernoulli"] and not token_df.empty:
            st.subheader("Word Influence")
            st.write("""
            Below is a list of words from your review and a simple score:
            - A positive score means the word pushes the review toward positive.
            - A negative score means the word pushes it toward negative.
            """)
            st.dataframe(token_df, hide_index=True)
            plot_token_sentiments(token_df)
            
            # Show the simple calculation behind the decision
            st.markdown("### Simple Calculation")
            st.write(f"**Starting bias:** {log_prior_diff:.4f}")
            st.write("**Word contributions:**")
            for _, row in token_df.iterrows():
                st.write(f"- {row['Token']}: {row['Score']:.4f}")
            st.write(f"**Total word effect:** {token_sum:.4f}")
            st.write(f"**Overall effect:** {overall_log_diff:.4f}")
            st.write(f"**Neutrality threshold:** ±{neutral_threshold:.4f}")
            
            if overall_sentiment == "Neutral":
                st.write("The total effect was very small, so the review is marked as Neutral.")
            else:
                direction = "above" if overall_log_diff > 0 else "below"
                st.write(f"The overall effect is {direction} the threshold, so the review is classified as {overall_sentiment}.")
        
        # For Gaussian NB, show a simple pie chart.
        elif nb_variant == "Gaussian":
            st.subheader("Probability Breakdown")
            fig, ax = plt.subplots()
            proba = model.predict_proba(X_new)[0]
            ax.pie(proba, labels=model.classes_, autopct='%1.1f%%', 
                   colors=['green' if c == "Positive" else 'red' for c in model.classes_])
            st.pyplot(fig)
            
            st.markdown("### Probability Details")
            st.write(f"Positive chance: {proba[pos_index]:.4f}")
            st.write(f"Negative chance: {proba[neg_index]:.4f}")
            st.write(f"Difference: {proba[pos_index] - proba[neg_index]:.4f}")
            st.write(f"Threshold: ±{neutral_threshold:.4f}")

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
