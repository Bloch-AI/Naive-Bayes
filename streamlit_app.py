#**********************************************
# Naive Bayes Demo App
# Version 1.0
# 9th February 2025
# Jamie Crossman-Smith
# jamie@bloch.ai
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
# 4. Shows charts and simple numbers to explain why the decision was made.
#
# In simple terms, Naive Bayes uses the words in your review to guess the sentiment.
# Even though it makes a "naive" assumption, that each word acts on its own, it works very well.
#
# For example, Gmail’s spam filter used Naive Bayes because its simple approach made it fast
# and effective, even when dealing with millions of messages.
#
# The three common versions are:
# - Multinomial NB: Looks at how often words appear.
# - Bernoulli NB: Checks whether words appear or not (like a checklist).
# - Gaussian NB: Works with numbers and measurements.
#
# A slider lets you adjust the "Neutrality Threshold." If the difference between the
# positive and negative scores is very small (or exactly at the threshold), the review is marked as Neutral.
#**********************************************

import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk


@st.cache_resource
def ensure_nltk_data():
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    return True


ensure_nltk_data()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =============================================================================
# Simple Text Cleaner Function
# =============================================================================
def custom_tokenizer(text):
    """
    Clean the text by:
    - Changing everything to lowercase.
    - Picking out words (including short phrases such as "not delicious" once
      bigrams are enabled downstream).
    - Removing common words and extra words like "food" or "service", but
      KEEPING negators and intensifiers ("not", "no", "very", ...) because they
      carry sentiment.
    - Simplifying words to their basic form.
    Returns a list of clean words.
    """
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    try:
        nltk_stopwords = set(stopwords.words('english'))
    except LookupError:
        nltk_stopwords = set()
    # Keep negators and intensifiers: they carry sentiment (e.g. "not delicious",
    # "very good"). Removing them would flip the meaning of a review.
    sentiment_keep = {"not", "no", "nor", "very", "too", "so", "never", "without", "n't"}
    nltk_stopwords = nltk_stopwords - sentiment_keep
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
    We convert the text into numbers and then train one of three models.
    Returns the trained model, the tool (vectoriser) that converts text to numbers,
    and the cross-validated accuracy measured on the training set.

    Implementation notes (methodology):
    - Bernoulli NB expects BINARY presence/absence features, so we build a binary
      vectoriser for that variant. Feeding it TF-IDF weights is a misuse that makes
      the "checklist" metaphor inaccurate.
    - Multinomial NB uses TF-IDF weights, which re-weight words by how rare they are
      across reviews (inverse document frequency). This is standard for text
      classification but is NOT plain word counting; the UI explains the difference.
    - Bigrams (ngram_range=(1,2)) are added so negation phrases like "not delicious"
      are seen as a single feature instead of being split into two unrelated words.
      This is the biggest lever for sentiment accuracy.
    - Gaussian NB assumes normally-distributed numeric features; TF-IDF values are
      sparse and skewed, so this is an *illustrative* misuse included for comparison,
      not a recommended configuration.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import cross_val_score
    df = get_training_data()
    binary = (nb_variant == "Bernoulli")
    vectoriser = TfidfVectorizer(
        tokenizer=custom_tokenizer,
        lowercase=False,
        ngram_range=(1, 2),
        binary=binary,
        token_pattern=None,
    )
    X = vectoriser.fit_transform(df['review'])
    # to_numpy(dtype=object) gives a plain numpy array of strings; .values can return a
    # pyarrow-backed array that some sklearn helpers (e.g. cross_val_score joblib
    # indexing) cannot slice, which would silently fail and hide the accuracy metric.
    y = df['sentiment'].to_numpy(dtype=object)
    if nb_variant == "Multinomial":
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB()
        model.fit(X, y)
        eval_X = X
    elif nb_variant == "Bernoulli":
        from sklearn.naive_bayes import BernoulliNB
        model = BernoulliNB()
        model.fit(X, y)
        eval_X = X
    elif nb_variant == "Gaussian":
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        model.fit(X.toarray(), y)
        eval_X = X.toarray()
    else:
        st.error("Unsupported Naive Bayes variant selected.")
        return None, None, None
    try:
        accuracy = float(cross_val_score(model, eval_X, y, cv=4, scoring='accuracy').mean())
    except Exception:
        accuracy = None
    return model, vectoriser, accuracy

# =============================================================================
# Word-Level Influence (for Multinomial and Bernoulli models)
# =============================================================================
def get_token_sentiments(review_text, model, vectoriser):
    """
    For Multinomial and Bernoulli models, this function works out how much each
    feature (a word or a bigram such as "not delicious") pushes the review toward
    positive or negative, and returns the per-feature contribution table along
    with the numbers needed to reconcile that table with the model's real decision.

    Returns a dict with:
      - df: per-feature table (Feature, Value, PerUnitScore, Score) for PRESENT features.
      - log_prior_diff: class prior difference (log P(Positive) - log P(Negative)).
      - present_sum: sum of present features' Score contributions.
      - absent_diff: extra contribution from ABSENT features (Bernoulli only; for
        Bernoulli, the absence of an expected word also shifts the score). 0 for Multinomial.
      - total_diff: the AUTHORITATIVE decision difference taken straight from the
        model's own joint log-likelihood, so the displayed math always reconciles:
        total_diff == log_prior_diff + present_sum + absent_diff.

    The contribution of a present feature is its log-probability difference
    (log P(feature|Positive) - log P(feature|Negative)) MULTIPLIED BY the actual
    feature value the model uses for this review (TF-IDF weight for Multinomial,
    1 for binary Bernoulli). This makes the displayed per-feature math identical to
    the math the model really performs, so learners see the true decision breakdown
    rather than an approximation. Without this weighting, a word repeated three
    times would be shown as contributing only once, which is misleading.
    """
    import numpy as np
    empty_result = {
        "df": pd.DataFrame(columns=["Feature", "Value", "PerUnitScore", "Score"]),
        "log_prior_diff": 0.0,
        "present_sum": 0.0,
        "absent_diff": 0.0,
        "total_diff": 0.0,
    }
    classes = model.classes_
    if not ("Positive" in classes and "Negative" in classes):
        return empty_result
    pos_index = list(classes).index("Positive")
    neg_index = list(classes).index("Negative")
    row = vectoriser.transform([review_text])
    feature_names = vectoriser.get_feature_names_out()
    row_values = row.toarray()[0]
    lp_pos = model.feature_log_prob_[pos_index]
    lp_neg = model.feature_log_prob_[neg_index]
    log_prior_diff = float(model.class_log_prior_[pos_index] - model.class_log_prior_[neg_index])

    # Per-feature contribution for PRESENT features.
    present_cols = row_values.nonzero()[0]
    contributions = []
    for col_index in present_cols:
        value = row_values[col_index]
        per_unit = lp_pos[col_index] - lp_neg[col_index]
        contributions.append({"Feature": feature_names[col_index], "Value": value, "PerUnitScore": per_unit, "Score": per_unit * value})
    df = pd.DataFrame(contributions)
    if not df.empty:
        df = df.sort_values("Score")
    present_sum = float(df["Score"].sum()) if not df.empty else 0.0

    # For Bernoulli, ABSENT features also contribute: each absent feature adds
    # log(1 - P(feature|y)) for each class, and the difference over all absent
    # features shifts the decision. This is why a checklist model cares about
    # what is NOT said, not just what is said.
    is_bernoulli = type(model).__name__ == "BernoulliNB"
    absent_diff = 0.0
    if is_bernoulli:
        with np.errstate(divide="ignore"):
            log1mp_pos = np.log1p(-np.exp(lp_pos))
            log1mp_neg = np.log1p(-np.exp(lp_neg))
        per_feat_diff = row_values * (lp_pos - lp_neg) + (1.0 - row_values) * (log1mp_pos - log1mp_neg)
        absent_diff = float(per_feat_diff.sum()) - present_sum

    # Authoritative decision difference from the model itself.
    try:
        jll = model._joint_log_likelihood(row)[0]
        total_diff = float(jll[pos_index] - jll[neg_index])
    except Exception:
        total_diff = log_prior_diff + present_sum + absent_diff

    return {"df": df, "log_prior_diff": log_prior_diff, "present_sum": present_sum, "absent_diff": absent_diff, "total_diff": total_diff}
# =============================================================================
def plot_token_sentiments(token_df):
    """
    Creates a bar chart that shows how each word in the review pushes the decision.
    Green means the word pushes the review toward positive.
    Red means it pushes toward negative.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    features = token_df["Feature"]
    scores = token_df["Score"]
    colors = ['green' if score > 0 else 'red' if score < 0 else 'gray' for score in scores]
    ax.bar(features, scores, color=colors)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Feature (word or bigram)")
    ax.set_ylabel("Contribution to decision")
    ax.set_title("How Each Feature Affects the Sentiment")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

# =============================================================================
# Learn About Naive Bayes (Plain Language Explanation)
# =============================================================================
with st.expander("Learn About Naive Bayes"):
    st.markdown(r"""
    **Naive Bayes in Simple Terms:**
    
    - **Simple but Powerful:**  
      Naive Bayes is a simple algorithm that still works very well. Instead of drawing complicated boundaries between classes, it
      simply calculates which outcome is most likely based on the words in a review.
      
    - **Real-World Success:**  
      For example, Google Gmails early spam filter used Naive Bayes. Its simple approach made it fast and effective, even with millions
      of emails.
      
    - **How It Works:**  
      1. **Learning:** The model learns from examples by counting how often each word appears in good and bad reviews, then turns
         those counts into probabilities.
      2. **Predicting:** When a new review comes in, it checks the words and combines the probabilities to guess if the review is good or bad.

      
    - **The 'Naive' Part:**  
      The model assumes each word works independently. In real language, words work together (like “not delicious”), so the model
      can be fooled. To reduce this, this app keeps small negators such as **"not"** and also creates **bigrams** (two-word phrases like
      "not delicious") as extra features, so the model can learn that the phrase itself points negative.
      
    - **Different Versions:**  
      - **Multinomial NB:** Uses weighted word counts (here, TF-IDF weights rather than raw counts, which down-weights very common words).
      - **Bernoulli NB:** Works like a checklist: it only cares if a word is there or not, so it uses binary 0/1 features here.
      - **Gaussian NB:** Assumes features are numbers that follow a bell-curve (normal) distribution. Text features don't really do that,
        so this variant is included for comparison/learning and is not the recommended choice for text.
      
    This simplicity is why Naive Bayes is used in many applications, from spam filtering to analysing customer reviews.
    """, unsafe_allow_html=True)

# =============================================================================
# Sidebar: Model Settings
# =============================================================================
st.sidebar.header("Model Settings")
nb_variant = st.sidebar.selectbox(
    "Choose a Naive Bayes Model",
    options=["Multinomial", "Bernoulli", "Gaussian"],
    help="Pick how the model turns words into a decision. Multinomial uses weighted word counts, Bernoulli uses a yes/no checklist, and Gaussian treats the numbers as measurements. For text, Multinomial or Bernoulli are the usual choices.",
)
st.sidebar.markdown("""
**Model Options Explained:**

- **Multinomial NB:**  
  Uses weighted word counts (TF-IDF weights, which down-weight very common words). Words that appear more often still have a bigger impact.

- **Bernoulli NB:**  
  Checks whether a word is present or not, like ticking off items on a checklist. It uses binary 0/1 features.

- **Gaussian NB:**  
  Works with continuous numbers and assumes they follow a bell-curve (normal) distribution. Text features don't really
  follow that distribution, so this variant is included for comparison/learning and is not the recommended choice for text.
""")

# =============================================================================
# Main App Layout
# =============================================================================
st.title("Naive Bayes Demo App for Sentiment Analysis")
st.markdown("""
Type a restaurant review below and choose a model to see:
- The final decision: Positive, Negative, or Neutral.
- How each word influenced the decision.
- A simple breakdown of the numbers behind the decision.
  
**Understanding the Process:**  
Imagine you’re a food critic. You know that words like "delicious" or "fantastic" are common in good reviews,
while words like "bland" or "disappointing" appear in bad reviews. The model uses this idea to decide the sentiment.
""")

# Train the model using the chosen variant (cached)
model, vectoriser, cv_accuracy = train_model(nb_variant)

# Show measured accuracy so users see how reliable the model really is.
# Cross-validated accuracy is estimated on the small training set, so treat it as
# a rough guide, not a production metric.
if cv_accuracy is not None:
    st.sidebar.markdown(f"""
**Measured Accuracy (cross-validation):**  
{cv_accuracy*100:.1f}%  
_Estimated from the small built-in training set (40 reviews), so treat this as a rough guide, not a guarantee._
""")
else:
    st.sidebar.markdown("""
**Measured Accuracy:**  
_Not available for this model._
""")

# Neutrality threshold. The units differ by model: for Multinomial/Bernoulli the
# decision variable is a log-probability difference (can be larger than 1), while
# for Gaussian it is a probability difference (always between 0 and 1). We label
# the units so the slider means the same kind of "smallness" to the learner.
if nb_variant == "Gaussian":
    threshold_units = "probability difference (0 to 1)"
    threshold_max = 0.5
    threshold_default = 0.05
else:
    threshold_units = "log-probability difference (can be larger than 1)"
    threshold_max = 5.0
    threshold_default = 0.5
st.sidebar.markdown(f"""
**Neutrality Threshold:**  
Use this slider to decide when a review is too balanced to call.  
If the difference between the positive and negative scores is very small (within ±
this threshold), the review is marked as Neutral.  
_Units: {threshold_units}_
""")
neutral_threshold = st.sidebar.slider(
    "Neutrality Threshold",
    0.0, threshold_max, threshold_default, step=0.01,
    help=f"If the gap between the positive and negative scores is within +/- this value, the review is called Neutral instead of being forced one way. Units: {threshold_units}. A bigger threshold means more reviews come out as Neutral.",
)

# Input area for your review
st.subheader("Enter a Restaurant Review")
default_review = "delicious food but very slow"

# Keep the review text in session state so the example buttons below can
# pre-fill it for learners who want to experiment quickly.
if "user_review" not in st.session_state:
    st.session_state.user_review = default_review

st.caption("Try one of these examples to get started, then edit the text or write your own.")
example_reviews = {
    "Clearly positive": "Absolutely delicious food and wonderful, friendly service.",
    "Clearly negative": "Cold, bland food and very slow, rude service.",
    "Negation (tricky)": "The food was not delicious and the service was slow.",
    "Mixed": "delicious food but very slow",
}
example_cols = st.columns(len(example_reviews))
for col, (label, review) in zip(example_cols, example_reviews.items()):
    if col.button(label, help=f"Fill the box with: {review}"):
        st.session_state.user_review = review
        st.rerun()

user_review = st.text_area(
    "Your Review:",
    key="user_review",
    help="Write a restaurant review in plain English. The model reads the words here (and pairs of words, such as 'not delicious') to decide the sentiment.",
)

if st.button("Predict Sentiment", help="Work out the sentiment of the review above and show why the model reached that decision."):
    # Clean the review text
    tokens = custom_tokenizer(user_review)
    if not tokens:
        st.subheader("Prediction")
        st.write("**Sentiment:** Neutral (Not enough useful words)")
        st.markdown("### Explanation")
        st.write("No important words were found in your review. When that happens, the review is marked as Neutral.")
    else:
        X_new = vectoriser.transform([user_review])
        if nb_variant == "Gaussian":
            X_new = X_new.toarray()
        
        # Find the positions for positive and negative in the model
        classes = model.classes_.tolist()
        pos_index = classes.index("Positive")
        neg_index = classes.index("Negative")
        
        # Per-feature breakdown (Multinomial/Bernoulli only). Initialise early so the
        # variable always exists regardless of which branch runs below.
        token_df = pd.DataFrame(columns=["Feature", "Value", "PerUnitScore", "Score"])

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
            # For Multinomial and Bernoulli models, look at each feature's influence.
            # The decision difference is taken straight from the model's own joint
            # log-likelihood, so the displayed arithmetic reconciles exactly:
            # total = log_prior + present_features + absent_features (Bernoulli).
            result = get_token_sentiments(user_review, model, vectoriser)
            token_df = result["df"]
            log_prior_diff = result["log_prior_diff"]
            present_sum = result["present_sum"]
            absent_diff = result["absent_diff"]
            overall_log_diff = result["total_diff"]
            
            # If the total effect is very small, mark as Neutral.
            if abs(overall_log_diff) <= neutral_threshold:
                overall_sentiment = "Neutral"
            else:
                overall_sentiment = "Positive" if overall_log_diff > 0 else "Negative"

        # Show the final prediction
        st.subheader("Prediction")
        st.write(f"**Sentiment:** {overall_sentiment}")
        
        # Result-specific explanation that names the words which actually drove
        # this decision, rather than a fixed sentence. This closes the loop for a
        # learner between the bar chart/table and the verdict.
        st.markdown("### Explanation")
        if nb_variant == "Gaussian":
            if overall_sentiment == "Neutral":
                st.write(f"The positive and negative chances are very close (difference {abs(pos_prob - neg_prob):.4f}), within the threshold, so the review is marked as **Neutral**.")
            else:
                stronger = "positive" if pos_prob > neg_prob else "negative"
                st.write(f"The model gives a {stronger} chance of {max(pos_prob, neg_prob):.1%} against {min(pos_prob, neg_prob):.1%} the other way, so the review is seen as **{overall_sentiment}**.")
            st.caption("Gaussian NB treats the numbers as measurements on a bell curve. For text this is only a rough comparison, not the recommended choice.")
        else:
            # Build a short phrase naming the features that pushed hardest.
            if not token_df.empty:
                top_pos = token_df[token_df["Score"] > 0].sort_values("Score", ascending=False)
                top_neg = token_df[token_df["Score"] < 0].sort_values("Score")
                top_pos_words = ", ".join(f"'{r['Feature']}' (+{r['Score']:.2f})" for _, r in top_pos.head(2).iterrows())
                top_neg_words = ", ".join(f"'{r['Feature']}' ({r['Score']:.2f})" for _, r in top_neg.head(2).iterrows())
                if overall_sentiment == "Neutral":
                    st.write("The review is marked as **Neutral** because the positive and negative pulls are very close (within the threshold), so the evidence is balanced.")
                elif overall_sentiment == "Positive":
                    st.write(f"The review is seen as **Positive** because the positive words outweigh the negative ones. Strongest pushes: {top_pos_words or 'none'}. Pushing the other way: {top_neg_words or 'none'}.")
                else:
                    st.write(f"The review is seen as **Negative** because the negative words outweigh the positive ones. Strongest pushes: {top_neg_words or 'none'}. Pushing the other way: {top_pos_words or 'none'}.")
            else:
                if overall_sentiment == "Neutral":
                    st.write("The review is marked as **Neutral** because the evidence is very balanced.")
                else:
                    st.write(f"The review is seen as **{overall_sentiment}** based on the overall score.")
        
        # Additional notes for each model type
        if nb_variant == "Bernoulli":
            st.write("**Note for Bernoulli NB:** Each word counts just once, regardless of how many times it appears.")
        elif nb_variant == "Multinomial":
            st.write("**Note for Multinomial NB:** Words that appear more often have a bigger impact.")
        
        # For Multinomial and Bernoulli models, show the feature-by-feature breakdown.
        if nb_variant in ["Multinomial", "Bernoulli"] and not token_df.empty:
            st.subheader("Feature Influence")
            st.write("""
            Here is a list of features (words or two-word phrases) from your review and how much each pushes the decision.
            - **PerUnitScore:** how strongly the feature leans positive vs negative (log P(feature|Positive) - log P(feature|Negative)).
            - **Value:** the weight the model actually uses for this review (TF-IDF weight, or 1 for binary Bernoulli).
            - **Score:** PerUnitScore × Value, the real contribution to the decision. Green = positive push, red = negative push.
            """)
            st.dataframe(token_df, hide_index=True, width="stretch")
            plot_token_sentiments(token_df)
            
            # Show the simple calculation behind the decision
            st.markdown("### Simple Calculation")
            st.caption("How to read this: the model adds up a starting bias plus each feature's contribution. The total is the 'overall effect'. If the total is above 0 it leans positive, below 0 it leans negative. If it is within +/- the threshold, it is too close to call and becomes Neutral. The further the total is from 0, the more confident the decision.")
            st.write(f"**Starting bias (base preference):** {log_prior_diff:.4f}")
            st.write("**Feature contributions (PerUnitScore × Value):**")
            for _, row in token_df.iterrows():
                st.write(f"- {row['Feature']}: {row['PerUnitScore']:.4f} × {row['Value']:.4f} = {row['Score']:.4f}")
            st.write(f"**Total present-feature effect:** {present_sum:.4f}")
            if nb_variant == "Bernoulli":
                st.write(f"**Absent-feature effect (words NOT in your review):** {absent_diff:.4f}")
                st.write("_Bernoulli is a checklist model: the absence of an expected word also shifts the score, so this term is why the present features alone don't add up to the total._")
            st.write(f"**Overall effect (bias + present + absent):** {overall_log_diff:.4f}")
            st.write(f"**Neutrality threshold:** ±{neutral_threshold:.4f}")
            
            if overall_sentiment == "Neutral":
                st.write("The overall effect is within ± the threshold, so the review is marked as Neutral.")
            else:
                direction = "above" if overall_log_diff > 0 else "below"
                st.write(f"The overall effect is {direction} the threshold, so the review is classified as {overall_sentiment}.")
        
        # For Gaussian NB, show a simple pie chart (reuse the probability we already computed).
        elif nb_variant == "Gaussian":
            st.subheader("Probability Breakdown")
            st.caption("Gaussian NB works with probabilities (0 to 1) rather than the log-score used by the other models, so its numbers and threshold are on a different scale. Use this view to compare, not to match the figures above.")
            fig, ax = plt.subplots()
            ax.pie(proba, labels=model.classes_, autopct='%1.1f%%', 
                   colors=['green' if c == "Positive" else 'red' for c in model.classes_])
            st.pyplot(fig)
            
            st.markdown("### Probability Details")
            st.write(f"Positive chance: {proba[pos_index]:.4f}")
            st.write(f"Negative chance: {proba[neg_index]:.4f}")
            st.write(f"Difference: {proba[pos_index] - proba[neg_index]:.4f}")
            st.write(f"Threshold: ±{neutral_threshold:.4f}")

        # ================================
        # Article Link Section (at the bottom)
        # ================================
        st.markdown("---")
        st.markdown("### Further Reading")
        st.markdown(
                "For a detailed discussion on Naive Bayes, check out my [Medium article](https://blochai.medium.com/the-paradox-of-naive-bayes-when-simple-becomes-sophisticated-5b86acb25696)."
        )

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

