import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Add TextBlob pipeline for sentiment
nlp.add_pipe("spacytextblob")

# Load the dataset
data = pd.read_csv("Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")

# Select the reviews column
reviews_data = data['reviews.text']

# Remove missing values
clean_data = data.dropna(subset=['reviews.text'])

# Extract cleaned reviews
reviews = clean_data['reviews.text']


# -------- TEXT PREPROCESSING FUNCTION --------# 
def preprocess_text(text):
    
    # Convert text to string and clean it
    text = str(text).lower().strip()

    doc = nlp(text)

    # Remove stop words and punctuation
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]

    cleaned_text = " ".join(tokens)

    return cleaned_text


# -------- SENTIMENT ANALYSIS FUNCTION --------# 
def sentiment_analysis(review):

    cleaned_review = preprocess_text(review)

    doc = nlp(cleaned_review)

    polarity = doc._.blob.polarity
    sentiment = doc._.blob.sentiment

    if polarity > 0:
        label = "Positive"
    elif polarity < 0:
        label = "Negative"
    else:
        label = "Neutral"

    return label, polarity


# -------- TEST THE MODEL --------# 
print("Sample Sentiment Predictions\n")

sample_reviews = reviews.head(5)

for review in sample_reviews:

    sentiment, polarity = sentiment_analysis(review)

    print("Review:", review)
    print("Sentiment:", sentiment)
    print("Polarity:", polarity)
    print("-"*60)


# -------- SIMILARITY EXAMPLE --------# 
review1 = reviews.iloc[0]
review2 = reviews.iloc[1]

doc1 = nlp(review1)
doc2 = nlp(review2)

similarity = doc1.similarity(doc2)

print("\nSimilarity between two reviews:", similarity)