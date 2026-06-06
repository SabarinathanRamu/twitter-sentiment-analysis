import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample Dataset
data = {
    'tweet': [
        'I love this product',
        'This is amazing',
        'Worst experience ever',
        'I hate this service',
        'Good quality',
        'Very bad product',
        'Excellent support',
        'Not satisfied',
        'Happy with purchase',
        'Terrible experience'
    ],
    'sentiment': [
        'positive',
        'positive',
        'negative',
        'negative',
        'positive',
        'negative',
        'positive',
        'negative',
        'positive',
        'negative'
    ]
}

df = pd.DataFrame(data)

# Feature Extraction
vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(df['tweet'])
y = df['sentiment']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

# User Input
new_tweet = ["I am very happy with this service"]

new_tweet_vector = vectorizer.transform(new_tweet)

result = model.predict(new_tweet_vector)

print("Sentiment:", result[0])
