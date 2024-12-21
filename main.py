import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the dataset
data = pd.read_csv('email_classification.csv')

# Verify the column names
print("Columns in the DataFrame:", data.columns.tolist())

# Map 'ham' to 0 and 'spam' to 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Handle missing values (if any)
data.dropna(inplace=True)

# Download NLTK data (run this once)
nltk.download('stopwords')

ps = PorterStemmer()
corpus = []

for email in data['email']:
    # Remove non-letter characters
    email = re.sub('[^a-zA-Z]', ' ', email)
    email = email.lower()
    email = email.split()
    
    # Remove stopwords and stem words
    email = [ps.stem(word) for word in email if word not in stopwords.words('english')]
    email = ' '.join(email)
    corpus.append(email)

# Feature extraction using TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(corpus)

# Target variable
y = data['label'].values

# Split the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler

# Since TF-IDF produces sparse matrices, use with_mean=False
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model using Logistic Regression
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(max_iter=1000, random_state=0)
classifier.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = classifier.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model, vectorizer, and scaler
import pickle

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('spam_classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)
