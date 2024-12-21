import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the model, vectorizer, and scaler
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
classifier = pickle.load(open('spam_classifier.pkl', 'rb'))

# Download stopwords
nltk.download('stopwords')

ps = PorterStemmer()

# Function to preprocess input text
def preprocess_text(text):
    # Clean the text
    email = re.sub('[^a-zA-Z]', ' ', text)
    email = email.lower()
    email = email.split()
    email = [ps.stem(word) for word in email if word not in stopwords.words('english')]
    email = ' '.join(email)
    return email

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>📧 Spam Email Classifier</h1>", unsafe_allow_html=True)
st.write("Enter the email content below to classify it as Spam or Not Spam.")

user_input = st.text_area("Email Content")

if st.button("Classify"):
    if user_input:
        # Preprocess the input text
        processed_text = preprocess_text(user_input)
        
        # Vectorize the input text
        vectorized_text = vectorizer.transform([processed_text])
        
        # Apply scaling
        vectorized_text = scaler.transform(vectorized_text)
        
        # Predict using the loaded model
        prediction = classifier.predict(vectorized_text)
        
        if prediction[0] == 1:
            st.error("🚫 The email is classified as **Spam**.")
        else:
            st.success("✅ The email is classified as **Not Spam**.")
    else:
        st.warning("Please enter email content to classify.")
