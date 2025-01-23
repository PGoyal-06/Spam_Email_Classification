# Spam Email Classifier
A machine learning project designed to classify emails as **Spam** or **Not Spam** (Ham) using Python, Scikit-learn, and Streamlit. This project includes:

Data preprocessing with NLTK and TF-IDF
Feature scaling using StandardScaler
Model training (Logistic Regression)
Interactive web application powered by Streamlit

## Project Overview
The **Spam Email Classifier** is a natural language processing (NLP) project that categorizes emails into two classes: **Spam or Ham** (not spam). By applying TF-IDF for feature extraction and Logistic Regression for classification, this project aims to demonstrate the end-to-end workflow of an NLP model, from data preprocessing to deployment in a user-friendly interface with Streamlit.

Key highlights:

**Data Preprocessing** using NLTK (tokenization, stopword removal, stemming)
**TF-IDF Vectorization** to convert text into numerical vectors
**Feature Scaling** (StandardScaler) to improve model performance
**Logistic Regression** as the primary classification model
**Streamlit Web App** for easy interaction and classification of custom email text

## Project Structure
```
Spam_Email_Classification/
├── email_classiciation.csv      # The dataset containing the emails and labels
├── main.py                      # Main script for data loading, preprocessing, and model training
├── app.py                       # Streamlit application
├── vectorizer.pkl               # Saved TF-IDF vectorizer
├── scaler.pkl                   # Saved StandardScaler
├── spam_classifier.pkl          # Saved classification model
├── README.md                    # Project documentation (this file)
└── requirements.txt             # Project dependencies
```
## Dataset
This project uses a dataset of synthetic emails labeled either 'ham' or 'spam':

**Columns**:
**email**: The textual content of the email.
**label**: Indicates whether the email is 'ham' (not spam) or 'spam'.
Example rows:
```
email,label
"Your account has been locked. Click here to verify your account information.",spam
"Happy holidays from our team! Wishing you joy and prosperity this season.",ham
```
## Installation and Setup
1. **Clone the Repository**
   ```
   git clone https://github.com/PGoyal-06/Spam_Email_Classification.git
   cd Spam_Email_Classification
   ```
2. **Create and Activate a Virtual Environment (Optional but Recommended)**
   ```
   python3 -m venv venv
   source venv/bin/activate     # On macOS/Linux
   venv\Scripts\activate        # On Windows
   ```
3. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```
4. **Download NLTK Stopwords**
   ```
   import nltk
   nltk.download('stopwords')
   ```
## How to Run the Project
1. **Train and Evaluate the Model**
   * Run main.py (or whichever file contains your training code).
   * This will train the model and save the necessary .pkl files.
2. **Launch the Streamlit App**
   ```
   streamlit run app.py
   ```
## Usage
1. **User Input**: Type or paste an email body into the text area in the Streamlit app.
2. **Prediction**: The classifier outputs a label:
   * **Spam** (if the probability of spam is high)
   * **Not Spam** (if the email is likely legitimate)
Feel free to experiment by adding or removing spam-like words to see how the classifier responds.

## Sample Email Contents for Testing
* Ham:
```
"Hi Team, just a reminder for our meeting tomorrow at 10 AM. Please be on time."
```
* Spam:
```
"Congratulations! You've won a free holiday to the Bahamas. Click here to claim your prize."
```
## Future Improvements
1. **Experiment with Different Models**:
   * RandomForestClassifier, SVC, or even Deep Learning methods for potentially better performance.
2. **Hyperparameter Tuning**:
   * Use Grid Search or Randomized Search to optimize the model’s parameters.
3. **Advanced Preprocessing**:
   * Implement lemmatization or advanced text cleaning.
   * Use more sophisticated methods like word embeddings (Word2Vec, GloVe).
4. **Deployment**:
   * Consider deploying the model on cloud platforms (e.g., Heroku, AWS, or Streamlit Community Cloud) for public access.
5. **Logging & Monitoring**:
   * Implement logging (e.g., using Python’s logging module) to track predictions and monitor performance in real-time.
## Thank you for checking out the Spam Email Classifier!
Feel free to open an issue or submit a pull request if you have any questions or issues. Contributions are always welcome!
