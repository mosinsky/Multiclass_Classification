import gradio as gr
import joblib
import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.tokenize import word_tokenize

# Load the pre-trained model and preprocessing components
model = joblib.load('pre-trained_models/logistic_regression_model.joblib')
cv = joblib.load('pre-trained_models/count_vectorizer_model.joblib')
tfidf_t = joblib.load('pre-trained_models/tfidf_transformer_model.joblib') 

# Load spaCy for lemmatization
nlp = spacy.load("en_core_web_sm")


# Define your text cleaning function
def clean_text(text):
    if text is None:
        return ""
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub('\S*\d\S*\s*','', text)
    text = re.sub('\[.*\]','', text)
    text = re.sub('xxxx','', text)
    text = re.sub('-PRON-', '', text)

    return text

def remove_stopwords(tokens):
    # Remove common English stop words
    # stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in STOP_WORDS]

    return filtered_tokens

def tokenize_text(text):
    # Tokenize the text into words
    tokens = word_tokenize(text)

    return tokens

# Define additional preprocessing functions
def preprocess_text(text):
    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text)  # Assuming you have a tokenize_text function
    tokens_without_stopwords = remove_stopwords(tokens)  # Assuming you have a remove_stopwords function
    preprocessed_text = ' '.join(tokens_without_stopwords)
    return preprocessed_text

def lemmatize(text):
    doc = nlp(text)
    sent = [token.lemma_ for token in doc if token.text not in STOP_WORDS]
    return ' '.join(sent)

def remove_pos_tags(text):
    doc = nlp(text)
    sent = [token.text for token in doc if token.tag_ == 'NN']
    return ' '.join(sent)



# Combine preprocessing and classification
def classify_text(input_text):
    if not input_text:
        return "", "", "Input text is empty. Please enter some text."
    preprocessed_text = preprocess_text(input_text)  # Use the preprocessing steps you need
    lemmatized_text = lemmatize(preprocessed_text)  # You can choose to include or exclude lemmatization
    # text_without_pos_tags = remove_pos_tags(lemmatized_text)  # You can choose to include or exclude removing POS tags


    # Vectorize and transform the cleaned text
    test = cv.transform([preprocessed_text])  # Use the processed text
    test_tfidf = tfidf_t.transform(test)

    # Make prediction using the loaded model
    prediction = model.predict(test_tfidf)[0]

    return preprocessed_text, prediction

# Create a Gradio interface with an API endpoint
iface = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(placeholder="Enter text...", label="Complaint"),
    outputs=[
        gr.Textbox(placeholder="text_without_pos_tags", label="cleaned text"),
        gr.Textbox(placeholder="Prediction", label="Prediction")]
)

# Launch the Gradio interface
iface.launch()

# I tried to make a transaction at a supermarket retail store, using my chase debit/atm card, but the transaction was declined. I am still able to withdraw money out of an ATM machine using the same debit card. Please resolve this issue.
            #  gr.Textbox(placeholder="Translated Text", label = "Preprocessed_text"),

