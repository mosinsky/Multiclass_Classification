import gradio as gr
import joblib
import re
import spacy
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')
from autocorrect import Speller
spell = Speller(lang='en')
nlp = spacy.load("en_core_web_sm")

# Load the pre-trained models
model = joblib.load('pre-trained_models/logistic_regression.joblib')
cv = joblib.load('pre-trained_models/cv.joblib')
tfidf_t = joblib.load('pre-trained_models/tfidf.joblib') 


def correct_text(text):
    spell = Speller(lang='en')
    corrected_text = spell(text)
    
    return corrected_text


def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub('\S*\d\S*\s*','', text)
    text = re.sub('\[.*\]','', text)

    return text

def remove_stopwords(tokens):
    filtered_tokens = [word for word in tokens if word.lower() not in STOP_WORDS]

    return filtered_tokens

def tokenize_text(text):
    # Tokenize the text into words
    tokens = word_tokenize(text)

    return tokens


def preprocess_text(text):
    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text)
    tokens_without_stopwords = remove_stopwords(tokens)
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
        return "", "", "Please describe Your issue first."
    input_text = input_text[0] if isinstance(input_text, list) else input_text
    input_text = correct_text(input_text)
    preprocessed_text = preprocess_text(input_text)
    lemmatized_text = lemmatize(preprocessed_text)

    test = cv.transform([lemmatized_text])
    test_tfidf = tfidf_t.transform(test)

    prediction = model.predict(test_tfidf)[0]

    return input_text, lemmatized_text, prediction

title = "Klasyfikacja tekstu"
desc = 'Zastosowanie technik uczenia maszynowego do klasyfikacji tekstu'
long_desc = "Michał Mosnski 2023-2024"

iface = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(placeholder="Enter text...", label="Complaint"),
    outputs=[
        gr.Textbox(label="Orginal text"),
        gr.Textbox(label="cleaned text"),
        gr.Textbox(label="Prediction")],
        title=title, 
        description = desc,
        article=long_desc,
        # theme=gr.themes.Soft()

)


# Launch app
if __name__ == "__main__":
    iface.launch(share=True)
