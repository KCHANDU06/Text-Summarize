import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import string
import PyPDF2
from docx import Document

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Custom CSS styling
st.markdown("""
    <style>
    .stApp {
        background-color: #FFF2F2;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FF4B4B;
    }
    .css-1cpxqw2 edgvbvh3 {  /* Adjust button text color if needed */
        color: white;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #e04444;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def lemmatize_words(words):
    return [lemmatizer.lemmatize(word) for word in words]

def summarize_text(text, summary_length='short'):
    cleaned_text = clean_text(text)
    sentences = sent_tokenize(text)

    if len(sentences) == 0:
        return "No valid sentences found to summarize."

    words = word_tokenize(cleaned_text)
    lemmatized_words = lemmatize_words(words)

    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in lemmatized_words if word.isalpha() and word not in stop_words]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()

    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        score = 0
        sentence_tokens = word_tokenize(clean_text(sentence))
        lemmatized_sentence = lemmatize_words(sentence_tokens)

        for word in lemmatized_sentence:
            if word in feature_names:
                word_index = feature_names.tolist().index(word)
                score += tfidf_matrix[i, word_index]

        sentence_scores[sentence] = score / len(sentence_tokens) if len(sentence_tokens) > 0 else 0

    if summary_length == 'short':
        num_sentences = 150
    elif summary_length == 'medium':
        num_sentences = 250
    elif summary_length == 'long':
        num_sentences = 500
    else:
        return "Invalid summary length. Choose 'short', 'medium', or 'long'."

    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

    if len(sorted_sentences) < num_sentences:
        num_sentences = len(sorted_sentences)

    if summary_length == 'short':
        summary = sorted_sentences[0]
    else:
        summary = ' '.join(sorted_sentences[:num_sentences])

    return summary

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def read_txt(file):
    text = file.read().decode("utf-8")
    return text

def read_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Streamlit App Layout
st.markdown("""
    <h1 style='text-align: center;'>📖 Text Summarizer App</h1>
    <p style='text-align: center; font-size:18px;'>Paste your text or upload a document — get a summary in seconds </p>
    <p style='text-align: center; font-size:18px;'>Upload a PDF, Word, TXT file or paste text — and get an summary </p>
    <hr style='border: 1px solid #FF4B4B;'>
""", unsafe_allow_html=True)

# Input method dropdown selector
input_option = st.selectbox(
    " Choose Input Type:",
    ("Select","Paste Text", "Upload PDF", "Upload TXT Document", "Upload Word Document (.docx)")
)

text_input = ""

if input_option == "Paste Text":
    text_input = st.text_area(" Enter Text to Summarize:", height=100)

elif input_option == "Upload PDF":
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_pdf is not None:
        text_input = read_pdf(uploaded_pdf)
        st.success("PDF text loaded successfully!")

elif input_option == "Upload TXT Document":
    uploaded_txt = st.file_uploader("Upload a TXT file", type=["txt"])
    if uploaded_txt is not None:
        text_input = read_txt(uploaded_txt)
        st.success("Text file loaded successfully!")

elif input_option == "Upload Word Document (.docx)":
    uploaded_docx = st.file_uploader(" Upload a Word document", type=["docx"])
    if uploaded_docx is not None:
        text_input = read_docx(uploaded_docx)
        st.success("Word document text loaded successfully!")

# Summary length option
summary_length_option = st.selectbox(
    " Select Summary Length:",
    ("short", "medium", "long")
)

# Summarize button
if st.button(" Summarize Now"):
    if text_input.strip() == "":
        st.warning("Please provide some text to summarize.")
    else:
        summary_result = summarize_text(text_input, summary_length_option)
        st.subheader(" Summary:")
        st.write(summary_result)


