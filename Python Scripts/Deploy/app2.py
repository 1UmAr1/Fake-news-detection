import pandas as pd
import requests
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import tensorflow.keras as keras
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)


def process(text):
    text = text.lower()  # lowercase

    text = re.sub(r'[!]+', '!', text)
    text = re.sub(r'[?]+', '?', text)
    text = re.sub(r'[.]+', '.', text)
    text = re.sub(r"'", "", text)
    text = re.sub('\s+', ' ', text).strip()  # Remove and double spaces
    text = re.sub(r'&amp;?', r'and', text)  # replace & -> and
    text = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", text)  # Remove URLs
    # remove some puncts (except . ! # ?)
    text = re.sub(r'[:"$%&\*+,-/:;<=>@\\^_`{|}~]+', '', text)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'EMOJI', text)

    tokens = []
    for token in text.split():
        if token not in stop_words:
            tokens.append(lemmatizer.lemmatize(token))
    return " ".join(tokens)


def three(df):
    x_train = process(df)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)

    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1

    x_train = pad_sequences(
        tokenizer.texts_to_sequences(x_train),
        maxlen=256)

    return x_train


def function(message):
    x_train = three(message)
    new_model = keras.models.load_model("D:/mod_temp/mod 2")
    preds = new_model.predict(x_train)
    return preds


@app.route('/')
def home():
    return render_template("index2.html")


@app.route('/predict', methods=['POST', "GET"])
def predict():
    if request.method == "POST":
        message = request.form['message']
        output = function(message=message)
        print(output)

        return render_template("output2.html", prediction=output)


if __name__ == "__main__":
    app.run(debug=True)
