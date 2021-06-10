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


def one():
    api_key = "d7fe52539c5e477899a44c13669e2d92"
    main_url = "https://newsapi.org/v2/top-headlines?country=in&apiKey="+api_key
    news = requests.get(main_url).json()
    art = news["articles"]
    art = pd.DataFrame(art)
    article = news["articles"]
    article = pd.DataFrame(article)
    article = article.drop("source", axis=1)
    article = article.drop("url", axis=1)
    article = article.drop("urlToImage", axis=1)
    article = article.drop("content", axis=1)
    article = article.drop("publishedAt", axis=1)
    return article, art


def two(df):

    df['description'].fillna("", inplace=True)
    df['author'] = df['author'].fillna('unknown')
    df['title_size'] = df['title'].apply(lambda x: len(str(x)))

    # If title size == 3("NAN") then change title size = 0
    train_nan_index = df[df['title_size'] == 3].index
    df['title_size'][train_nan_index] = 0
    train_avg_title_size = int(df.value_counts(['title_size']).mean())
    df['title'][train_nan_index] = df['description'][train_nan_index].apply(lambda x: str(x)[:train_avg_title_size])
    df["description"] = df["title"] + " " + df["description"]
    df['description'] = df['author'] + " " + df['description']
    df.drop(['title', 'author'], axis=1, inplace=True)
    return df


def preprocess(text, stem=False):
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
    df['description'] = df['description'].apply(lambda x: preprocess(x))
    x_train = df['description']

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)

    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1

    x_train = pad_sequences(
        tokenizer.texts_to_sequences(x_train),
        maxlen=256)

    return x_train


def function():
    new, art = one()
    df = two(new)
    df['description'] = df['description'].apply(lambda x: preprocess(x))
    thr = three(df)
    new_model = keras.models.load_model("D:/mod_temp/mod 2")
    preds = new_model.predict(thr)
    art["How-Real-Is_This"] = preds
    # filtered = art[art['preds'] >=0.5]
    return art


app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST', "GET"])
def true_news():
    if request.method == "POST":
        output = function()
        output = pd.DataFrame(output)
        output= output.drop('urlToImage', axis=1)
        output = output.drop('url', axis=1)
        return render_template("output.html", data=output)


if __name__ == "__main__":
    app.run(debug=True)
