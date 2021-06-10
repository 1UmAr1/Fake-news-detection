import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from transformers import TFBertModel, BertTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout
from tensorflow.keras.layers import SpatialDropout1D, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import re
import keras
import time
from tensorflow.keras.callbacks import TensorBoard
Name = "Fakenews{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="BERT_FAKE2/".format(Name))

df = pd.read_csv(r"C:\Users\Um Ar\R Projects\UN MINOR\Datasets\Nai\train.csv")

print(df.head())
print(df.columns)

df[df['text'].isnull()].head()

# df["author"]

#df = df[df['author'].str.split(',').map(len) < 40]
#df.info()
#len(df)

df['text'].fillna("", inplace=True)

df['author'] = df['author'].fillna('unknown')
# Get average title size
df['title_size'] = df['title'].apply(lambda x: len(str(x)))

# If title size == 3("NAN") then change title size = 0
train_nan_index = df[df['title_size'] == 3].index
df['title_size'][train_nan_index] = 0

train_avg_title_size = int(df.value_counts(['title_size']).mean())
print("train avg_title_size: ", train_avg_title_size)

df['title'][train_nan_index] = df['text'][train_nan_index].apply(lambda x: str(x)[:train_avg_title_size])

df.drop('title_size', axis=1)

df["text"] = df["title"] + " " + df["text"]
df['text'] = df['author'] + " " + df['text']


df.drop(['id', 'title', 'author'], axis=1, inplace=True)
stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
# Author Data has some missing values
# sns.heatmap(Data.isnull(), cbar=False)


# df.columns
# !pip install transformers[tf-cpu]
# !pip install git+https://github.com/huggingface/transformers


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


df['text'] = df['text'].apply(lambda x: preprocess(x))
x_train = df['text']
y_train = df['label']


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
print("Vocab size : ", vocab_size)

x_train = pad_sequences(
    tokenizer.texts_to_sequences(x_train),
    maxlen=200)
x_val = pad_sequences(
    tokenizer.texts_to_sequences(x_val),
    maxlen=200)

print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)

inputs = Input(shape=(200,), dtype='int32')

embedding = tf.keras.layers.Embedding(vocab_size, 300)(inputs)
net = SpatialDropout1D(0.2)(embedding)
net = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(net)
net = Dense(32, activation='relu')(net)
net = Dropout(0.3)(net)
net = Dense(1, activation='sigmoid')(net)

outputs = net
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(),
             loss ='binary_crossentropy',
             metrics =['accuracy'])
model.fit(
    x_train,
    y_train,
    batch_size=2000,
    epochs=7,
    validation_data=(x_val, y_val),
    callbacks=[tensorboard]
)

"""
model.save("D:/mod_temp/mod 2")

new_model = keras.models.load_model("D:/mod_temp/mod 2")

converter = tf.lite.TFLiteConverter.from_keras_model(new_model) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
"""


