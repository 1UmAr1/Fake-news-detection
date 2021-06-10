import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from transformers import TFBertModel, BertTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from keras.layers import Dropout, Dense
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf
import re
import keras


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


def bert_encode(data, max_len):
    input_ids = []
    attention_masks = []

    for i in range(len(data)):
        encoded = bert_tokenizer.encode_plus(data[i],
                                             add_special_tokens=True,
                                             max_length=max_len,
                                             pad_to_max_length=True,
                                             return_attention_mask=True)

        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    return np.array(input_ids), np.array(attention_masks)


# get BERT layer
bert_layers = TFBertModel.from_pretrained('bert-base-uncased')

# get BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

df['text'] = df['text'].apply(lambda x: preprocess(x))
x_train_bert = df['text']
y_train_bert = df['label']


train_input_ids, train_attention_masks = bert_encode(x_train_bert, 60)
input_ids = tf.keras.Input(shape=(60,), dtype='int32', name='input_ids')
attention_masks = tf.keras.Input(shape=(60,), dtype='int32', name='attention_masks')

output = bert_layers([input_ids, attention_masks])
output = output[1]
net = tf.keras.layers.Dense(16, activation='relu')(output)
net = tf.keras.layers.Dropout(0.2)(net)
net = tf.keras.layers.Dense(1, activation='sigmoid')(net)
outputs = net

model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(
    [train_input_ids, train_attention_masks],
    y_train_bert,
    validation_split=0.2,
    epochs=4,
    batch_size=128)


