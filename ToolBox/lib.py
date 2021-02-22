
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer



#Prepcossing first
def cleaning(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')

    text_=text.lower()
    text =''.join(word for word in text_ if not word.isdigit())
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    clean_text = [w for w in word_tokens if not w in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in clean_text]
    return lemmatized

#data["clean_text"] = data['text'].apply(cleaning)
#data["clean_text"]= data['clean_text'].astype('str')


#training the LDA model
def vector(data):
    vectorizer = CountVectorizer().fit(data['clean_text'])
    return vectorizer
def LDA_model(data):
    vectorizer= vector(data)
    data_vectorized = vectorizer.transform(data['clean_text'])
    lda_model = LatentDirichletAllocation(n_components=2).fit(data_vectorized)
    return lda_model


#printing of the words associated with the potential topics
def print_topics(model, vectorizer):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], round(topic[i],1))
                        for i in topic.argsort()[:-10 - 1:-1]])


def testing(data, text_test):
    vectorizer= vector(data)
    lda_model =LDA_model(data)
    data_vectorized = vectorizer.fit_transform(data['clean_text'])
    data['Topic'] = np.argmax(lda_model.transform(data_vectorized), axis = 1)
    example_vectorized = vectorizer.transform(text_test)

    lda_vectors = lda_model.transform(example_vectorized)

    if lda_vectors[0][0] > lda_vectors[0][1]:
        print("topic 0 :", lda_vectors[0][0])
    else:
        print("topic 1 :", lda_vectors[0][1])
    return


if'__name__'=='__main__':
    data = pd.read_csv('raw_data/data', sep=",", header=None)
    data.columns = ['text']
    data["clean_text"] = data['text'].apply(cleaning)
    data["clean_text"]= data['clean_text'].astype('str')
