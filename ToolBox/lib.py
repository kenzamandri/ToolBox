
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import string
import nltk
try:
    nltk.data.find('punkt')
    nltk.data.find('stopwords')
    nltk.data.find('averaged_perceptron_tagger')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer



#Prepcossing first
def cleaning(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')

    #text_=text.lower()
    text =''.join(word for word in text if not word.isdigit())
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

    vectorizer = CountVectorizer().fit(data)
    print('vec passed')
    return vectorizer

def LDA_model(data):
    vectorizer= vector(data)
    data_vectorized = vectorizer.transform(data)
    lda_model = LatentDirichletAllocation(n_components=2).fit(data_vectorized)
    print(lda_model)
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
    data_vectorized = vectorizer.fit_transform(data)
    data['Topic'] = np.argmax(lda_model.transform(data_vectorized), axis = 1)

    example_vectorized = vectorizer.transform(text_test)

    lda_vectors = lda_model.transform(example_vectorized)

    if lda_vectors[0][0] > lda_vectors[0][1]:
          return f"topic 0 :, {lda_vectors[0][0]}"
    else:
        return f"topic 1 :, {lda_vectors[0][1]}"



if'__name__'=='__main__':

    data = pd.read_csv('raw_data/data', sep=",", header=None)
    data.columns = ['text']
    data["clean_text"] = data['text'].apply(cleaning)
    data["clean_text"]= data['clean_text'].astype('str')
    ext_test=['I Love football but wht I love most is baskettball where people have fun and jump']
    print(  )
