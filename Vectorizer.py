import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
from gensim.models import Word2Vec,FastText
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
from tqdm import tqdm

class Vectorizer:
    
    def __init__(self,type,pre_trained=False,retrain=False,params={}):
        self.type = type
        self.pre_trained = pre_trained
        self.params = params
        self.retrain = retrain

    def word2vec(self):
        from os import listdir
        if not self.pre_trained:
            if 'word2vec.model' not in listdir('.') or self.retrain:
                model = self.train_w2v()
            else:
                model = Word2Vec.load("word2vec.model")
        else:
            model = Word2Vec(self.data)
        vectorizer = model.wv
        vectors = [
            np.array([vectorizer[word] if word in model else np.zeros(100) for word in tweet]).flatten() for tweet in tqdm(self.data,'Vectorizing')
            ]
        max_len = np.max([len(vector) for vector in vectors])
        self.vectors = [
            np.array(vector.tolist()+[0 for _ in range(max_len-len(vector))]) for vector in vectors
            ]
        return self.vectors

    def train_w2v(self):
        from gensim.test.utils import get_tmpfile
        path = get_tmpfile("word2vec.model")
        model = Word2Vec(self.data, size=100, window=5, min_count=1, workers=5)
        model.train(self.data, total_examples=len(self.data), epochs=500)
        model.save("word2vec.model")
        print("\nDone training w2v model!")
        return model

    def tfidf(self):
        vectorizer = TfidfVectorizer(**self.params)
        untokenized_data =[' '.join(tweet) for tweet in self.data] 
        self.vectors = vectorizer.fit_transform(untokenized_data).toarray()
        return self.vectors

    def glove(self):
        if 'glove-twitter.model' not in listdir('.') or self.train:
            print('Downloading Glove Embeddings')
            model = api.load('glove-twitter-100')
        else:
            model = Word2Vec.load('glove-twitter.model')
        vectorizer = model.wv
        vectors = [np.array([vectorizer[word] if word in model else np.zeros(100) for word in tweet]).flatten() for tweet in tqdm(self.data,'Vectorizing')]
        max_len = np.max([len(vector) for vector in vectors])
        self.vectors = [
            np.array(vector.tolist()+[0 for _ in range(max_len-len(vector))]) for vector in vectors
            ]
        return self.vectors

    def vectorize(self,data):
        self.data = data
        vectorize_call = getattr(self, self.type, None)
        if vectorize_call:
            vectorize_call()
        else:
            raise Exception(str(self.type),'is not an available function')
        return self.vectors
    
    def fit(self,data):
        self.data = data


