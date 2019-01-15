import numpy as np
import copy
from tqdm import tqdm
import imp
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
class Preprocessor:

    def __init__(self,*args):
        self.params =[]
        if args:
            if isinstance(args[0],tuple):
                self.params = list(*args)
            else:
                self.params = list(args)
        self.params = ['tokenize']+self.params

    def tokenize(self):
        from nltk import word_tokenize
        for i,tweet in tqdm(enumerate(self.data),'Tokenization'):
            self.data[i] = word_tokenize(tweet.lower())
        return self.data

    def remove_stopwords(self):
        from nltk.corpus import stopwords
        import re
        stop = set(stopwords.words("english"))
        noise = ['user']
        for i,tweet in tqdm(enumerate(self.data),'Stopwords Removal'):
            self.data[i] = [w for w in tweet if w not in stop and not re.match(r"[^a-zA-Z\d\s]+", w) and w not in noise]
        return self.data
    
    def get_pos(self, word):
        from nltk import pos_tag
        from nltk.corpus import wordnet
        tag = pos_tag([word])[0][1]
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatize(self):
        from nltk.stem import WordNetLemmatizer
        wnl = WordNetLemmatizer()
        for i, tweet in tqdm(enumerate(self.data),'Lemmatization'):
            for j, word in enumerate(tweet):
                self.data[i][j] = wnl.lemmatize(word, pos=self.get_pos(word))
        return self.data
    
    def stem(self):
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        for i,tweet in tqdm(enumerate(self.data),'Stemming'):
            for j,word in enumerate(tweet):
                self.data[i][j] = stemmer.stem(word)
        return self.data
    
    def word_cloud(self,labels=None,filter=None):
        if not isinstance(self.data[0],list):
            raise Exception('Data must be tokenized before using word cloud.')
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
        filters = ['NOT','UNT','TIN','GRP','OTH','OFF']
        if not filter:
            plot_data = [w for i,tweet in enumerate(self.data) for w in tweet]
        else:
            if not labels:
                raise Exception('Labels must be provided for filtering text.')
            filter = filters.index(filter)
            if filter == 4:
                plot_data = [w for i,tweet in enumerate(self.data) for w in tweet if labels[i] >0]
            else:
                plot_data = [w for i,tweet in enumerate(self.data) for w in tweet if labels[i]==filter]
        all_words = ' '.join(plot_data)
        wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()

    def clean(self, data):
        self.data = copy.deepcopy(data)
        for param in tqdm(self.params,'Preprocessing'):
            clean_call = getattr(self, param,None)
            if clean_call:
                clean_call()
            else:
                raise Exception(str(param)+' is not an available function')
        return self.data