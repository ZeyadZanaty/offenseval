import numpy as np
import copy
from tqdm import tqdm
class Preprocess:

    def __init__(self,data):
        self.data = copy.deepcopy(data)

    def tokenize(self):
        from nltk import word_tokenize
        for i,tweet in tqdm(enumerate(self.data),'Tokenization'):
            self.data[i] = word_tokenize(tweet)
        return self.data

    def remove_stopwords(self):
        from nltk.corpus import stopwords
        import re
        stop = set(stopwords.words("english"))
        noise = ['@','USER','#']
        for i,tweet in tqdm(enumerate(self.data),'Stopwords Removal'):
            self.data[i] = [w for w in tweet if w not in stop and not re.match(r"[^a-zA-Z\d\s]+", w) and w not in noise]
        return self.data
    
    def get_pos(self, word):
        from collections import Counter
        from nltk.corpus import wordnet  # To get words in dictionary with their parts of speech
        w_synsets = wordnet.synsets(word)
        pos_counts = Counter()
        pos_counts["n"] = len([item for item in w_synsets if item.pos() == "n"])
        pos_counts["v"] = len([item for item in w_synsets if item.pos() == "v"])
        pos_counts["a"] = len([item for item in w_synsets if item.pos() == "a"])
        pos_counts["r"] = len([item for item in w_synsets if item.pos() == "r"])
        most_common_pos_list = pos_counts.most_common(3)
        return most_common_pos_list[0][0]  # first indexer for getting the top POS from list, second indexer for getting POS from tuple( POS: count )

    def lemmatize(self):
        from nltk.stem import WordNetLemmatizer  # lemmatizes word based on it's parts of speech
        wnl = WordNetLemmatizer()
        for i, tweet in tqdm(enumerate(self.data),'Lemmatization'):
            for j, word in enumerate(tweet):
                self.data[i][j] = wnl.lemmatize(word, pos=self.get_pos(word))
        return self.data

    def clean(self, params):
        params = ['tokenize']+list(params)
        for param in tqdm(params,'Preprocessing'):
            clean_call = getattr(self, param)
            if clean_call:
                clean_call()
            else:
                raise Exception(str(p)+' is not an available function')
        return self.data