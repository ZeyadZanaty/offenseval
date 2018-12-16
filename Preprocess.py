import numpy as np
import copy

class Preprocess:

    def __init__(self,data):
        self.data = copy.deepcopy(data)

    def tokenize(self):
        from nltk import word_tokenize
        for i,tweet in enumerate(self.data):
            self.data[i] = word_tokenize(tweet)
        return self.data

    def remove_stopwords(self):
        from nltk.corpus import stopwords
        import re
        stop = set(stopwords.words("english"))
        noise = ['@','USER','#']
        for i,tweet in enumerate(self.data):
            self.data[i] = [w for w in tweet if w not in stop and not re.match(r"[^a-zA-Z\d\s]+", w) and w not in noise]
        return self.data

    def clean(self, params):
        params = ['tokenize']+list(params)
        for p in params:
            clean_call = getattr(self, p)
            if clean_call:
                clean_call()
            else:
                raise Exception(str(p)+' is not an available function')
        return self.data