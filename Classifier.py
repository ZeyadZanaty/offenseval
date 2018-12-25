import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from itertools import product
import imp 
class Classifier:
    
    def __init__(self,type,params={}):
        __classifers__ = {
        'KNN': KNeighborsClassifier,
        'SVC': SVC,
        'DecisionTree': DecisionTreeClassifier,
        'RandomForestClassifier': RandomForestClassifier,
        'LogisticRegression': LogisticRegression,
        'MLP': MLPClassifier,
        'AdaBoost': AdaBoostClassifier,
        'Bagging': BaggingClassifier
        }
        if type not in __classifers__.keys():
            raise Exception('Available Classifiers: ',__classifers__.keys())
        self.classifier = __classifers__[type]
        self.params = params
        self.model = self.classifier(**self.params)   

    def fit(self,tr_data,tr_labels):
        return self.model.fit(tr_data,tr_labels)

    def predict(self,tst_data):
        return self.model.predict(tst_data)

    def score(self,tst_data,tst_labels):
        return self.model.score(tst_data,tst_labels)

    def tune(self,tr_data,tr_labels,tune_params=None,best_only=False):
        if not tune_params:
            tune_params = self.params
        tuner = GridSearchCV(self.model,tune_params,n_jobs=3)
        tuner.fit(tr_data,tr_labels)
        self.model = tuner.best_estimator_
        if best_only:
            return tuner,{'score':tuner.best_score_,'parmas':tuner.best_params_}
        else:
            return tuner,tuner.cv_results_
    
    def get_model(self):
        if getattr(self,'model',None):
            return self.model
        else:
            raise Exception('Model has not been created yet.')

        
        
