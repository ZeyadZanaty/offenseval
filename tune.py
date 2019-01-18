from DataReader import DataReader
from Preprocessor import Preprocessor
from Vectorizer import Vectorizer
from Classifier import Classifier
import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import train_test_split as split

def clean_string(str):
  newstr = ''
  noise = ['[',']','{','}',"'",'(',')',',']
  for c in str:
    if c == ' ':
      newstr+='\n'
    elif c not in noise:
      newstr+= c
  return newstr

def get_plot_arrs(params,acc_num, items):
  plt_arr = [acc[0]['test_score'] for acc in items]
  plt_arr = plt_arr[:acc_num]
  ticks_arr = [clean_string(str(acc[1]))+clean_string(str(acc[2]))+'\n Parameters: '+
               clean_string(str(params))for acc in items]
  ticks_arr = ticks_arr[:acc_num]
  return plt_arr,ticks_arr

def plot_clfs(width=0.15,max_num=3,acc_sep=1.2,param_sep=2.8):

  # width: width of bar
  # max_num: number of accuracies to plot per parameter
  # acc_sep: seperation between accuracies for a prameter
  # param_sep: seperation between parameters
  for key, value in clf_dict.items():
    print(key)
    plt_acc = []
    plt_tks = []
    my_dpi = 192
    fig = plt.figure(figsize=(4096 / my_dpi, 2160 / my_dpi), dpi=my_dpi)
    acc_sep = acc_sep
    param_sep = param_sep
    acc_sep += len(value)/20
    param_sep += len(value)/10
    for k, v in value.items():
      plt.title(str(key))
      plt_arr, plt_arr_ticks = get_plot_arrs(k, max_num, v)
      print(plt_arr, plt_arr_ticks)
      if not plt_acc:
        plt_acc = plt_arr
        plt_tks = plt_arr_ticks
      else:
        plt_acc += plt_arr
        plt_tks += plt_arr_ticks
    ind = np.array(1)
    x = 1
    for i in range(1, len(plt_acc)):
      if ((i) % max_num == 0):
        x += param_sep * width
      else:
        x += acc_sep * width
      ind = np.append(ind, x)
    ind = ind[0:len(plt_acc)]
    colors =['#009688','#35a79c','#54b2a9','#65c3ba','#83d0c9','#8fd4ce','#9bd9d3']
    colors =[colors[i] for i in range(0,max_num)]
    plt.bar(ind, plt_acc, width=width,color=colors)
    plt.xticks(ind, plt_tks)
  plt.show()

cleaning_operations = ['remove_stopwords','lemmatize','stem']

prp_list = [
    i for j in range(len(cleaning_operations)) for i in itertools.combinations(cleaning_operations,j+1)
    ]

vec_list = [['fasttext',{}],['tfidf',{}],['word2vec',{}],['glove',{}],['count',{}]]

clf_list = [['RandomForest',{'n_estimators': [n for n in range(10,200,10)]}],
            ['KNN',{'n_neighbors':[n for n in range(1,8,2)]}],
            ['SVC',{'C':[0.1,10,100],'kernel':['rbf','poly']}],
            ['M-NaiveBayes',{'alpha':[0,1,10],'fit_prior':[True,False]}],
            ['G-NaiveBayes',{'var_smoothing':[1e-15,1e-09,1e-05]}],
            ['LogisticRegression',{'penalty':['l2'],'solver' : ['newton-cg', 'lbfgs', 'sag']}],
            ['DecisionTree',{'criterion':['gini','entropy']}],
            ['MLP',{'activation':['tanh', 'relu'],'solver':['sgd','adam','lbfgs']}]]

dr = DataReader('./datasets/training-v1/offenseval-training-v1.tsv','A')
data, labels = dr.get_labelled_data()
data, labels = dr.shuffle(data,labels,'random')
data,_,labels,_ = split(data,labels,test_size=0.5,stratify=labels)

clf_dict = {clf[0]: {} for clf in clf_list}

for vec in vec_list:
  print('Vectorization: ',vec[0])
  for prp in prp_list:
    print('Preprocessing: ',prp)
    preprocessor = Preprocessor(prp)
    clean_data = preprocessor.clean(data)

    vectorizer = Vectorizer(type=vec[0],params=vec[1])
    vectorized_data = vectorizer.vectorize(clean_data)

    for cl in clf_list:
      if vec[0] not in ['BoW','tfidf'] and cl[0]=='M-NaiveBayes':
        continue
      print('Classifier: ',cl[0])
      clf = Classifier(cl[0])
      params_accs = clf.tune(vectorized_data,labels,cl[1], best_only=False)
      print('Scores:',params_accs)
      for key, value in params_accs.items():
        if key in clf_dict[cl[0]]:
          clf_dict[cl[0]][key] += [(value, vec,prp)]
        else:
          clf_dict[cl[0]][key] = [(value, vec, prp)]

for key,value in clf_dict.items():
  for k,v in value.items():
    v = (sorted(v, key=lambda x: x[0]['test_score'], reverse=True))
    clf_dict[key][k] = v
print(clf_dict)

plot_clfs(width=0.15,max_num=3,acc_sep=1.2,param_sep=2.8)

with open('tuning.txt','w') as f:
  f.write('\n'+str(clf_dict)+'\n')