from DataReader import DataReader
from Preprocess import Preprocess

dr = DataReader('./datasets/training-v1/offenseval-training-v1.tsv')
data,labels = dr.get_labelled_data()
print(data[0:4])
prp = Preprocess(data)
data = prp.clean(['remove_stopwords'])
print(data[0:4])

