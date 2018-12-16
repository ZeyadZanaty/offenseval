from DataReader import DataReader

dr = DataReader('./datasets/training-v1/offenseval-training-v1.tsv')
tr_data,tr_labels = dr.get_labelled_data()
print(tr_data[0:4],tr_labels[0:4])
