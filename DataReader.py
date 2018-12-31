import numpy as np
import csv
from tqdm import tqdm
class DataReader:

    def __init__(self,file_path,sub_task=None):
        self.file_path = file_path
        self.sub_task = sub_task

    def get_labelled_data(self):
        data = []
        labels = []
        with open(self.file_path,encoding='utf8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for i,line in enumerate(tqdm(reader,'Reading Data')):
                if i is 0:
                    continue
                label = self.str_to_label(line[-3:])
                if  self.sub_task:
                    self.filter_subtask(data,labels,line[1],label)
                else:
                    labels.append(label)
                    data.append(line[1])
        return data,labels
    
    def str_to_label(self,all_labels):
        label = 0
        if all_labels[0] == 'OFF':
            if all_labels[1] == 'UNT':
                label = 1
            elif all_labels[1] == 'TIN':
                if all_labels[2] == 'IND':
                    label = 2
                elif all_labels[2] == 'GRP':
                    label = 3
                elif all_labels[2] =='OTH':
                    label = 4
        return label
    
    def filter_subtask(self,data,labels,sample,label):
        if self.sub_task == 'A':
            data.append(sample)
            labels.append(int(label>0))
        elif self.sub_task =='B':
            if label > 0:
                data.append(sample)
                labels.append(int(label>1))
        elif self.sub_task == 'C':
            if label > 1:
                data.append(sample)
                labels.append(label-2)
        


#0 - Not offensive
#1 - Offensive untargeted
#2 - Offensive targeted indiviualds
#3 - Offensive targeted groups 
#4 - Offensive targeted others