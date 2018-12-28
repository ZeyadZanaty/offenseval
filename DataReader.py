import numpy as np
import csv

class DataReader:

    def __init__(self,file_path,sub_task=None):
        self.file_path = file_path
        self.sub_task = sub_task

    def get_labelled_data(self):
        data = []
        labels = []
        with open(self.file_path,encoding='utf8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for i,line in enumerate(reader):
                if i!=0:
                    label = self.str_to_label(line[-3:])
                    if not self.sub_task:
                        labels.append(label)
                        data.append(line[1])
                    elif self.sub_task == 'A':
                        data.append(line[1])
                        labels.append(int(label>0))
                    elif self.sub_task =='B':
                        if label > 0:
                            data.append(line[1])
                            labels.append(int(label>1))
                    elif self.sub_task == 'C':
                        if label > 1:
                            data.append(line[1])
                            labels.append(label-2)
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

#0 - Not offensive
#1 - Offensive untargeted
#2 - Offensive targeted indiviualds
#3 - Offensive targeted groups 
#4 - Offensive targeted others