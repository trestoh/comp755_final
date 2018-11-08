import string
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import urllib
import os.path
import sys
from sklearn import linear_model
from data_import import percolator_import

try:
    import cPickle as pickle
    kwargs = {}
except:
    import _pickle as pickle
    kwargs = {'encoding':'bytes'}
    
import gzip


'''
filename = "/Users/trentstohrer/Desktop/UNC CS/COMP755/final_project/" + sys.argv[1] + "_target.pin"
decoy_file = "/Users/trentstohrer/Desktop/UNC CS/COMP755/final_project/" + sys.argv[1] + "_decoy.pin"
true_hit_file = "/Users/trentstohrer/Desktop/UNC CS/COMP755/final_project/" + sys.argv[1] + "_percolator_psms.txt"

with open(filename) as f:
    content = f.readlines()

with open(decoy_file) as f:
    content += f.readlines()


with open(true_hit_file) as f:
    true_hit_text = f.readlines()

count = 0

data = []

peptides = {}
peptide_number = {}

pept_count = 0

for line in content:
    
    lines = line.split("\t")

    if lines[0] != "SpecId":

        stripped_peps = lines[23][2:(len(lines[23])-2)]

        if not peptides.get(stripped_peps):
            peptides[stripped_peps] = pept_count
            peptide_number[pept_count] = stripped_peps
            pep = pept_count
            pept_count += 1

        else:
            pep = peptides[stripped_peps]

        dat = [int(lines[2]), int(lines[1]), float(lines[5]), float(lines[6]), float(lines[7]), float(lines[8]), float(lines[9]), float(lines[10]), 
            int(lines[11]), int(lines[12]), int(lines[13]), int(lines[14]), int(lines[15]), int(lines[16]), int(lines[17]),
            int(lines[18]), int(lines[19]), float(lines[20]), float(lines[21]), float(lines[22]), pep]
        data.append(dat)
    count += 1

data = np.array(data)

data = data[np.argsort(data[:,5])][::-1]
data = data[np.argsort(data[:,0], kind='mergesort')]

true_hits = {}

for line in true_hit_text:
    
    lines = line.split("\t")
    if lines[0] != "file_idx":
        #print ("q = %s" % lines[7])
        if float(lines[7]) > .01:
            break
        true_hits[int(lines[1])] = lines[10]
        #print (true_hits[lines[1]])
'''

data, true_hits, peptide_number = percolator_import(sys.argv[1])

'''
for i in range(0, len(data)):
    if true_hits.get(int(data[i,0])) == peptide_number[int(data[i,20])]:
        print(data[i, 0], data[i,1] ,data[i,5], peptide_number[int(data[i,20])])
'''

#print (data[0,:])

pre_trans_data = data

scan_nums = data[:,0]
labels = data[:,1]
data = data [:, [i for i in range(1, 20)]]

#print (scan_nums[0])

#print (np.mean(data[:,1]), np.std(data[:,1]))

for i in range (1, len(data[0,:])):
    data[:,i] = (data[:,i] - np.mean(data[:,i])) / (np.std(data[:,i]))

data[:,[8,13,14]] = 0

data = data [:, [i for i in range(1, 19)]]

classes = [-1,1]

sgd_clf = linear_model.SGDClassifier(loss="hinge", warm_start=True)

count = 0
for scan in scan_nums:
    if int(scan) > 1000:
        break
    #init_set.append(data[count,:])
    count += 1 

#labels.reshape(-1,1)

init_set = data[[i for i in range(0,count)], :]
init_labels = labels[[i for i in range(0,count)]]

#print (init_set)
#print (init_labels)

sgd_clf.fit(init_set, init_labels)

true_pos = 0
true_neg = 0
false_pos = 0 
false_neg = 0

#print(pre_trans_data[count])

print(pre_trans_data[count])

current_scan = int(pre_trans_data[count][0])
last_scan = 0

true_true_hits = 0
false_true_hits = 0
expected_true_hits = 0

while (count < len(data)):
    current_scan = int(pre_trans_data[count][0])
    if current_scan != last_scan:
        last_scan = current_scan
        if true_hits.get(current_scan) is not None:
            expected_true_hits += 1
        pred = sgd_clf.predict(data[count, :].reshape(1,-1))
        #print(int(pre_trans_data[count][1]), pred[0])
        if int(pre_trans_data[count][1]) == int(pred[0]):
            if int(pred[0]) == 1:
                if true_hits.get(current_scan) == peptide_number[int(pre_trans_data[count][20])]:
                    true_true_hits += 1
                else:
                    false_true_hits += 1
                #print (peptide_number[int(pre_trans_data[count][20])])
                true_pos += 1
            else:
                true_neg += 1
        else:
            if int(pred[0]) == 1:
                false_pos += 1
            else:
                false_neg += 1
    count += 1

print ("True pos: %d, True neg: %d, False pos: %d, False neg: %d" % (true_pos, true_neg, false_pos, false_neg) )
print ("Final true pos: %d, final false neg: %d, expected true hits: %d" % (true_true_hits, false_true_hits, expected_true_hits))
