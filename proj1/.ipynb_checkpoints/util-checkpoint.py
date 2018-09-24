import numpy as np
import string
import binascii
from itertools import combinations # to find all pairs in a set, basically (N choose 2)

# load data
def parse_data(fname):
    data = []
    for line in open(fname):
        # split string in two by space so first word is id, second word is rest of document
        id_data = line.split(' ',1)[0]
        # apply lower case and removal of all punctuations + next line (\n) and white space ( )
        s_data = line.split(' ',1)[1].lower().translate(None, ' \n'+string.punctuation)
        data.append((id_data,s_data))
    return data

# load truth data
def parse_data_truth(fname):
    data = []
    for line in open(fname):
        # split string in two
        f1 = line.split(' ',1)[0].translate(None, ' \n')
        f2 = line.split(' ',1)[1].translate(None, ' \n')
        data.append((f1,f2))
    return data

def shingle_document(astring,k):
    data = []
    for i in range(len(astring)-k+1):
        data.append(binascii.crc32(astring[i:(i+k)]))
        #data.append(astring[i:(i+k)]) # for testing
    return np.unique(data) # need to return sets, not bags

def jaccard(st1,st2):    
    # concat, and sort
    st = np.sort(np.concatenate((st1,st2)))
    
    # assume initially no intersection and all union
    inter = 0.
    union = len(st)
    
    for i in range(union-1):
        if st[i] == st[i+1]:
            # for each intersection add count to inter and remove count from union
            inter += 1
            union -= 1
            
    return inter / union # Jaccard distance would be d(x,y) = 1 - JSIM(x,y)

def jaccard_test(st1,st2):
    # test method using numpy functions to make sure my jaccard method works correctly.
    return len(np.intersect1d(st1,st2))*1.0 / len(np.union1d(st1,st2))*1.0

def processData(fid,k):
    data = []
    dt = parse_data(fid) # parse data
    dt_truth = parse_data_truth(fid[0:-5]+'truth') # parse truth data
    N = len(dt)
    
    for dtpair in combinations(dt, 2):  # N choose 2 to loop through all pairs
        dt1 = shingle_document(dtpair[0][1],k)
        dt2 = shingle_document(dtpair[1][1],k)
        dtjac = jaccard(dt1,dt2)
        if ((dtpair[0][0],dtpair[1][0]) or (dtpair[1][0],dtpair[0][0])) in dt_truth:
            data.append((dtpair[0][0],dtpair[1][0],dtjac,'Y'))
        else:
            data.append((dtpair[0][0],dtpair[1][0],dtjac,'N'))
    return data
