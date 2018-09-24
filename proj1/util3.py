# Functions from part1 are in util.py
# Functions from part2 are in util2.py

# util.py functions:
#
# parse_data(filename) , parse_data_truth(filename)
# shingle_document(string,k)
# jaccard(string1,string2)
# processData(fileID,k)
#
# util2.py functions:
#
# processShingles(filename,k), invert_shingles(shingled_documents)
# make_hashes(numHashes)
# make_minhash_signature(shingled_data numHashes)
# minhash_similarity(id1,id2,minhash_matrix, docids)
# processJsimMH(filename,k,numHashes):

from util import *
from util2 import *
import scipy.optimize as opt
import math
import time

# Given functions
def _make_vector_hash(num_hashes, m=4294967295):
    hash_fns = make_hashes(num_hashes)
    def _f(vec):
        acc = 0
        for i in range(len(vec)):
            h = hash_fns[i]
            acc += h(vec[i])
        return acc % m
    return _f

def _choose_nbands(threshold, nhashes):
    error_fun = lambda x: (threshold-((1/x[0])**(x[0]/nhashes)))**2
    res = opt.minimize(error_fun, x0=(10), method='Nelder-Mead')
    b = int(math.ceil(res['x'][0]))
    r = int(nhashes / b)
    final_t = (1/b)**(1/r)
    return b, final_t
    
    from collections import defaultdict

def do_lsh(minhash, numhashes, docids, threshold):
    b, _ = _choose_nbands(threshold, numhashes)
    r = int(numhashes / b)
    narticles = len(docids)
    hash_func = _make_vector_hash(r)
    buckets = []
    for band in range(b):
        start_index = int(band * r)
        end_index = min(start_index + r, numhashes) - 1 # w/o -1 you get an off by 1 index error
        cur_buckets = defaultdict(list)
        for j in range(narticles):
        # THIS IS WHAT YOU NEED TO IMPLEMENT
        # ok, but do I have to use dictionaries...
            cur_buckets[docids[j]] = hash_func(minhash[start_index:end_index,j])
        buckets.append(cur_buckets)
    
    # hash tables per a band
    return buckets

# Load data and calculate minHash matrix for given hashes and shingle size
fname = 'data/articles_10000.'
dat_t = parse_data_truth(fname+'truth')

k_size = 10
numHashes = 1000

if 1:
    start_time = time.time()
    # get a list of tuples of the form (docid, [shingles])
    shingles = processShingles(fname+'train',k_size)
    print 'SHINGLING | --- '+str(time.time()-start_time)+' seconds --- |'  
    # create minhash signature from data
    mh_matrix, docids = make_minhash_signature(shingles,numHashes)
    print 'MINHASH | --- '+str(time.time()-start_time)+' seconds --- |'
    np.save('data_gen/data_10000_lsh.npy', mh_matrix)
    np.save('data_gen/data_10000_lsh_docid.npy', docids)