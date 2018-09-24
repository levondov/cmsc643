from util import *
import random    
import numpy
import time


def processShingles(fid,k):
    # given a file, will return a list of tuples of the form (docid, [shingles])
    data = []
    dt = parse_data(fid) # parse data
    
    for doc in dt:
        data.append((doc[0],shingle_document(doc[1],k)))

    return data

def invert_shingles(shingled_documents):
    data = [] # tuple list to return
    docs = [] # list of doc ids
    ii=0
    start_time = time.time()
    for doc in shingled_documents:
        docs.append(doc[0])
        for shingle in doc[1]:
            data.append((shingle,doc[0]))
        ii+=1
        print str(ii) +' / ' + str(len(shingled_documents)) + ' | shingle time: ' + str(time.time()-start_time) 
    # sort shingles and return
    return sorted(data, key=lambda x: x[0]), docs

def make_random_hash_fn(p=2**33-355, m=4294967295):
    a = random.randint(1,p-1)
    b = random.randint(0, p-1)
    return lambda x: ((a * x + b) % p) % m

def make_hashes(num_hash=1):
    h_functions = []
    for i in range(num_hash):
        h_functions.append(make_random_hash_fn()) 
    return h_functions
    

def make_minhash_signature(shingled_data, num_hashes):
    inv_index, docids_full = invert_shingles(shingled_data)
    num_docs = len(docids_full)

    # initialize the signature matrix with infinity in every entry
    sigmatrix = np.full([num_hashes, num_docs], np.inf)

    # generate hash functions
    hash_funcs = make_hashes(num_hashes)

    # iterate over each non-zero entry of the characteristic matrix
    ii = 0
    start_time = time.time()
    for row, docid in inv_index: # for each row in characteristic matrix
        docid_idx = docids_full.index(docid) # find index of document in full list
        for i,hash_func in enumerate(hash_funcs): # for each hash function
            if hash_func(row) < sigmatrix[i,docid_idx]:
                sigmatrix[i,docid_idx] = hash_func(row)
        ii+=1
        print str(ii) +' / ' + str(len(inv_index)) + ' | row minhash time: '+ str(time.time()-start_time) 
    return sigmatrix, docids_full


def minhash_similarity(id1, id2, minhash_sigmat, docids):
    # get column of the similarity matrix for the two documents
    id1_idx = docids.index(id1)
    id2_idx = docids.index(id2)
    
    # calculate the fraction of rows where two columns match
    return np.mean(minhash_sigmat[:,id1_idx] == minhash_sigmat[:,id2_idx])
    
def processJsimMH(fid,k,numHashes):
    
    # get a list of tuples of the form (docid, [shingles])
    shingles = processShingles(fid,k)
    
    # create minhash signature from data
    mh_matrix, docids = make_minhash_signature(shingles,numHashes)
    
    data = []
    for dtpair in combinations(docids, 2):  # N choose 2 to loop through all pairs of doc ids
        
        # calculate minhash jsim estimate
        mhJsim = minhash_similarity(dtpair[0],dtpair[1], mh_matrix, docids)
        
        # add to data
        data.append((dtpair[0],dtpair[1],mhJsim))
        
    return data        
