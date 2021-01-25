from nltk import ngrams
import random


def get_hash(gram, mask):
    """
    the "random hash function used to get different hash functions for each different permutation
    :param gram: the object to be hashed
    :param mask: the random string (of size 64 bits) used to make every hash function different but consistently so
    :return: the hash value of gram with our specific hash function
    """
    return hash(gram) ^ mask


def permutation(df, mask, gram_size):
    """
    generate a list of permutations based on the minhash algorithm for each document in the collection df
    :param df: the collection of all documents
    :param mask: a supportvalue used by the hashfunction to change the function each time a new permutation is done
    :param gram_size: size of the shingles
    :return:  a list of permutations based on the minhash algorithm for each document in the collection df
    """
    result = []
    for index, line in df.iterrows(): # loop over each document in df
        doc_id = line["News_ID"]
        doc = line["article"]

        grams = list((ngrams(doc.split(), gram_size))) #split our documnet into shingles of size gram_size

        lowest = float("inf") # initialize the minimal value infinity so another will always take it's place

        for g in grams: #for each shingle get the hash and check it to our smallest hash, if it's smaller , replace it
            h = get_hash(g, mask)

            if h < lowest:
                lowest = h

        result.append((doc_id, lowest)) #append the minhash for each document to the results

    return result


def build_signature_matrix(df, iterations, gram_size, debug = False):
    """
    build a signature matrix of all documents in df using a number of different permutations
    :param df:the collection of all documents
    :param iterations:the amount of different permutations we will do
    :param gram_size: size of the shingles
    :param debug:used to make the algorithm do a consistent value instead of random
    :return: the complete signature matrix, which is a list of lists, where each inner list is a different minHash
    permutation for all documents
    """
    m = []
    if debug:
        random.seed(1)
    for i in range(iterations): #get iterations amount of permutations

        mask = random.getrandbits(64) #get the mask used to make each permutation different
        res = permutation(df, mask, gram_size) # get a list of the permutation for each document
        m.append(res) #append it to the signature matrix

    return m
