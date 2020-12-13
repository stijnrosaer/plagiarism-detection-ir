import pandas as pd
import itertools
from sklearn.metrics import jaccard_score as js
import os
import numpy as np
import matplotlib.pyplot as plt
from signatureMatrix import build_signature_matrix
from jaccard import plot, jaccard_df

filename = "news_articles_small.csv"
df = pd.read_csv(filename)

# df2 = jaccard_df(df, filename)
# plot(df2)


signature_matrix = build_signature_matrix(df, 10, 3)

# https://towardsdatascience.com/understanding-locality-sensitive-hashing-49f6d1f6134
b = 5
r = 2

bands = []

def transpose(m):
    """
    transpose a 2-dimensional matrix, making the outer list the inner list
    :param m: the matrix to transpose
    :return: the transposed matrix
    """
    trans = [[m[j][i][1] for j in range(len(m))] for i in range(len(m[0]))]
    return trans


def calc_signature_similarity(sig1, sig2):
    if len(sig1)!= len(sig2):
        raise Exception("compared signatures not of same size")
    val = 0
    for i in range(len(sig1)):
        if sig1[i]==sig2[i]:
            val+=1

    return val/len(sig1)


def LSH_sig_matrix(signature_matrix, threshhold):
    fullist = []
    exceed_list = []
    signature_matrix = transpose(signature_matrix)
    for i in range(len(signature_matrix)):
        for j in range(i+1, len(signature_matrix), 1):
            similarity = calc_signature_similarity(signature_matrix[i], signature_matrix[j])
            fulval = ((i,j), similarity)
            fullist.append(fulval)
            if similarity> threshhold:
                exceed_list.append(fulval)

    return (fullist, exceed_list)



def use_bands(trans_matrix):

    for signature in trans_matrix:
        s = np.array_split(signature, b)
        bands.append(s)

val = LSH_sig_matrix(signature_matrix,0.8)

print("oke")


