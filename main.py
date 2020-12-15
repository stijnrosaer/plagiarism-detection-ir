import pandas as pd
import itertools
from sklearn.metrics import jaccard_score as js
import os
import numpy as np
import matplotlib.pyplot as plt
from signatureMatrix import build_signature_matrix
from jaccard import plot, jaccard_df

filename = "preprocess-news_articles_small.csv"
df = pd.read_csv(filename)

# df2 = jaccard_df(df, filename)
# plot(df2)


signature_matrix = build_signature_matrix(df, 16, 3, True)

# https://towardsdatascience.com/understanding-locality-sensitive-hashing-49f6d1f6134



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

    return val/float(len(sig1))


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



def use_bands(signature_matrix, nr_bands):
    bands = []
    signature_matrix = transpose(signature_matrix)
    for signature in signature_matrix:
        s = np.array_split(signature, nr_bands)
        bands.append(s)

    possible_plags = []

    for i in range(len(bands)):
        for j in range(i + 1, len(bands), 1):

            for k in range(len(bands[i])):
                if list(bands[i][k]) == list(bands[j][k]):
                    possible_plags.append((i,j))
                    break
    return possible_plags


def get_number_of_bands(thershold, nr_iters, thresh_maximum = 1.0):
    diff = float("inf")
    val = -1
    for b in range(1,nr_iters+1):
        if nr_iters% b == 0 or b == 1:
            r = nr_iters/b
            t = pow(1/b, 1/r)
            if abs(thershold -t)< diff and (t <= thresh_maximum):
                diff = abs(thershold -t)
                val = (t,b,r)

    return val



def debug_nr_bandz(b, r):
    for i in range(10):
        s =i*0.1
        t =1-pow(1-pow(s,r),b)
        print("prob "+ str(s) + ": " + str(t))
nr_bandz = get_number_of_bands(0.8,16)
print(nr_bandz)
# debug_nr_bandz(nr_bandz[1],nr_bandz[2])
# #
band_val = use_bands(signature_matrix,nr_bandz[1])
print(band_val)
val = LSH_sig_matrix(signature_matrix,0.8)
print(val[1])



print("oke")


