import pandas as pd
import itertools
from sklearn.metrics import jaccard_score as js
import os
import numpy as np
import matplotlib.pyplot as plt
from signatureMatrix import build_signature_matrix
from jaccard import plot, jaccard_df

from LSH import *
filename = "preprocess-news_articles_small.csv"
df = pd.read_csv(filename)

# df2 = jaccard_df(df, filename)
# plot(df2)


signature_matrix = build_signature_matrix(df, 16, 3, True)

# https://towardsdatascience.com/understanding-locality-sensitive-hashing-49f6d1f6134




nr_bandz = get_number_of_bands(0.8,16)
print(nr_bandz)
# debug_nr_bandz(nr_bandz[1],nr_bandz[2])
# #
band_val = use_bands(signature_matrix,nr_bandz[1])
print(band_val)
val = LSH_sig_matrix(signature_matrix,0.8)
print(val[1])



print("oke")


