import pandas as pd
from signatureMatrix import build_signature_matrix
from jaccard import plot, jaccard_df
from config import Config

from LSH import *
filename = "preprocess-news_articles_small.csv"
df = pd.read_csv(filename)

df2 = jaccard_df(df, filename, Config["shingle_size"])
plot(df2)

iterations, bands, rows = get_ideal_bands(Config["d1"], Config["d2"], Config["s1"], Config["s2"])
print(iterations, bands, rows)
print("False postive chance at", Config["d1"], ":", calc_prob_sim(Config["d1"], bands, rows))
print("False negative chance at", Config["d2"], ":", 1 - calc_prob_sim(Config["d2"], bands, rows))

signature_matrix = build_signature_matrix(df, iterations, Config["shingle_size"], Config["DEBUG"])

print("generated signature matrix")
# https://towardsdatascience.com/understanding-locality-sensitive-hashing-49f6d1f6134


# nr_bandz = get_number_of_bands(0.8, 16)
# print(nr_bandz)
# debug_nr_bandz(nr_bandz[1],nr_bandz[2])
# #
band_val = use_bands(signature_matrix, bands)
print(band_val)
val = LSH_sig_matrix(signature_matrix, Config["d2"])
print(val[1])


print("oke")
