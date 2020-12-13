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
for signature in signature_matrix:
    s = np.array_split(signature, b)
    bands.append(s)

print("oke")


