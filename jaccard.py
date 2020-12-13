import pandas as pd
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt


OVERWRITE = False
DEBUG = False

def jaccard_index(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / len(union)


def jaccard_df(df, filename):
    if os.path.isfile("jaccard-" + filename) and not OVERWRITE:
        print("Using existing jaccard-library")
        df2 = pd.read_csv("jaccard-" + filename)
    else:
        print(f"Building jaccard-library for {filename}")

        scores = []
        for pair in itertools.combinations(df.to_records(index=False), 2):
            if pair[0]["News_ID"] != pair[1]["News_ID"]:
                jaccard_score = jaccard_index(pair[0]["article"].split(), pair[1]["article"].split())
                scores.append([pair[0]["News_ID"], pair[1]["News_ID"], jaccard_score])

        df2 = pd.DataFrame(scores, columns=["id1", "id2", "score"])


        df2.to_csv("jaccard-" + filename, index=False)

        if DEBUG:
            print(len(scores))
            print(df2.head(20))

    return df2


def plot(df):
    t = df.groupby(pd.cut(df["score"], np.arange(0, 1 + 1e-10, 0.1))).size()

    t.plot(kind="bar")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.show()

    if DEBUG:
        print(t.all)

        q = df[df.score > 0.9]
        print(q.head(10))
