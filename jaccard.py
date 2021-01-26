import pandas as pd
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
from nltk import ngrams
from config import Config

OVERWRITE = Config["Jacard_OVERWRITE"]  # used to state that the index of jacard values needs to be overwritten
DEBUG = Config["DEBUG"]  # used to print extra info


def jaccard_index(query, document):
    """
    calculate the jacard value between the 2 documents according to the algorithm
    :param query: the first document
    :param document: the second document
    :return: the jacard value
    """
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / len(union)


def jaccard_df(df, filename, shingles: int = 2):
    """
    either read or generate the jacard index and return the similarity between all different documents
    :param shingles: the length of the shingles
    :param df: the collection of documents
    :param filename: the name used to determine the location where the index exists or will be written
    :return:the jacard values of all document combos
    """
    if os.path.isfile("jaccard-" + filename) and not OVERWRITE:  # check wether or not we can use an existing index
        print("Using existing jaccard-library")
        df2 = pd.read_csv("jaccard-" + filename)
    else:
        print(f"Building jaccard-library for {filename}")

        scores = []
        for pair in itertools.combinations(df.to_records(index=False),
                                           2):  # iterate over every pair of 2 documents in the collection
            if pair[0]["News_ID"] != pair[1]["News_ID"]:  # check that you don't compare a document with itself
                # jaccard_score = jaccard_index(pair[0]["article"].split(), pair[1]["article"].split()) #calculate the jacard value for the pair of documents
                jaccard_score = jaccard_index(list(ngrams(pair[0]["article"].split(), shingles)), list(
                    ngrams(pair[1]["article"].split(),
                           shingles)))  # calculate the jacard value for the pair of documents

                scores.append([pair[0]["News_ID"], pair[1]["News_ID"], jaccard_score])  # store this score in the list

        df2 = pd.DataFrame(scores, columns=["id1", "id2", "score"])

        df2.to_csv("jaccard-" + filename,
                   index=False)  # write all scores to a file to no longer require calculations when redoing this function

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
