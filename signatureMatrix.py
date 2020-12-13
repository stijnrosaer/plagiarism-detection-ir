from nltk import ngrams
import random


def get_hash(gram, mask):
    return hash(gram) ^ mask


def permutation(df, mask, gram_size):
    result = []
    for index, line in df.iterrows():
        doc_id = line["News_ID"]
        doc = line["article"]

        grams = list((ngrams(doc.split(), gram_size)))

        lowest = float("inf")

        for g in grams:
            h = get_hash(g, mask)

            if h < lowest:
                lowest = h

        result.append((doc_id, lowest))

    return result


def build_signature_matrix(df, iterations, gram_size, debug = False):
    m = []
    for i in range(iterations):
        if debug:
            random.seed(i)
        mask = random.getrandbits(32)
        res = permutation(df, mask, gram_size)
        m.append(res)

    return m
