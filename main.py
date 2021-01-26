import pandas as pd
from signatureMatrix import build_signature_matrix
from jaccard import plot, jaccard_df
from config import Config

from LSH import *

def jaccard(filename, config):
    df = pd.read_csv(filename)

    df2 = jaccard_df(df, filename, config["shingle_size"])
    plot(df2)

def lsh(filename, config):
    df = pd.read_csv(filename)

    iterations, bands, rows = get_ideal_bands(config["d1"], config["d2"], config["p1"], config["p2"],
                                              config["min_permutations"], config["max_permutations"])
    print(iterations, bands, rows)
    print("False postive chance at", config["d1"], ":", calc_prob_sim(config["d1"], bands, rows))
    print("False negative chance at", config["d2"], ":", 1 - calc_prob_sim(config["d2"], bands, rows))

    signature_matrix = build_signature_matrix(df, iterations, config["shingle_size"], config["DEBUG"])

    print("generated signature matrix")
    # https://towardsdatascience.com/understanding-locality-sensitive-hashing-49f6d1f6134

    band_val = use_bands(signature_matrix, bands)
    print(band_val)
    val = LSH_sig_matrix(signature_matrix, config["d2"])
    print(val[1])


if __name__ == '__main__':
    file = "preprocess-news_articles_small.csv"
    config = Config

    d2 = np.arange(0.65, 0.9+1e-2, step=0.05)
    itter = [4, 7, 8, 9, 12, 15, 18, 24, 200]

    for j in itter:
        for i in d2:
            config["d2"] = i
            config["d1"] = i - 0.05
            config["min_permutations"] = j
            config["max_permutations"] = j

            print('-'*20)
            print(f"d2: {i}, permutation: {j}")
            lsh(filename=file, config=config)
