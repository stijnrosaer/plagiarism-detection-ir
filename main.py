import pandas as pd
from signatureMatrix import build_signature_matrix
from jaccard import plot, jaccard_df
from config import Config

from LSH import *


def export_to_file(candidate_pairs: list, filename: str):
    with open(filename, 'w') as file:
        for pair in candidate_pairs:
            file.write(f"{pair[0]},{pair[1]}\n")


def jaccard(filename, config):
    df = pd.read_csv(filename)

    df2 = jaccard_df(df, filename, config["shingle_size"])
    plot(df2)

def lsh(filename, config):
    df = pd.read_csv(filename)

    iterations, bands, rows = get_ideal_bands(config["d1"], config["d2"], config["p1"], config["p2"],
                                              config["min_permutations"], config["max_permutations"])
    print(f'bands: {bands}, rows: {rows}')
    print("False postive chance at", config["d1"], ":", calc_prob_sim(config["d1"], bands, rows))
    print("False negative chance at", config["d2"], ":", 1 - calc_prob_sim(config["d2"], bands, rows))

    signature_matrix = build_signature_matrix(df, iterations, config["shingle_size"], config["DEBUG"])

    # print("generated signature matrix")
    # https://towardsdatascience.com/understanding-locality-sensitive-hashing-49f6d1f6134

    band_val = use_bands(signature_matrix, bands)
    if "export" in config and config["export"] and "export_filename" in config:
        export_to_file(band_val, config["export_filename"])

    print(band_val)
    val = LSH_sig_matrix(signature_matrix, config["d2"])
    print(val[1])


if __name__ == '__main__':
    file = "preprocess-news_articles_small.csv"
    config = Config

    jaccard(file, config)


    # d2 = [0.7]
    # itter = [24]
    lsh(filename=file, config=config)

    # d2 = np.arange(0.7, 0.9 + 1e-2, step=0.05)
    # itter = [4, 7, 8, 9, 12, 15, 18, 24, 200]
    #
    # for j in itter:
    #     for i in d2:
    #         i = np.round(i, 2)
    #         config["d2"] = i
    #         config["d1"] = np.round(i - 0.05, 2)
    #         config["min_permutations"] = j
    #         config["max_permutations"] = j
    #
    #         print('-'*20)
    #         print(f"d2: {i}, permutation: {j}")
    #         lsh(filename=file, config=config)
