import numpy as np
from config import Config


def transpose(m):
    """
    transpose a 2-dimensional matrix, making the outer list the inner list
    :param m: the matrix to transpose
    :return: the transposed matrix
    """
    trans = [[m[j][i][1] for j in range(len(m))] for i in range(len(m[0]))]
    return trans


def calc_signature_similarity(sig1: list, sig2: list) -> float:
    """
    :param sig1: signature of doc1
    :param sig2: signature of doc2
    :return: similarity of signatures; 0 < sim < 1
    """
    # Check that nothing bad happened
    if len(sig1) != len(sig2):
        raise Exception("compared signatures not of same size")

    # count the amount of values that are equal
    val = 0
    for i in range(len(sig1)):
        if sig1[i] == sig2[i]:
            val += 1

    # divide by amount to get a percentage
    return val/float(len(sig1))


def LSH_sig_matrix(signature_matrix: list, threshhold: float):
    """
    Calculates the similarity exactly based on the signature matrix, only used to debug our code
    :param signature_matrix: signature matrix of all containing all signatures of all docs
    :param threshold: 
    :return: (list containing all pairs and their similarity, list containing only the pairs that exceed the threshold)
    """
    # Initialize variables
    fullist = []
    exceed_list = []
    signature_matrix = transpose(signature_matrix)  # TODO cleanup?

    # loop over every pair of signatures
    for i in range(len(signature_matrix)):
        for j in range(i+1, len(signature_matrix), 1):
            # Calculate their similarity
            similarity = calc_signature_similarity(
                signature_matrix[i], signature_matrix[j])

            # Append to list(s)
            fulval = ((i, j), similarity)
            fullist.append(fulval)
            if similarity > threshhold:
                exceed_list.append(fulval)

    return (fullist, exceed_list)


def use_bands(signature_matrix: list, nr_bands: int) -> list:
    """
    Generates candidate pairs for near-duplicate detection
    :param signature_matrix: signature matrix
    :param nr_bands: amount of bands to use, must be a divider of the amount of cols(signature_matrix)
    :return: list of tuples
    """
    bands = []
    signature_matrix = transpose(signature_matrix)

    # rows are signatures
    # create bands by splitting up signatures
    for signature in signature_matrix:
        s = np.array_split(signature, nr_bands)
        bands.append(s)

    candidate_pairs = []

    # check every band pair
    for i in range(len(bands)):
        for j in range(i + 1, len(bands), 1):

            # loop over all corresponding partial signatures in these two bands
            for k in range(len(bands[i])):
                if list(bands[i][k]) == list(bands[j][k]):  # check in same bucket
                    candidate_pairs.append((i, j))
                    break       # stop search if in same bucket because this pair is already a candidate
    return candidate_pairs


def get_number_of_bands(threshold, nr_iters, thresh_maximum=1.0):
    """
    :param thershold: desired threshold = step point
    :param nr_iters: curent amount of iterations
    :param thresh_maximum: maximum threshold = max step point
    :returns: (threshold, bands, rows)
    """
    diff = float("inf")
    val = -1
    for b in range(1, nr_iters + 1):
        if nr_iters % b == 0 or b == 1:
            r = nr_iters/b
            t = pow(1.0 / b, 1.0 / r)   # approximate step point
            if abs(threshold - t) < diff and (t <= thresh_maximum):
                diff = abs(threshold - t)
                val = (t, b, r)

    return val


def debug_nr_bandz(b, r):
    """
    Prints the chance of a candidate pair at 0% 10% 20% ... 90%
    given b bands and r rows.
    Used for debugging purposes.
    """
    for i in range(10):
        s = i * 0.1
        t = 1-pow(1-pow(s, r), b)
        print("prob", format(s, ".2f"), ":", t)


def calc_prob_sim(sim: float, bands: int, rows: int) -> float:
    """
    Calculates the chance of creating a candidate pair given that they have a similarity of sim
    and we use bands bands and rows rows
    """
    prob = 1-pow(1-pow(sim, rows), bands)
    return prob


def get_ideal_bands(d1: float, d2: float, p1: float, p2: float, eps=0.01, printing=False) -> tuple:
    """
    :param d1:  upper bound of false positives
    :param d2:  lower bound on false negatives
    :param p1:  max chance for false positive before d1
    :param p2:  max chance for false negeative after d2
    :param eps: max error between p1 and calculated p1 and between p2 and calc p2
    :return: (nr_bands, nr_rows)
    """
    min_iterations = Config["min_permutations"]
    max_iterations = Config["max_permutations"]

    bands_best, rows_best, iter_best = 0, 0, 0
    error_best = float('inf')

    # Loop over all possible amounts of iterations, this allows us to select a broader range of
    # bands and rows
    for nr_iters in range(min_iterations, max_iterations+1):
        # check all possible bands
        for bands in range(1, nr_iters+1):
            # filter bands that cannot divide perfectly
            if nr_iters % bands == 0 or bands == 1:
                # compute nr of rows per band
                rows = nr_iters//bands
                # caclulate the probability of generating a candidate pair with sim d1 and d2
                p1_calc = calc_prob_sim(d1, bands, rows)
                p2_calc = calc_prob_sim(d2, bands, rows)

                # check if this is better than what we had before
                diff = abs(p1_calc - p1) + abs(p2_calc - p2)
                if diff < error_best:
                    error_best = diff
                    bands_best, rows_best, iter_best = bands, rows, nr_iters
                # stop if we have reached our desired max error
                if error_best < eps:
                    break

    if printing:
        print(bands_best, rows_best, iter_best)
        threshold = pow(1.0 / bands_best, 1.0 / rows_best)
        print(threshold)
    return iter_best, bands_best, rows_best
