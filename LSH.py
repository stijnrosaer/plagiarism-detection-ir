import numpy as np


def transpose(m):
    """
    transpose a 2-dimensional matrix, making the outer list the inner list
    :param m: the matrix to transpose
    :return: the transposed matrix
    """
    trans = [[m[j][i][1] for j in range(len(m))] for i in range(len(m[0]))]
    return trans


def calc_signature_similarity(sig1, sig2):
    if len(sig1) != len(sig2):
        raise Exception("compared signatures not of same size")
    val = 0
    for i in range(len(sig1)):
        if sig1[i] == sig2[i]:
            val += 1

    return val/float(len(sig1))


def LSH_sig_matrix(signature_matrix, threshhold):
    fullist = []
    exceed_list = []
    signature_matrix = transpose(signature_matrix)
    for i in range(len(signature_matrix)):
        for j in range(i+1, len(signature_matrix), 1):
            similarity = calc_signature_similarity(
                signature_matrix[i], signature_matrix[j])
            fulval = ((i, j), similarity)
            fullist.append(fulval)
            if similarity > threshhold:
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
                    possible_plags.append((i, j))
                    break
    return possible_plags


def get_number_of_bands(thershold, nr_iters, thresh_maximum=1.0):
    diff = float("inf")
    val = -1
    for b in range(1, nr_iters + 1):
        if nr_iters % b == 0 or b == 1:
            r = nr_iters/b
            t = pow(1.0 / b, 1.0 / r)
            if abs(thershold - t) < diff and (t <= thresh_maximum):
                diff = abs(thershold - t)
                val = (t, b, r)

    return val


def debug_nr_bandz(b, r):
    for i in range(10):
        s = i*0.1
        t = 1-pow(1-pow(s, r), b)
        print("prob", format(s, ".2f"), ":", t)


def calc_prob_sim(sim: float, bands: int, rows: int) -> float:
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
    # Config vars
    min_iterations = 1
    max_iterations = 500

    bands_best, rows_best, iter_best = 0, 0, 0
    error_best = float('inf')

    first = True
    for nr_iters in range(min_iterations, max_iterations):
        for bands in range(1, nr_iters+1):
            if nr_iters % bands == 0 or bands == 1:
                rows = nr_iters/bands
                p1_calc = calc_prob_sim(d1, bands, rows)
                p2_calc = calc_prob_sim(d2, bands, rows)
                if first:
                    first = False
                    bands_best, rows_best, iter_best = bands, rows, nr_iters
                else:
                    diff = abs(p1_calc - p1) + abs(p2_calc - p2)
                    if diff < error_best:
                        error_best = diff
                        bands_best, rows_best, iter_best = bands, rows, nr_iters
                    if diff < eps:
                        break

    if printing:
        print(bands_best, rows_best, iter_best)
        threshold = pow(1.0 / bands_best, 1.0 / rows_best)
        print(threshold)
    return iter_best, bands_best, rows_best
