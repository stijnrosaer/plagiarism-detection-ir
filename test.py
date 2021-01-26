import unittest

from LSH import *
from jaccard import *


class TestJaccard(unittest.TestCase):
    # Tests our jaccord scoring method

    def test_simple_score(self):
        doc1 = ["the", "great", "wall"]
        doc2 = ["the", "bad", "wall"]
        score = jaccard_index(doc1, doc2)
        self.assertAlmostEqual(score, 0.5)

    def test_nothing_equal(self):
        doc1 = "i have a bad feeling about this".split()
        doc2 = "the evil robber".split()
        score = jaccard_index(doc1, doc2)
        self.assertEqual(score, 0)

    def test_exact_duplicate(self):
        doc1 = "i have a bad feeling about this".split()
        doc2 = "i have a bad feeling about this".split()
        score = jaccard_index(doc1, doc2)
        self.assertEqual(score, 1)

    # TODO test with small file?


class TestIdealIterRowsBands(unittest.TestCase):

    def test_one_band(self):
        iter, bands, rows = get_ideal_bands(0.7, 0.9, 0.75, 0.80, 1, 1)
        self.assertEqual(iter, 1)
        self.assertEqual(bands, 1)
        self.assertEqual(rows, 1)

    def test_find_correct_bands(self):
        iter, bands, rows = get_ideal_bands(0.5, 0.8, 0.470, 0.99, 100, 100)
        self.assertEqual(iter, 100)
        self.assertEqual(bands, 20)
        self.assertEqual(rows, 5)

    def test_find_correct_iter_and_bands(self):
        iter, bands, rows = get_ideal_bands(0.5, 0.8, 0.470, 0.99, 60, 200)
        self.assertEqual(iter, 100)
        self.assertEqual(bands, 20)
        self.assertEqual(rows, 5)


    def test_cursus_values(self):
        # test the probability calculations used by get_ideal_bands
        bands = 20
        rows = 5
        self.assertAlmostEqual(calc_prob_sim(0.2, bands, rows), 0.006, 3)
        self.assertAlmostEqual(calc_prob_sim(0.3, bands, rows), 0.047, 3)
        self.assertAlmostEqual(calc_prob_sim(0.4, bands, rows), 0.186, 3)
        self.assertAlmostEqual(calc_prob_sim(0.5, bands, rows), 0.470, 3)
        self.assertAlmostEqual(calc_prob_sim(0.6, bands, rows), 0.802, 3)
        self.assertAlmostEqual(calc_prob_sim(0.7, bands, rows), 0.975, 3)
        self.assertAlmostEqual(calc_prob_sim(0.8, bands, rows), 0.9996, 4)


if __name__ == "__main__":
    unittest.main()