import unittest
from gamerec import analyze_binary, assign, cuped, sample_data, sample_size, srm, temporal_split


class RecTests(unittest.TestCase):
    def test_temporal(self):
        train, test = temporal_split(sample_data()[0])
        for user in {r["user"] for r in test}:
            self.assertLess(max(r["time"] for r in train if r["user"] == user), min(r["time"] for r in test if r["user"] == user))

    def test_experiment(self):
        self.assertEqual(assign("alice"), assign("alice"))
        self.assertTrue(srm(900, 100))
        self.assertFalse(srm(500, 500))
        self.assertGreater(sample_size(.1, .01), 1000)
        adjusted = cuped([1, 3, 5], [1, 2, 3])
        self.assertAlmostEqual(adjusted[0], adjusted[1])
        result = analyze_binary(10, 100, 20, 100)
        self.assertAlmostEqual(result["difference"], .1)
        self.assertLess(result["p_value"], 1)


if __name__ == "__main__":
    unittest.main()
