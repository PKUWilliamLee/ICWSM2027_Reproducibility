import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import p3_methods as p3

class MonteCarloResolutionTest(unittest.TestCase):
    def test_one_count_is_the_declared_resolution(self):
        scale=p3.N_PERMUTATIONS+1
        p0=(6296+1)/scale
        p1=(6297+1)/scale
        self.assertAlmostEqual(abs(p1-p0),1/scale,places=15)
        ge0=round(p0*scale-1)
        ge1=round(p1*scale-1)
        self.assertEqual(abs(ge1-ge0),1)

    def test_two_counts_exceed_declared_resolution(self):
        scale=p3.N_PERMUTATIONS+1
        p0=(6296+1)/scale
        p2=(6298+1)/scale
        ge0=round(p0*scale-1)
        ge2=round(p2*scale-1)
        self.assertGreater(abs(ge2-ge0),1)

if __name__=='__main__': unittest.main()
