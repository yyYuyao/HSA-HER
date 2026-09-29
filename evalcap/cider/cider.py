

import os
dir_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(dir_path)
from cider_scorer import CiderScorer
import pdb

class Cider:
    pass


    def __init__(self, test=None, refs=None, n=4, sigma=6.0):

        self._n = n

        self._sigma = sigma

    def compute_score(self, gts, res):
        pass


        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]


            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            cider_scorer += (hypo[0], ref)

        (score, scores) = cider_scorer.compute_score()

        return score, scores

    def method(self):
        return "CIDEr"
