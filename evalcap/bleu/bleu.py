

import os
dir_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(dir_path)
from bleu_scorer import BleuScorer


class Bleu:
    def __init__(self, n=4):

        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res, verbose=0):

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]


            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)


        score, scores = bleu_scorer.compute_score(option='closest', verbose=verbose)


        return score, scores

    def method(self):
        return "Bleu"
