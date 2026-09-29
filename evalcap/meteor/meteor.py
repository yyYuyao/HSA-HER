

from __future__ import division

import atexit
import logging
import os
import re
import subprocess
import sys
import threading

import psutil


METEOR_JAR = 'meteor-1.5.jar'


def enc(s):
    return s.encode('utf-8')


def dec(s):
    return s.decode('utf-8')


class Meteor:

    def __init__(self):

        self.lock = threading.Lock()

        mem = '1G'
        mem_available_G = psutil.virtual_memory().available / 1E9
        if mem_available_G < 2:
            logging.warning("There is less than 2GB of available memory.\n"
                            "Will try with limiting Meteor to 1GB of memory but this might cause issues.\n"
                            "If you have problems using Meteor, "
                            "then you can try to lower the `mem` variable in meteor.py")
            mem = '1G'

        meteor_cmd = ['java',
                           '--add-opens', 'java.base/java.lang=ALL-UNNAMED',
                           '--add-opens', 'java.base/java.io=ALL-UNNAMED',
                           '-jar', '-Xmx2G',
                           METEOR_JAR, '-', '-', '-spec', 'en', '-norm']
        env = os.environ.copy()
        env['LC_ALL'] = "C"
        self.meteor_p = subprocess.Popen(meteor_cmd,
                                         cwd=os.path.dirname(os.path.abspath(__file__)),
                                         env=env,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)

        atexit.register(self.close)

    def close(self):
        with self.lock:
            if self.meteor_p:
                self.meteor_p.kill()
                self.meteor_p.wait()
                self.meteor_p = None


        if atexit is not None and atexit.unregister is not None:
            atexit.unregister(self.close)

    def compute_score(self, gts, res):
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        with self.lock:
            for i in imgIds:
                assert (len(res[i]) == 1)
                stat = self._stat(res[i][0], gts[i])
                eval_line += ' ||| {}'.format(stat)

            self.meteor_p.stdin.write(enc('{}\n'.format(eval_line)))
            self.meteor_p.stdin.flush()
            for i in range(0, len(imgIds)):
                v = self.meteor_p.stdout.readline()
                try:
                    scores.append(float(dec(v.strip())))
                except:
                    sys.stderr.write("Error handling value: {}\n".format(v))
                    sys.stderr.write("Decoded value: {}\n".format(dec(v.strip())))
                    sys.stderr.write("eval_line: {}\n".format(eval_line))


                    raise
            score = float(dec(self.meteor_p.stdout.readline()).strip())
        self.close()
        return score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):

        hypothesis_str = hypothesis_str.replace('|||', '')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        score_line = re.sub(r'\s+', ' ', score_line)
        self.meteor_p.stdin.write(enc(score_line))
        self.meteor_p.stdin.write(enc('\n'))
        self.meteor_p.stdin.flush()
        return dec(self.meteor_p.stdout.readline()).strip()

    def _score(self, hypothesis_str, reference_list):
        with self.lock:

            hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
            score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
            self.meteor_p.stdin.write(enc('{}\n'.format(score_line)))
            self.meteor_p.stdin.flush()
            stats = dec(self.meteor_p.stdout.readline()).strip()
            eval_line = 'EVAL ||| {}'.format(stats)

            self.meteor_p.stdin.write(enc('{}\n'.format(eval_line)))
            self.meteor_p.stdin.flush()
            score = float(dec(self.meteor_p.stdout.readline()).strip())


            score = float(dec(self.meteor_p.stdout.readline()).strip())
        return score

    def __del__(self):
        self.close()
