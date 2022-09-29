"""
(Note: This script computes sentence-level GLEU score.)
This script calculates the GLEU score of a sentence, as described in
our ACL 2015 paper, Ground Truth for Grammatical Error Correction Metrics
by Courtney Napoles, Keisuke Sakaguchi, Matt Post, and Joel Tetreault.

This script was adapted from gleu.py by Courtney Napoles.
<https://github.com/keisks/jfleg/blob/master/eval/gleu.py/>
"""

import math
import numpy as np
import scipy.stats
import random
from collections import Counter
from itertools import islice
from typing import List
from src.preprocessing import tokenize_text


class GLEU:
    def __init__(self, n=4):
        self.order = n

    # def load_hypothesis_sentence(self, hypothesis):
    #     """load ngrams for a single sentence"""
    #     self.hlen = len(hypothesis)
    #     self.this_h_ngrams = [self.get_ngram_counts(hypothesis, n)
    #                           for n in range(1, self.order + 1)]

    # def load_sources(self, spath):
    #     """load n-grams for all source sentences"""
    #     self.all_s_ngrams = [[self.get_ngram_counts(line.split(), n)
    #                           for n in range(1, self.order + 1)]
    #                          for line in open(spath)]

    def load_hypothesis_sentence(self, hypothesis: str) -> None:
        """load ngrams for a single sentence"""
        self.hlen = len(hypothesis)
        self.this_h_ngrams = [self.get_ngram_counts(hypothesis, n) for n in range(1, self.order + 1)]

    def load_inputs(self, inputs: List[List[str]]) -> None:
        """load n-grams for all input sentences"""
        self.all_s_ngrams = [[self.get_ngram_counts(input, n) for n in range(1, self.order + 1)] for input in inputs]

    def load_references(self, references: List[List[List[str]]]):
        """load n-grams for all references"""

        # transposed_refs = [list(x) for x in zip(*references)]
        # for ref in transposed_refs:
        #     for i, ref_example in enumerate(ref):
        #         self.refs[i].append(ref_example.split())
        #         self.rlens[i].append(len(ref_example.split()))

        self.refs = references
        self.rlens = [[len(ref_i) for ref_i in refs] for refs in self.refs]
        self.num_refs = len(self.refs[0])

        # count number of references each n-gram appear sin
        self.all_rngrams_freq = [Counter() for i in range(self.order)]

        self.all_r_ngrams = []
        for refset in self.refs:
            all_ngrams = []
            self.all_r_ngrams.append(all_ngrams)

            for n in range(1, self.order + 1):
                ngrams = self.get_ngram_counts(refset[0], n)
                all_ngrams.append(ngrams)

                for k in ngrams.keys():
                    self.all_rngrams_freq[n - 1][k] += 1

                for ref in refset[1:]:
                    new_ngrams = self.get_ngram_counts(ref, n)
                    for nn in new_ngrams.elements():
                        if new_ngrams[nn] > ngrams.get(nn, 0):
                            ngrams[nn] = new_ngrams[nn]

    # def get_ngram_counts(self, sentence, n):
    #     """get ngrams of order n for a untokenized sentence"""
    #     return Counter(zip(*[islice(sentence.split(" "), i, None) for i in range(n)]))

    def get_ngram_counts(self, sentence, n):
        """get ngrams of order n for a tokenized sentence"""
        return Counter([tuple(sentence[i : i + n]) for i in range(len(sentence) + 1 - n)])

    def get_ngram_diff(self, a, b):
        """returns ngrams in a but not in b"""
        diff = Counter(a)
        for k in set(a) & set(b):
            del diff[k]
        return diff

    def normalization(self, ngram, n):
        """get normalized n-gram count"""
        return float(self.all_rngrams_freq[n - 1][ngram] / len(self.rlens[0]))

    def gleu_stats(self, i, r_ind=None):
        """
        Collect BLEU-relevant statistics for a single hypothesis/reference pair.
        Return value is a generator yielding:
        (c, r, numerator1, denominator1, ... numerator4, denominator4)
        Summing the columns across calls to this function on an entire corpus
        will produce a vector of statistics that can be used to compute GLEU
        """
        hlen = self.hlen
        rlen = self.rlens[i][r_ind]

        yield hlen
        yield rlen

        for n in range(1, self.order + 1):
            h_ngrams = self.this_h_ngrams[n - 1]
            s_ngrams = self.all_s_ngrams[i][n - 1]
            r_ngrams = self.get_ngram_counts(self.refs[i][r_ind], n)

            s_ngram_diff = self.get_ngram_diff(s_ngrams, r_ngrams)

            yield max([sum((h_ngrams & r_ngrams).values()) - sum((h_ngrams & s_ngram_diff).values()), 0])

            yield max([hlen + 1 - n, 0])

    def gleu(self, stats, smooth=False):
        """Compute GLEU from collected statistics obtained by call(s) to gleu_stats"""
        # smooth 0 counts for sentence-level scores
        if smooth:
            stats = [s if s != 0 else 1 for s in stats]
        if len(list(filter(lambda x: x == 0, stats))) > 0:
            return 0
        (c, r) = stats[:2]
        log_gleu_prec = sum([math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]) / 4
        return math.exp(min([0, 1 - float(r) / c]) + log_gleu_prec)

    def get_gleu_stats(self, scores):
        """calculate mean and confidence interval from all GLEU iterations"""
        mean = np.mean(scores)
        std = np.std(scores)
        ci = scipy.stats.norm.interval(0.95, loc=mean, scale=std)
        return mean
        # return ["%f" % mean, "%f" % std, "(%.3f,%.3f)" % (ci[0], ci[1])]

    def run_iterations(self, hypotheses, num_iterations=500, n=4, debug=False, per_sent=True):
        """run specified number of iterations of GLEU, choosing a reference
        for each sentence at random"""
        # first generate a random list of indices, using a different seed
        # for each iteration
        indices = []
        for j in range(num_iterations):
            random.seed(j * 101)
            indices.append([random.randint(0, self.num_refs - 1) for i in range(len(hypotheses))])

        if debug:
            print("===== Sentence-level scores =====")
            print("SID Mean Stdev 95%CI GLEU")

        iter_stats = [[0 for i in range(2 * n + 2)] for j in range(num_iterations)]

        for i, hypothesis in enumerate(hypotheses):

            self.load_hypothesis_sentence(hypothesis)

            # we are going to store the score of this sentence for each ref
            # so we don't have to recalculate them 500 times
            stats_by_ref = [None for r in range(self.num_refs)]

            for j in range(num_iterations):
                ref = indices[j][i]
                this_stats = stats_by_ref[ref]

                if this_stats is None:
                    this_stats = [s for s in self.gleu_stats(i, r_ind=ref)]

                    stats_by_ref[ref] = this_stats

                iter_stats[j] = [sum(scores) for scores in zip(iter_stats[j], this_stats)]

            if debug or per_sent:
                # sentence-level GLEU is the mean GLEU of the hypothesis
                # compared to each reference
                for r in range(self.num_refs):
                    if stats_by_ref[r] is None:
                        stats_by_ref[r] = [s for s in self.gleu_stats(i, r_ind=r)]
                if debug:
                    print(i, " ".join(self.get_gleu_stats([self.gleu(stats, smooth=True) for stats in stats_by_ref])))
        if not per_sent:
            score = self.get_gleu_stats([self.gleu(stats) for stats in iter_stats])
            return score

        return np.average(scores)


if __name__ == "__main__":

    gleu_calculator = GLEU(n=4)

    references = [
        ["New technology has been introduced to society .", "New technology has been introduced into the society."],
        [
            "One possible outcome is that an environmentally-induced reduction in motorization levels in richer countries will outweigh any rise in motorization levels in poorer countries.",
            "One possible outcome is that an environmentally-induced reduction in motorization levels in the richer countries will outweigh any rise in motorization levels in the poorer countries.",
        ],
        [
            "Every person needs to know a bit about math , science , art , literature and history in order to stand out in society.",
            "Every person needs to know a bit about math , sciences , arts , literature and history in order to stand out in society.",
        ],
        [
            "While the travel company will most likely show them some interesting sites in order for their customers to advertise for their company to their family and friends , it is highly unlikely that the company will tell about the sites that were not included in the tour -- for example due to entrance fees that would make the total package price overly expensive.",
            "While the travel company will most likely show them some interesting sites in order to advertise their company to their family and friends , it is highly unlikely that the company will advertise the sites that were not included in the tour -- for example , due to entrance fees that would make the total package price overly expensive.",
        ],
        [
            "A disadvantage is that parking their cars is very difficult.",
            "A disadvantage is that parking their car is very difficult.",
        ],
    ]

    sources = [
        "Old and new technology has been introduced to the society.",
        "One possible outcome is that an environmentally-induced reduction in motorization levels in the richer countries will outweigh any rise in motorization levels in the poorer countries .",
        "Every person needs to know a bit about math , sciences , arts , literature and history in order to stand out in society .",
        "While the travel company will most likely show them some interesting sites in order for their customers to advertise for their company to their family and friends , it is highly unlikely , that the company will tell about the sites that were not included in the tour -- for example due to entrance fees that would make the total package price overly expensive .",
        "Disadvantage is parking their car is very difficult .",
    ]

    source_tokens = [tokenize_text(x) for x in sources]
    predictions_tokens = source_tokens
    reference_tokens = [[tokenize_text(x) for x in ref] for ref in references]

    gleu_calculator.load_inputs(source_tokens)
    gleu_calculator.load_references(reference_tokens)
    # print(gleu_calculator.all_s_ngrams)
    # print(gleu_calculator.all_rngrams_freq)
    # print(gleu_calculator.all_r_ngrams)

    # for i, prediction in enumerate(predictions_tokens):
    # print(prediction)
    if len(reference_tokens[0]) == 1:
        print("There is one reference. NOTE: GLEU is not computing the confidence interval.")
        print(gleu_calculator.run_iterations(hypotheses=predictions_tokens, num_iterations=500, per_sent=False))
    else:
        print(gleu_calculator.run_iterations(hypotheses=predictions_tokens, num_iterations=500, per_sent=False))
