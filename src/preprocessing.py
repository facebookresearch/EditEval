# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import random
import nltk
from typing import List, Union

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)


def normalize_text(text: Union[str, List[str]]) -> str:
    if isinstance(text, str):
        text = text.split(" ")
    untokenized = nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(text)
    return untokenized


def tokenize_text(text: str) -> List[str]:
    return nltk.word_tokenize(text)


def randomly_delete_word(text: str) -> str:
    words = text.split()
    i = random.choice(range(len(words)))
    return " ".join(words[:i] + words[i + 1 :])


if __name__ == "__main__":
    example = "Members gather money for the funeral and help them .\n"
    example2 = "But I disagree with this opinion because often the advertisement does n't just speak about.\n"
    print(normalize_text(example2))
    print(tokenize_text(example2))
    print(randomly_delete_word(example))
