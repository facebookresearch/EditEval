# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from cdifflib import CSequenceMatcher
from typing import List, Tuple, Optional


def get_opcodes(source: str, target: str, filter_equal: bool = True) -> List[Tuple[str, int, int, int, int]]:
    """Generate a set of opcodes representing the difference between `source` and `target` by means of insertions,
    deletions and replacements.

    :param source: The source text.
    :param target: The target text.
    :param filter_equal: Whether to remove operations that correspond to keeping a piece of text unchanged.
    :return: The list of opcodes. See https://docs.python.org/3/library/difflib.html for the format of these opcodes.
    """
    if source == target:
        return []
    matcher = CSequenceMatcher(None, source, target)
    opcodes = matcher.get_opcodes()
    opcodes = [(tag, i1, i2, j1, j2) for tag, i1, i2, j1, j2 in opcodes if tag != "equal" or not filter_equal]
    return opcodes


def apply_opcodes(source: str, target: str, start: int, end: int, opcodes: List[Tuple[str, int, int, int, int]]) -> str:
    """Apply a list of opcodes to a view of the source string that starts at `start` and ends at `end`. It is required
    that all `opcodes` only access parts of the source string that are within these bounds.

    :param source: The source text.
    :param target: The target text.
    :param start: The start of the substring from the source text to be considered (inclusive).
    :param end: The end of the substring from the source text to be considered (non-inclusive).
    :param opcodes: The opcodes to apply.
    :return: The modified substring, i.e. `source[start:end]` with all opcodes applied to it.
    """
    result_str = ""
    previous_right_bound = start
    for tag, i1, i2, j1, j2 in opcodes:
        result_str += source[previous_right_bound:i1] + target[j1:j2]
        previous_right_bound = i2
    result_str += source[previous_right_bound:end]
    return result_str


def pretty_print_diff(source: str, target: str, opcodes: Optional[List[Tuple[str, int, int, int, int]]] = None) -> None:
    """Pretty print the difference between a source and target string."""
    if opcodes is None:
        opcodes = get_opcodes(source, target)
    for tag, i1, i2, j1, j2 in opcodes:
        print("{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}".format(tag, i1, i2, j1, j2, source[i1:i2], target[j1:j2]))


def get_last_change_indices(source: str, target: str) -> Tuple[int, int]:
    """Get the first indices from the right at which a source and target string differ.

    :param source: The source string.
    :param target: The target string.
    :return: A tuple consisting of the index in the source and target string, respectively.
    """
    source_idx = len(source) - 1
    target_idx = len(target) - 1
    while source_idx >= 0 and target_idx >= 0:
        if source[source_idx] != target[target_idx]:
            break
        source_idx -= 1
        target_idx -= 1
    return source_idx, target_idx


def source_diffs_to_target(source: str, diffs: List[Tuple[str, str]]) -> str:
    """Convert a source string and a list of (source, target) pairs to a target text.

    :param source: The source string.
    :param diffs: The diffs, where each diff is a tuple mapping a source string that occurs exactly once in `source` to
    a target string.
    :return: The result of applying all diffs to the source.
    """

    # TODO: Handle cases where a diff causes a subsequent diff to not be unique anymore.
    # We should first identify all spans for doing replacements and then do the replacements in a subsequent step.
    for src, trg in diffs:
        assert source.count(src) == 1
        source = source.replace(src, trg)

    return source


def source_target_to_diffs(
    source: str,
    target: str,
    num_context_words: int = 1,
    enforce_unique: bool = True,
) -> List[Tuple[str, str]]:
    """Convert a source string and a target string into a list of diffs, where each diff is a mapping from a source
    string that occurs exactly once in `source` to a target string.

    :param source: The source string.
    :param target: The target string.
    :param num_context_words: The minimum number of context words to include on both the left and right of each diff.
    :param enforce_unique: Whether to enforce that the source part of each diff is unique (i.e., occurs exactly once
    in `source`).
    :return: A list of (source, target) pairs that transform `source` into `target`.
    """
    opcodes = get_opcodes(source, target, filter_equal=True)
    if not opcodes:
        return []

    # TODO: This function does a lot of different things. It should be split into multiple, smaller functions.

    bounds = []
    for tag, i1, i2, _, _ in opcodes:
        left_bound = i1

        if left_bound == len(source):
            left_bound -= 1

        remaining_words_left = num_context_words
        while left_bound >= 0 and remaining_words_left >= 0:
            if source[left_bound].isspace():
                remaining_words_left -= 1
            if remaining_words_left >= 0:
                left_bound -= 1

        left_bound += 1

        right_bound = i2
        remaining_words_right = num_context_words
        while right_bound < len(source) and remaining_words_right >= 0:
            if source[right_bound].isspace():
                remaining_words_right -= 1
                if remaining_words_right == -1:
                    if enforce_unique and not check_unique(source, left_bound, right_bound):
                        remaining_words_right += 1
            if remaining_words_right >= 0:
                right_bound += 1

        bounds.append((left_bound, right_bound))

    opcode_groups = []

    for opcode, (left_bound, right_bound) in zip(opcodes, bounds):
        previous_right_bound = opcode_groups[-1]["right"] if opcode_groups else None
        if previous_right_bound is not None and left_bound <= previous_right_bound:
            opcode_group = opcode_groups[-1]
            opcode_group["right"] = right_bound
            opcode_group["opcodes"].append(opcode)
        else:
            opcode_groups.append({"left": left_bound, "right": right_bound, "opcodes": [opcode]})

    source_target_pairs = []

    for opcode_group in opcode_groups:
        og_source = source[opcode_group["left"] : opcode_group["right"]]
        og_target = apply_opcodes(source, target, opcode_group["left"], opcode_group["right"], opcode_group["opcodes"])
        source_target_pairs.append((og_source, og_target))

    return source_target_pairs


def check_unique(text: str, left_idx: int, right_idx: int) -> bool:
    """Check whether the substring `text[left_idx:right_idx]` is unique in `text`."""
    return text.count(text[left_idx:right_idx]) == 1
