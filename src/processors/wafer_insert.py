# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Tuple, Set
import json
import os
from datasets import Dataset, DatasetDict
import string
from spacy.lang.en import English
from src.base_processor import ExternalProcessor
from src.parse_wikitext import (
    parse_input_into_structured_form,
    SECTION_TITLE,
    TITLE,
    NON_CLAIM_PARAGRAPH,
    CLAIM_PARAGRAPH,
    UL,
)
from src.utils import download_url

sentencizer = English()
sentencizer.add_pipe("sentencizer")


def join_non_empty(texts: List[str], types: List[int], skip: Set[int] = set(), separator: str = " "):
    filtered = []
    for te, ty in zip(texts, types):  # filter empty str and skip certain types
        if ty in skip:
            continue
        if len(te) == 0:
            continue
        filtered.append(te)
    return separator.join(filtered)


def trim_text(text: str, position: str = "begin"):
    """
    Trim incomplete sentence if any at the beginning/end of the text.
    """
    text = text.strip()
    if len(text) <= 0:
        return text
    if position == "begin":
        first_char = text[0]
        if first_char.isalpha() and first_char.islower():  # incomplete sentence
            sents = list(sentencizer(text).sents)[1:]  # drop the first sentence
            text = " ".join(map(str, sents))
    elif position == "end":
        end_char = text[-1]
        if end_char not in string.punctuation:  # incomplete sentence
            sents = list(sentencizer(text).sents)[:-1]  # drop the last sentence
            text = " ".join(map(str, sents))
    else:
        raise ValueError
    return text


def trim_element(element: Dict, position: str = "begin"):
    """
    In-place trim incomplete sentence if any at the beginning/end of the element depending on its type.
    """
    cls = element["class"]

    if cls == UL:  # nested
        if position == "begin":  # trim the first bullet
            return trim_element(element["bullets"][0], position=position)
        elif position == "end":  # trim the last bullet
            return trim_element(element["bullets"][-1], position=position)

    if cls in {SECTION_TITLE, TITLE}:  # no trim needed
        return
    if cls == NON_CLAIM_PARAGRAPH:
        element["text"] = trim_text(element["text"], position=position)
    elif cls == CLAIM_PARAGRAPH:
        if position == "begin":
            element["pre_text"] = trim_text(element["pre_text"], position=position)
        elif position == "end":
            element["post_text"] = trim_text(element["post_text"], position=position)
    else:
        raise NotImplementedError


def linearize_element(
    element: Dict, method: str = "markdown", replace_claim_with: str = None
) -> Tuple[List[str], List[int]]:
    """
    Linearize a single element depending on its type.
    """
    assert method in {"markdown", "plain", "wikipedia"}
    cls = element["class"]
    types = None
    if cls == SECTION_TITLE:
        if method == "markdown":
            text = [f'{"#" * (i + 2)} {t}\n' for i, t in enumerate(element["sections"])]
        elif method == "plain":
            text = [f"{t}\n" for t in element["sections"]]
        elif method == "wikipedia":
            text = [f'{"=" * (i + 2)} {t} {"=" * (i + 2)}\n' for i, t in enumerate(element["sections"])]
    elif cls == NON_CLAIM_PARAGRAPH:
        t = element["text"].strip()
        text = []
        if len(t):
            text.append(t + "\n")
    elif cls == CLAIM_PARAGRAPH:
        text = []
        types = []
        for chunk in ["pre_text", "claim", "post_text"]:
            t = element[chunk].strip()
            if len(t) == 0:
                continue
            if chunk == "claim":
                text.append(t if replace_claim_with is None else replace_claim_with)
                types.append(1)
            else:
                text.append(t)
                types.append(0)
        if len(text):
            text[-1] = text[-1] + "\n"
    elif cls == UL:
        text = []
        types = []
        for b in element["bullets"]:
            te, ty = linearize_element(b, replace_claim_with=replace_claim_with)
            if len(te) == 0:
                continue
            if method == "markdown":
                te[0] = f"- {te[0]}"
            elif method == "plain":
                pass
            elif method == "wikipedia":
                te[0] = f"* {te[0]}"
            text.extend(te)
            types.extend(ty)
    else:
        raise NotImplementedError
    if types is None:
        types = [0] * len(text)
    assert len(text) == len(types)
    return text, types


def linearize_elements(
    elements: List[Dict],
    method: str = "markdown",
    replace_claim_with: str = None,
    start_ind: int = None,
    end_ind: int = None,
) -> Tuple[List[str], List[int]]:
    """
    Linearize multiple elements.
    """
    start_ind = start_ind or 1  # the first is always title
    end_ind = end_ind or len(elements)
    text: List[str] = []
    types: List[int] = []
    for ind in range(start_ind, end_ind):
        te, ty = linearize_element(elements[ind], method=method, replace_claim_with=replace_claim_with)
        text.extend(te)
        types.extend(ty)
    return text, types


def keep_elements(elements: List[Dict], left: str = "all", right: str = "all") -> Tuple[int, int]:
    """
    Keep elements specified by left and right.
    """
    # all: keep everything
    # multiple: keep multiple sections (if any) but remove incomplete sections
    # single: only keep the current section that contains the claim
    assert left in {"all", "multiple", "single"} and right in {"all", "single"}

    start_ind = 1  # the first is always title
    end_ind = len(elements)

    if left == right == "all":
        return start_ind, end_ind

    saw_claim = saw_section_title = False
    for ind in range(1, len(elements)):
        ele = elements[ind]
        cls = ele["class"]
        assert cls != TITLE, "duplicated title"
        if cls == CLAIM_PARAGRAPH or (
            cls == UL and CLAIM_PARAGRAPH in {b["class"] for b in ele["bullets"]}
        ):  # found claim potentially nested in UL
            saw_claim = True
            continue
        if not saw_claim and cls == SECTION_TITLE:
            if left == "multiple":
                if saw_section_title == False:
                    start_ind = ind
            elif left == "single":
                start_ind = ind
            saw_section_title = True
        elif saw_claim and cls == SECTION_TITLE:
            if right == "multiple":
                end_ind = ind
            elif right == "single":
                end_ind = ind
                break
            saw_section_title = True
    assert saw_claim, "no claim included"
    return start_ind, end_ind


def preprocess(example, linearize: str, keep_context: str = "all-all", debug: bool = False):
    """
    Trim incomplete sentences, keep necessary context, and linearize.
    """
    # special cases of claims (based on Jane's code)
    if example["meta"]["sentences"][-1] in {"C", "B"}:
        del example["meta"]["sentences"][-1]

    # parse raw text into a structured format
    elements = parse_input_into_structured_form(example)

    # get title
    assert elements[0]["class"] == TITLE
    title = elements[0]["text"]

    # get claim
    claim = example["meta"]["sentences"][-1]

    # trim incomplete sentence for the first/last element
    trim_element(elements[1], position="begin")
    trim_element(elements[-1], position="end")

    # keep necessary context
    keep_left, keep_right = keep_context.split("-")
    start_ind, end_ind = keep_elements(elements, left=keep_left, right=keep_right)

    # linearization
    texts, types = linearize_elements(
        elements, method=linearize, replace_claim_with=None, start_ind=start_ind, end_ind=end_ind
    )
    source = join_non_empty(texts, types, skip={1}, separator=" ")
    target = join_non_empty(texts, types, skip={}, separator=" ")
    if source == target:
        raise Exception(f"Error in {example['id']}: source (w/o claim) and target (w/ claim) shouldn't be the same")
    if not source.strip() or not target.strip():
        raise Exception(f"empty input or output")

    if debug:
        print(example["id"])
        print(target)
        input()

    return title, claim, source, target, texts, types


class WAFERInsertProcessor(ExternalProcessor):
    def __init__(self, raw_path: str):
        super().__init__()
        self.dataset_name: str = "wafer_insert"
        self.raw_path: str = raw_path
        self._needs_normalize: bool = False
        self.split_mapping: Dict[str, str] = {}
        self.features_mapping: Dict[str, str] = {"output": "edits"}
        self.task_name: str = "insertion"
        self.use_retrieved_documents: bool = False
        self.host_link = "https://dl.fbaipublicfiles.com/wafer_insert/"

    def download_and_process(self) -> DatasetDict:
        if not os.path.exists(self.raw_path):
            download_url(
                os.path.join(self.host_link, "wafer-dev-kiltweb.jsonl"),
                self.raw_path,
                filename="wafer-dev-kiltweb.jsonl",
            )
            download_url(
                os.path.join(self.host_link, "wafer-test-kiltweb.jsonl"),
                self.raw_path,
                filename="wafer-test-kiltweb.jsonl",
            )

        dataset_dict = {}
        for split in ["test", "dev"]:
            wafer_path = self.raw_path + "/wafer-" + split + "-kiltweb.jsonl"
            dataset_dict[split] = self.parse_from_json(path=wafer_path)
        self.dataset = DatasetDict(dataset_dict)

    def remove_claim(self, x: Dict) -> str:
        claim = x["meta"]["sentences"][-1]

        # handle exceptions
        if claim == "C":
            claim = x["meta"]["sentences"][-2]
        if claim == "B":
            claim = x["meta"]["sentences"][-2]
        if claim == "Sri Sairam College of Engineering":
            claim = "Sri Sairam College of Engineering[CIT]"
        if claim == "Ciaotou District":
            claim = "Ciaotou District[CIT]"
        if claim == "Jurassic World: The Ride":
            claim = "Jurassic World: The Ride[CIT]"

        offset = x["input"].find(claim)
        if offset > 0:
            ixp = x["input"]
            pre_text = ixp[:offset]
            post_text = ixp[offset + len(claim) :].replace("[CIT].", "").replace("[CIT]", "")
            new_input = pre_text + post_text
            x["meta"]["claim_offset"] = offset
            x["meta"]["claim"] = claim
        else:
            print("ERROR")
            print(x["input"])
            print(x["meta"]["sentences"])

        if claim in new_input:
            return None, None

        return new_input, claim

    def parse_from_json(
        self,
        path: str = None,
    ) -> Dataset:
        dataset = {}
        try:
            open_file = open(path, "r")
        except FileNotFoundError:
            print("file {} does not exist".format(path))
        else:
            with open_file:
                dataset = {}

                dataset["input"] = []
                dataset["output"] = []
                dataset["retrieved_documents"] = []
                dataset["retrieved_documents_non_gold"] = []
                dataset["meta"] = []
                dataset["id"] = []
                dataset["title"] = []
                dataset["wiki_id"] = []
                dataset["chunks"] = []

                lines = open_file.readlines()

                for line in lines:
                    example = json.loads(line)

                    # TODO: any exception not handled?
                    # new_input, claim = self.remove_claim(example)
                    try:
                        title, claim, source, target, texts, types = preprocess(
                            example, linearize="wikipedia", keep_context="all-all"
                        )
                        assert title == example["meta"]["wikipedia_title"]
                        example["meta"]["claim"] = claim

                        dataset["input"].append(source)
                        dataset["output"].append(target)
                        dataset["wiki_id"].append(example["meta"]["wikipedia_id"])
                        dataset["title"].append(example["meta"]["wikipedia_title"])
                        dataset["chunks"].append({"texts": texts, "types": types})

                        provenance = []
                        for o in example["output"]:
                            try:
                                provenance.extend(o["provenance"])
                            except:
                                continue

                        dataset["retrieved_documents"].append(provenance)  # take all provenances
                        dataset["retrieved_documents_non_gold"].append(
                            example["retrieved_documents"]
                        )  # retrieved documents
                        dataset["meta"].append(example["meta"])
                        dataset["id"].append(example["id"])
                    except KeyboardInterrupt as e:
                        raise e
                    except Exception as e:  # skip problematic cases
                        pass

            return Dataset.from_dict(dataset)


def add_retrieved_documents(raw_file: str, ret_file: str, out_file: str):
    """
    Add the retrieved documents from the file specified by ret_file to the file specified raw_file.
    """
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(raw_file, "r") as rfin, open(ret_file, "r") as efin, open(out_file, "w") as fout:
        for l in rfin:
            raw_example = json.loads(l)
            ret_example = json.loads(efin.readline())
            assert raw_example["id"] == ret_example["id"]
            raw_example["retrieved_documents"] = ret_example["retrieved_documents"]
            fout.write(json.dumps(raw_example) + "\n")


if __name__ == "__main__":
    # wafer_raw_path = "/checkpoint/fabiopetroni/EditEval/WAFER_insert"
    # wafer_raw_path = "/checkpoint/janeyu/side_eval_datasets/raw/wafer_insert"
    # wafer_raw_path = "/checkpoint/janeyu/side_eval_datasets/raw/wafer_insert/reranker_outputs"
    # e = WAFERInsertProcessor(wafer_raw_path)
    # e.download_and_process()
    # print(e)

    for split in ["test", "dev"]:
        add_retrieved_documents(
            raw_file=f"/checkpoint/janeyu/side_eval_datasets/raw/wafer_insert/reranker_outputs/wafer-{split}-kiltweb.jsonl",
            ret_file=f"/checkpoint/fabiopetroni/EditEval/WAFER_insert/wafer_insert-{split}.jsonl",
            out_file=f"/checkpoint/zhengbao/side_eval_datasets/raw/wafer_insert/reranker_outputs/wafer-{split}-kiltweb.jsonl",
        )
