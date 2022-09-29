# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import sys

SECTION_TITLE = "section_title"
TITLE = "title"
NON_CLAIM_PARAGRAPH = "non_claim_paragraph"
CLAIM_PARAGRAPH = "claim_paragraph"
BULLET = "bullet"
UL = "unordered_list"

CIT_SYMBOL = "[CIT]"
SEP_SYMBOL = "[SEP]"

SECTION_MARKUP = "Section::::"
BULLET_MARKUP = "BULLET::::"
SECTION_SEP_SYMBOL = ".:"


def remove_section_markup(text):
    text = text.strip()
    sections = [s.strip().strip(".") for s in text.split(SECTION_MARKUP)[1].split(".:")]
    return sections


def remove_bullet_markup(text):
    text = text.strip()
    new_text = text.replace(BULLET_MARKUP, "").strip()
    if new_text.startswith("-"):
        new_text = new_text[1:]
    return new_text


def split_into_paragraphs(text):
    return [t for t in text.split("\n") if t.strip()]


def get_claim_text(dp):
    claim = dp["meta"]["sentences"][-1].strip()
    return claim


def seperate_title_and_section_from_text(text):
    title, section, text = text.split(SEP_SYMBOL)
    title, section, text = title.strip(), section.strip(), text.strip()
    section = remove_section_markup(section)
    return title, section, text


def _has_section(text):
    return SECTION_MARKUP in text


def _has_bullet(text):
    return BULLET_MARKUP in text


def split_claim_paragraph(para, claim_text):
    split = para.split(claim_text)
    if len(split) > 2:
        left = split[0]
        right = claim_text.join(split[1:])
    else:
        left, right = split
    return left, right


def _parse_input(input_text, claim_text):
    title, section, text = seperate_title_and_section_from_text(input_text)

    text = text.replace(CIT_SYMBOL + ".", "").replace(CIT_SYMBOL, "")  # based on Fabio's code
    structured = [{"class": TITLE, "text": title}, {"class": SECTION_TITLE, "text": section[-1], "sections": section}]

    paragraphs = split_into_paragraphs(text)
    claim_found = False

    bullet_buffer = []

    for para in paragraphs:
        if _has_section(para):
            assert para.strip().startswith(SECTION_MARKUP)
            section = remove_section_markup(para)
            element = {"class": SECTION_TITLE, "text": section[-1], "sections": section}

        elif claim_text in para:
            left, right = split_claim_paragraph(para, claim_text)

            element = {
                "class": CLAIM_PARAGRAPH,
                "pre_text": left.strip(),
                "claim": claim_text.strip(),
                "post_text": right.strip(),
                "text": para,
            }
            claim_found = True
        else:
            element = {"class": NON_CLAIM_PARAGRAPH, "text": para.strip()}

        if _has_bullet(element["text"]):
            element = {k: remove_bullet_markup(v) for k, v in element.items()}
            bullet_buffer.append(element)
        else:
            if len(bullet_buffer) > 0:
                structured.append({"class": UL, "bullets": bullet_buffer})
                bullet_buffer = []
            structured.append(element)

    if len(bullet_buffer) > 0:
        structured.append({"class": UL, "bullets": bullet_buffer})

    if not claim_found:
        return None

    return structured


def get_bad_data_recovery_parse(input_text, claim_text):
    title, section, text = seperate_title_and_section_from_text(input_text)

    text = text.replace(CIT_SYMBOL + ".", "").replace(CIT_SYMBOL, "")  # based on Fabio's code
    structured = [
        {"class": TITLE, "text": title},
        {"class": SECTION_TITLE, "text": section[-1], "sections": section},
        {
            "class": CLAIM_PARAGRAPH,
            "pre_text": text.strip(),
            "claim": claim_text.strip(),
            "post_text": "",
            "text": text,
        },
    ]
    return structured


def _parse_input_into_structured_form(input_text, claim_text):

    elements = _parse_input(input_text, claim_text)
    if elements is None:  # try to recover by removing newlines:
        input_text = input_text.replace(claim_text, claim_text.replace("\n", " "))
        claim_text = claim_text.replace("\n", " ")
        elements = _parse_input(input_text, claim_text)
    if elements is None:  # cant recover, bad datapoint, try a simpler parse.
        print("Warning, A datapoint was hard to parse, fell back to simple parse, will not display well")
        elements = get_bad_data_recovery_parse(input_text, claim_text)

    elements = filter_abstracts_and_repeated_titles(elements)

    return elements


def parse_input_into_structured_form(datapoint):
    input_text = datapoint["input"]
    claim_text = get_claim_text(datapoint)
    return _parse_input_into_structured_form(input_text, claim_text)


def filter_abstracts_and_repeated_titles(elements):
    first_title = elements[1]
    remove_first_title = False
    for e in elements[2:]:
        if e.get("text", "") == first_title["text"] and e["class"] == SECTION_TITLE:
            remove_first_title = True

    if remove_first_title:
        elements = [elements[0]] + elements[2:]

    elements = [e for e in elements if e.get("text") != "Abstract"]
    return elements


if __name__ == "__main__":

    input_fi, output_fi = sys.argv[1:]

    # input_fi = './verify_wikipedia_test_data.json'
    # output_fi = './wikipedia_test_data_with_structured_data.json'
    output = []

    for line in open(input_fi):
        dp = json.loads(line)
        elements = parse_input_into_structured_form(dp)
        dp["structured_input"] = elements
        output.append(dp)

    with open(output_fi, "w") as f:
        for o in output:
            f.write(json.dumps(o) + "\n")
