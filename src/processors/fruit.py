# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from src.base_processor import ExternalProcessor
from datasets import DatasetDict, Dataset
import tensorflow as tf
import os
import re
from typing import Tuple, Dict, List
import ast


class FRUITProcessor(ExternalProcessor):
    def __init__(self, raw_path):
        super().__init__()
        self.dataset_name = "fruit"
        self.raw_path = raw_path
        self._needs_normalize = False
        self.split_mapping = {}
        self.features_mapping = {"inputs": "input", "targets": "edits"}
        self.task_name = "info_addition"
        self.host_link = "gs://gresearch/FRUIT/dataset"

    def clean_title(self, title: str) -> str:
        return title.replace("_", " ")

    def get_index_to_sentence_dict(self, full_context: str) -> Dict[str, str]:
        """This function takes the input sentences to the FRUIT system and parses it as a dict of index -> sentence.
        This is needed for reformatting the output.
        :param full_context: String where sentences are proceeded by their index in square brackets [i].

        Sample:
        full_context: "[0] Mike McMeeken (born 10 May 1994) is an English rugby league footballer who plays as a  forward for the Castleford Tigers in the Super League. [1] McMeeken has also represented England at international level, playing in two games at the 2017 World Cup. [2] He started his career in the Super League with the London Broncos, also playing on loan in League 1 at the London Skolars before joining the Tigers."
        returns: {'[0]': 'Mike McMeeken (born 10 May 1994) is an English rugby league footballer who plays as a  forward for the Castleford Tigers in the Super League.', '[1]': 'McMeeken has also represented England at international level, playing in two games at the 2017 World Cup.', '[2]': 'He started his career in the Super League with the London Broncos, also playing on loan in League 1 at the London Skolars before joining the Tigers.'}
        """
        i = 0
        index_to_sentence = {}
        regex = "\[[0-9]+\]"
        splits = re.split(f"({regex})", full_context)
        prev_sentence_index = -1
        while i < len(splits):
            if bool(re.search(f"^{regex}$", splits[i])):
                # Need to check that the matched group is actually a FRUIT indexing. Sometimes [i] will happen naturally in text.
                if splits[i][1:-1] == str(prev_sentence_index + 1):
                    index_to_sentence[splits[i]] = splits[i + 1].strip()
                    i += 1
                    prev_sentence_index += 1
                # If the matched group is not a FRUIT indexing, continue appending it to the previous sentence.
                elif prev_sentence_index != -1:
                    index_to_sentence["[" + str(prev_sentence_index) + "]"] += splits[i]
            i += 1
        return index_to_sentence

    def get_index_to_context_dict(self, full_context: str) -> Dict[str, str]:
        """This function takes the sentences from the "retrieved document" (aka the contexts) and parses them as a dict
        of index -> sentence. These sentences are used for updating the input sentences. This is needed for reformatting the output.
        :param full_context: String where sentences are proceeded by their index in parentheses (i).

        Sample:
        full_context: "(0) 2022_New_Caledonian_independence_referendum Background In accordance with the Nouméa Accord, New Caledonians are allowed up to three referendums on independence; the first in 2018, then two more in 2020 and 2022 if the previous ones had not resulted in independence, but one-third of members of the Congress of New Caledonia voted for another one. (1) 2022_New_Caledonian_independence_referendum INTRODUCTION The poll will be the third to be held under the terms of the Nouméa Accord, following votes in 2018 and 2020, in which independence was rejected by 56.7% and 53.3% respectively. (2) 2020–2021_New_Caledonian_protests Background Two referendums relating the question of independence were held, and a third could follow, as per the Nouméa Accord. (3) 2022_New_Caledonian_independence_referendum Background The Nouméa Accord, signed 5 May 1998 by the French government and the main independence and anti-independence parties, set in motion a 20-year transition period that transferred certain powers to the local government and laid the groundwork for an independence referendum in 2018."
        returns: {'(0)': '2022_New_Caledonian_independence_referendum Background In accordance with the Nouméa Accord, New Caledonians are allowed up to three referendums on independence; the first in 2018, then two more in 2020 and 2022 if the previous ones had not resulted in independence, but one-third of members of the Congress of New Caledonia voted for another one.', '(1)': '2022_New_Caledonian_independence_referendum INTRODUCTION The poll will be the third to be held under the terms of the Nouméa Accord, following votes in 2018 and 2020, in which independence was rejected by 56.7% and 53.3% respectively.', '(2)': '2020–2021_New_Caledonian_protests Background Two referendums relating the question of independence were held, and a third could follow, as per the Nouméa Accord.', '(3)': '2022_New_Caledonian_independence_referendum Background The Nouméa Accord, signed 5 May 1998 by the French government and the main independence and anti-independence parties, set in motion a 20-year transition period that transferred certain powers to the local government and laid the groundwork for an independence referendum in 2018.'}
        """
        i = 0
        prev_sentence_index = -1
        index_to_context = {}
        regex = "\([0-9]+\)"
        splits = re.split(f"({regex})", full_context)
        while i < len(splits):
            # Need to check that the matched group is actually a FRUIT indexing. Sometimes (i) will happen naturally in text.
            if (
                bool(re.search(f"^{regex}$", splits[i]))
                and (splits[i][1:-1]) == str(prev_sentence_index + 1)
                and bool(
                    re.search(
                        r"^[^;!?\[\]\"\<\>\|\~\`\@\#\$\%\^\&\*\+\{\}\=]+$",
                        splits[i + 1].strip().split(" ")[0],
                    )
                )
            ):
                index_to_context[splits[i]] = splits[i + 1].strip()
                prev_sentence_index += 1
                i += 1
            # If the matched group is not a FRUIT indexing, continue appending it to the previous sentence.
            elif prev_sentence_index != -1:
                index_to_context["(" + str(prev_sentence_index) + ")"] += splits[i]
            i += 1
        return index_to_context

    def transform_input(self, input: str) -> Tuple[str, Dict[str, str], Dict[str, str], Dict[str, str]]:
        """This function transforms the input so that are no indices and transforms the retrieved documents (contexts)
        to be the same dict formatting as wafer (a list of dictionaries with url, title, and text).
        :param input: String containing both the original input and the context sentences

        Sample:
        input: "[0] A sound module is an electronic musical instrument without a human-playable interface such as a piano-style musical keyboard. [1] Sound modules have to be operated using an externally connected device, which is often a MIDI controller, of which the most common type is the musical keyboard (although wind controllers, guitar controllers and electronic drum pads are also used). [2] Controllers are devices that provide the human-playable interface and which may or may not produce sounds of its own. [3] Another common way of controlling a sound module is through a sequencer, which is computer hardware or software designed to record and play back control information for sound-generating hardware (e.g., a DJ may program a bassline and use the sound module to produce the sound). [4] Connections between sound modules, controllers, and sequencers are generally made with MIDI (Musical Instrument Digital Interface), which is a standardized protocol designed for this purpose, which includes special ports (jacks) and cables. [5] Sound modules may use any number of technologies to produce their sounds. [6] A sound module may be an analog or digital synthesizer, a sampler, or a rompler. [7] Electronic drum modules are sound modules which specialize in drumkit and percussion sounds. [8] Drum modules may be triggered by external trigger pads or pickups attached to an acoustic drum as well as through MIDI controller pads. [9] Drum modules are distinguished from drum machines through their lack of dedicated on-board triggers and lack of an integrated sequencer. [10] Sound modules are often rack-mountable, but might also have a table-top form factor, particularly when the intended user is a DJ or record producer. [11] The height of a sound module is often described in rack units ("U") or unit. [12] Small sound modules are mostly 1U in height, the larger models a multiplication e.g. [13] 2U or 3U. [14] Despite the name, "sound module", most sound modules do not produce any audible sound until their output is plugged into a keyboard amplifier and loudspeaker or a PA system. [15] Because most electronic instruments are designed in a modularized way, manufacturers often release a sound module version of their fully integrated instruments. [16] For example, the 1980s-era DX-7 synthesizer/keyboard was also sold as a standalone "sound module", the TX-7. [CONTEXT] (0) Accordion History Some 2010s-era accordions may incorporate MIDI sensors and circuitry, enabling the accordion to be plugged into a synth module and produce accordion sounds or other synthesized instrument sounds, such as piano or organ. (1) Digital_accordion External sound modules Accordionists who add aftermarket solid state contacts or spring contacts and a MIDI system to an acoustic accordion need to plug the MIDI out into an external sound module. (2) Yamaha_YMF278 Applications It is used in various Yamaha electronic musical instruments, including the Yamaha MU5 and TG-100 sound modules, Yamaha Portasound electronic keyboards (PSS-51, PSR-200, PSR-210, PSR-215, PSR-300, PSR-310, PSR-400, PSR-410, PSR-500, PSR-510 and PSR-600), QR-10 music accompaniment player, and QY-20 music workstation. (3) The_X-Files Production - Music After attempting to craft the theme with different sound effects, Snow used a Proteus 2 rackmount sound module with a preset sound called "Whistling Joe". (4) Windows_Sound_System INTRODUCTION WSS featured support for up to 16-bit, 48 kHz digital sampling, beyond the capabilities of the popular contemporary Sound Blaster Pro, although it was less frequently supported than Sound Blaster and Gravis sound cards, as well as Roland sound cards, daughterboards, and sound modules. (5) Finger_vibrato Keyboard instruments Some 2010s and 2020s MIDI controllers and synthesizer keyboards have pressure or aftertouch sensors which sense if the player is continuing to press down a key after the initial striking; on some synth module patches (sounds), continued pressure on a key triggers an electronic vibrato effect, in imitation of the expressive vocal, bowed strings, or wind technique of adding vibrato to a held note."
        transformed_input: "A sound module is an electronic musical instrument without a human-playable interface such as a piano-style musical keyboard. Sound modules have to be operated using an externally connected device, which is often a MIDI controller, of which the most common type is the musical keyboard (although wind controllers, guitar controllers and electronic drum pads are also used). Controllers are devices that provide the human-playable interface and which may or may not produce sounds of its own. Another common way of controlling a sound module is through a sequencer, which is computer hardware or software designed to record and play back control information for sound-generating hardware (e.g., a DJ may program a bassline and use the sound module to produce the sound). Connections between sound modules, controllers, and sequencers are generally made with MIDI (Musical Instrument Digital Interface), which is a standardized protocol designed for this purpose, which includes special ports (jacks) and cables. Sound modules may use any number of technologies to produce their sounds. A sound module may be an analog or digital synthesizer, a sampler, or a rompler. Electronic drum modules are sound modules which specialize in drumkit and percussion sounds. Drum modules may be triggered by external trigger pads or pickups attached to an acoustic drum as well as through MIDI controller pads. Drum modules are distinguished from drum machines through their lack of dedicated on-board triggers and lack of an integrated sequencer. Sound modules are often rack-mountable, but might also have a table-top form factor, particularly when the intended user is a DJ or record producer. The height of a sound module is often described in rack units ("U") or unit. Small sound modules are mostly 1U in height, the larger models a multiplication e.g. 2U or 3U. Despite the name, "sound module", most sound modules do not produce any audible sound until their output is plugged into a keyboard amplifier and loudspeaker or a PA system. Because most electronic instruments are designed in a modularized way, manufacturers often release a sound module version of their fully integrated instruments. For example, the 1980s-era DX-7 synthesizer/keyboard was also sold as a standalone "sound module", the TX-7."
        transformed_contexts: [{'url': 'https://en.wikipedia.org/wiki/Accordion', 'title': 'Accordion', 'text': 'History Some 2010s-era accordions may incorporate MIDI sensors and circuitry, enabling the accordion to be plugged into a synth module and produce accordion sounds or other synthesized instrument sounds, such as piano or organ.'}, {'url': 'https://en.wikipedia.org/wiki/Digital_accordion', 'title': 'Digital_accordion', 'text': 'External sound modules Accordionists who add aftermarket solid state contacts or spring contacts and a MIDI system to an acoustic accordion need to plug the MIDI out into an external sound module.'}, {'url': 'https://en.wikipedia.org/wiki/Yamaha_YMF278', 'title': 'Yamaha_YMF278', 'text': 'Applications It is used in various Yamaha electronic musical instruments, including the Yamaha MU5 and TG-100 sound modules, Yamaha Portasound electronic keyboards (PSS-51, PSR-200, PSR-210, PSR-215, PSR-300, PSR-310, PSR-400, PSR-410, PSR-500, PSR-510 and PSR-600), QR-10 music accompaniment player, and QY-20 music workstation.'}, {'url': 'https://en.wikipedia.org/wiki/The_X-Files', 'title': 'The_X-Files', 'text': 'Production - Music After attempting to craft the theme with different sound effects, Snow used a Proteus 2 rackmount sound module with a preset sound called "Whistling Joe".'}, {'url': 'https://en.wikipedia.org/wiki/Windows_Sound_System', 'title': 'Windows_Sound_System', 'text': 'INTRODUCTION WSS featured support for up to 16-bit, 48 kHz digital sampling, beyond the capabilities of the popular contemporary Sound Blaster Pro, although it was less frequently supported than Sound Blaster and Gravis sound cards, as well as Roland sound cards, daughterboards, and sound modules.'}, {'url': 'https://en.wikipedia.org/wiki/Finger_vibrato', 'title': 'Finger_vibrato', 'text': 'Keyboard instruments Some 2010s and 2020s MIDI controllers and synthesizer keyboards have pressure or aftertouch sensors which sense if the player is continuing to press down a key after the initial striking; on some synth module patches (sounds), continued pressure on a key triggers an electronic vibrato effect, in imitation of the expressive vocal, bowed strings, or wind technique of adding vibrato to a held note.'}]
        """
        input_context = input.split("[CONTEXT]")
        index_to_sentence = self.get_index_to_sentence_dict(input_context[0])
        transformed_input = " ".join(index_to_sentence.values())

        transformed_contexts = []
        index_to_context = self.get_index_to_context_dict(input_context[1])

        for ind, context in index_to_context.items():
            url_context_split = context.split(" ", maxsplit=1)
            if len(url_context_split) > 1:
                nc = {
                    "url": "https://en.wikipedia.org/wiki/" + url_context_split[0],
                    "title": self.clean_title(url_context_split[0]),
                    "text": url_context_split[1],
                }
                transformed_contexts.append(nc)
                index_to_context[ind] = nc
            else:
                index_to_context[ind] = {"url": None, "title": None, "text": context}

        return (
            transformed_input,
            transformed_contexts,
            index_to_sentence,
            index_to_context,
        )

    # Function moves the citation (e.g., (0)) from the front of the claim to the end of the claim. FRUIT puts them in front.
    # Function also changes citation formatting from parentheses () to brackets [].
    def move_citation_from_front_to_back(
        self, output: str, index_to_sentence: Dict[str, str], index_to_context: Dict[str, str]
    ) -> str:
        """This function transforms the output so that the citations are at the end of the sentence.
        :param output: String containing the citations in the front, which is how FRUIT formats their output.

        Sample:
        output: "(0) (1) (2) Mike McMeeken (born 10 May 1994) is an English rugby league footballer who plays as a forward for the Catalans Dragons in the Super League. [1] He started his career in the Super League with the London Broncos, also playing on loan in League 1 at the London Skolars before joining the Tigers. (0) (1) (2) He joined Catalans Dragons in December 2020, ahead of the 2021 season."
        returns: "Mike McMeeken (born 10 May 1994) is an English rugby league footballer who plays as a forward for the Catalans Dragons in the Super League. (0) (1) (2) [1] He started his career in the Super League with the London Broncos, also playing on loan in League 1 at the London Skolars before joining the Tigers.    He joined Catalans Dragons in December 2020, ahead of the 2021 season. (0) (1) (2)"
        """
        parentheses_regex = "\([0-9]+\)"
        bracket_regex = "\[[0-9]+\]"
        citation_split = re.split(f"({parentheses_regex}|{bracket_regex})", output)
        elements: List[str] = []
        element_types: List[int] = []  # 0 text, 1 sentence index, 2 context index

        def filter_merge_elements(elements: List[str], element_types: List[int]) -> Tuple[List[str], List[int]]:
            """Trim, remove empty elements, merge consecutive texts."""
            assert len(elements) == len(element_types)
            new_eles: List[str] = []
            new_ele_types: List[int] = []
            for i, (e, et) in enumerate(zip(elements, element_types)):
                # trim text based on the surrounding elements
                if et == 0:
                    # trim left if it's the first element or the previous element is not text
                    if len(new_ele_types) == 0 or new_ele_types[-1] != 0:
                        e = e.lstrip()
                    # trim right if it's the last element or the next element is not text
                    if i == len(element_types) - 1 or element_types[i + 1] != 0:
                        e = e.rstrip()
                # always trim non-text elements
                else:
                    e = e.strip()
                # skip empty elements
                if not e:
                    continue
                # merge with previous text
                if et == 0 and len(new_ele_types) and new_ele_types[-1] == 0:
                    new_eles[-1] += e
                else:
                    new_eles.append(e)
                    new_ele_types.append(et)
            return new_eles, new_ele_types

        def group_contexts(elements: List[str], element_types: List[int]) -> List[Dict]:
            """Group consecutive context indices and associate them with the next element."""
            context_buffer: List[str] = []
            grouped: List[Dict] = []
            for e, et in zip(elements, element_types):
                if et == 0:
                    grouped.append(
                        {
                            "type": "text",
                            "content": e,
                            "context": context_buffer,
                            "context_real": [
                                index_to_context[ctx] for ctx in context_buffer if ctx in index_to_context
                            ],
                        }
                    )
                    context_buffer = []
                elif et == 1:
                    grouped.append({"type": "copy", "content": index_to_sentence[e], "content_index": e})
                elif et == 2:
                    context_buffer.append(e)
                else:
                    raise ValueError
            return grouped

        def linearize(grouped: List[Dict], context_position: str = "back") -> str:
            buffer: List[str] = []
            for e in grouped:
                if e["type"] == "copy":
                    buffer.append(e["content"])
                elif e["type"] == "text":  # add context
                    if context_position == "front":
                        buffer.append(" ".join(e["context"] + [e["content"]]))
                    elif context_position == "back":
                        buffer.append(" ".join([e["content"]] + e["context"]))
                    else:
                        raise NotImplementedError
                else:
                    raise ValueError
            return " ".join(buffer)

        for split_output in citation_split:
            # A context index included in index_to_context.
            if re.search(f"^{parentheses_regex}$", split_output) and split_output in index_to_context:
                elements.append(split_output)
                element_types.append(2)
            # A sentence index included in index_to_sentence.
            elif re.search(f"^{bracket_regex}$", split_output) and split_output in index_to_sentence:
                elements.append(split_output)
                element_types.append(1)
            # Normal text
            else:
                elements.append(split_output)
                element_types.append(0)

        elements, element_types = filter_merge_elements(elements, element_types)
        grouped = group_contexts(elements, element_types)
        new_string = linearize(grouped)

        return new_string, grouped

    def transform_output(self, output: str, index_to_sentence: Dict[str, str], index_to_context: Dict[str, str]) -> str:
        """This function transforms replaces the sentence (input) indices with their original text and the citations
        (context indices) are moved to the end of the sentence.
        :param output: String containing the updated text in EditEval format.

        FRUIT output formatting:
        Example: "(0) (1) Aoraki / Mount Cook, often referred to as Mount Cook Village, is located within New Zealand's Aoraki / Mount Cook National Park at the end of ... [2] [3] [4]"
        This means that the first sentence is updated using the context item (0) and (1). '[2] [3] [4]' means that the these sentences are copied directly from the source article.

        Sample:
        output: "(0) (1) (2) Mike McMeeken (born 10 May 1994) is an English rugby league footballer who plays as a forward for the Catalans Dragons in the Super League. [1] He started his career in the Super League with the London Broncos, also playing on loan in League 1 at the London Skolars before joining the Tigers. (0) (1) (2) He joined Catalans Dragons in December 2020, ahead of the 2021 season."
        returns: "Mike McMeeken (born 10 May 1994) is an English rugby league footballer who plays as a forward for the Catalans Dragons in the Super League. (0) (1) (2) McMeeken has also represented England at international level, playing in two games at the 2017 World Cup. He started his career in the Super League with the London Broncos, also playing on loan in League 1 at the London Skolars before joining the Tigers.    He joined Catalans Dragons in December 2020, ahead of the 2021 season. (0) (1) (2)"
        """
        output, grouped = self.move_citation_from_front_to_back(output, index_to_sentence, index_to_context)
        for index, sentence in index_to_sentence.items():
            output = output.replace(index, sentence)
        return output, grouped

    def parse_from_tfrecord(self, split_name: str) -> Dict[str, List]:
        complete_path = os.path.join(self.raw_path, f"dataset/{split_name}")
        filenames = [os.path.join(complete_path, x) for x in os.listdir(complete_path) if "tfrecords" in x]
        raw_dataset = tf.data.TFRecordDataset(filenames)

        dataset = {}
        features = ["inputs", "targets", "retrieved_documents", "wiki_id", "title", "raw"]

        for feature in features:
            if feature not in dataset:
                dataset[feature] = []

        for raw_record in raw_dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            for i in range(len(example.features.feature["inputs"].bytes_list.value)):

                raw_input = example.features.feature["inputs"].bytes_list.value[i].decode()
                raw_output = example.features.feature["targets"].bytes_list.value[i].decode()
                _id = example.features.feature["id"].int64_list.value[i]

                (
                    transformed_input,
                    transformed_contexts,
                    index_to_sentence,
                    index_to_context,
                ) = self.transform_input(raw_input)

                transformed_output, grouped = self.transform_output(raw_output, index_to_sentence, index_to_context)

                dataset["inputs"].append(transformed_input)
                dataset["retrieved_documents"].append(transformed_contexts)
                dataset["targets"].append(transformed_output)
                dataset["wiki_id"].append(_id)
                dataset["raw"].append({"output": grouped})

                try:
                    wiki_titles = list(
                        ast.literal_eval(
                            example.features.feature["generatable_surfaces"].bytes_list.value[i].decode()
                        ).keys()
                    )
                    dataset["title"].append(self.clean_title(wiki_titles[0].encode().decode()))
                except:
                    dataset["title"].append("")
        return dataset

    def download_and_process(self) -> None:

        if not os.path.exists(self.raw_path):
            os.system(f"mkdir {self.raw_path}")
            os.system(f"gsutil cp -R {self.host_link} {self.raw_path}")

        train_dataset = self.parse_from_tfrecord("train")
        dev_dataset = self.parse_from_tfrecord("test")
        test_dataset = self.parse_from_tfrecord("gold_test")

        dataset_dict = {}
        dataset_dict["train"] = Dataset.from_dict(train_dataset)
        dataset_dict["dev"] = Dataset.from_dict(dev_dataset)
        dataset_dict["test"] = Dataset.from_dict(test_dataset)

        self.dataset = DatasetDict(dataset_dict)


if __name__ == "__main__":
    e = FRUITProcessor("/checkpoint/janeyu/side_eval_datasets/raw/fruit")
    e.download_and_process()
    print(e.dataset)
