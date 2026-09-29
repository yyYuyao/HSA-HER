import json
import re
import sys
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
import os
from tqdm import tqdm
import numpy as np
import time
from .utils import (
    gather_processes,
    make_prompt,
    clean_responses,
    compute_largest_cluster,
    flatten_values_lists_of_list_dicts_to_dict,
)


def truncate_to_max_len(sentences, max_len):
    return [" ".join(sentence.split()[:max_len]) for sentence in sentences]


def load_report_pairs(refs_json_path, hyps_json_path):
    with open(refs_json_path, "r", encoding="utf-8") as refs_file:
        refs_data = json.load(refs_file)
    with open(hyps_json_path, "r", encoding="utf-8") as hyps_file:
        hyps_data = json.load(hyps_file)

    common_ids = sorted(set(refs_data) & set(hyps_data))
    if len(common_ids) == 0:
        raise ValueError("No overlapping report ids were found between refs and hyps.")

    if len(common_ids) != len(refs_data) or len(common_ids) != len(hyps_data):
        print(
            f"Warning: using {len(common_ids)} overlapping report ids "
            f"(refs={len(refs_data)}, hyps={len(hyps_data)})."
        )

    refs = []
    hyps = []
    for report_id in common_ids:
        ref_value = refs_data[report_id]
        hyp_value = hyps_data[report_id]

        if isinstance(ref_value, list):
            ref_value = ref_value[0] if len(ref_value) > 0 else ""
        if isinstance(hyp_value, list):
            hyp_value = hyp_value[0] if len(hyp_value) > 0 else ""

        refs.append(ref_value)
        hyps.append(hyp_value)

    return refs, hyps


class Inferer:
    def __init__(
        self,
        dataset=None,
        model=None,
        tokenizer=None,
        model_name="",
        output_dir=".",
        num_examples=None,
        batch_size=10,
        max_length=2048,
    ):

        self.dataset = Dataset.from_dict(
            {"reference": dataset[0], "prediction": dataset[1]}
        )
        self.process_data()

        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.num_examples = num_examples

        self.output_dir = output_dir

        self.batch_size = batch_size

        self.prompts = None
        self.completions = None
        self.green_scores = None
        self.error_counts = None

        self.categories = [
            "Clinically Significant Errors",
            "Clinically Insignificant Errors",
            "Matched Findings",
        ]

        self.sub_categories = [
            "(a) False report of a finding in the candidate",
            "(b) Missing a finding present in the reference",
            "(c) Misidentification of a finding's anatomic location/position",
            "(d) Misassessment of the severity of a finding",
            "(e) Mentioning a comparison that isn't in the reference",
            "(f) Omitting a comparison detailing a change from a prior study",
        ]

        self.max_length = max_length

    def process_data(self):
        print("Processing data...making prompts")

        def promting(examples):
            return {
                "prompt": [
                    make_prompt(r, p)
                    for r, p in zip(examples["reference"], examples["prediction"])
                ]
            }

        self.dataset = self.dataset.map(promting, batched=True)
        print("Done.")

    @torch.inference_mode()
    def infer(self):


        dataset_dist = self.dataset

        print("==== Beginning Inference ====")
        local_completions = []
        local_references = []

        for batch in tqdm(
            dataset_dist.iter(batch_size=self.batch_size),
            total=len(dataset_dist) // self.batch_size,
        ):
            local_references.extend(batch["prompt"])
            local_completions.extend(self.get_response(batch))


        self.completions = local_completions
        self.prompts = local_references

        print("==== End Inference ====")

        if len(self.completions) != len(self.prompts):
            print("length of prompts and completions are not equal!")

        self.process_results()

    def tokenize_batch_as_chat(self, batch):

        batch = [
            self.tokenizer.apply_chat_template(
                i, tokenize=False, add_generation_prompt=True
            )
            for i in batch["conv"]
        ]


        batch = self.tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(int(os.environ.get("LOCAL_RANK", 0)))

        return batch

    def get_response(self, batch):


        assert "prompt" in batch.keys(), "prompt is not in batch keys"

        batch["conv"] = [
            [
                {"from": "human", "value": i},
            ]
            for i in batch["prompt"]
        ]

        batch = self.tokenize_batch_as_chat(batch)

        outputs = self.model.generate(
            **batch,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            max_new_tokens=self.max_length,
        )


        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


        response_list = []
        if isinstance(responses, list):
            for response in responses:
                response = clean_responses(response)
                response_list.append(response)
        else:
            responses = clean_responses(responses)
            response_list.append(responses)

        return response_list

    def process_results(self):

        self.green_scores = [
            self.compute_green(response) for response in self.completions
        ]
        self.error_counts = pd.DataFrame(
            [self.compute_error_count(response) for response in self.completions],
            columns=self.sub_categories + ["Matched Findings"],
        )

        results_df = pd.DataFrame(
            {
                "reference": self.dataset["reference"],
                "predictions": self.dataset["prediction"],
                "evaluation": self.completions,
                "green": self.green_scores,
                **self.error_counts,
            }
        )
        path = self.output_dir + f"/results_{self.model_name}.csv"
        os.makedirs(self.output_dir, exist_ok=True)
        print("Saving generated response to prompt to ", path)
        results_df.to_csv(path, index=False)

        self.compute_summary()

        return results_df

    def compute_error_count(self, response):
        _, sig_errors = self.parse_error_counts(response, self.categories[0])

        matched_findings, _ = self.parse_error_counts(response, self.categories[2])
        return sig_errors + [matched_findings]

    def compute_green(self, response):

        sig_present, sig_errors = self.parse_error_counts(response, self.categories[0])

        matched_findings, _ = self.parse_error_counts(response, self.categories[2])


        if matched_findings == 0:
            return 0

        if (
            sig_present is None or matched_findings is None
        ):
            return None

        return matched_findings / (matched_findings + sum(sig_errors))

    def parse_error_counts(self, text, category, for_reward=False):

        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )


        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, text, re.DOTALL)


        sum_counts = 0
        sub_counts = [0 for i in range(6)]


        if not category_text:
            if for_reward:

                return None, None
            return sum_counts, sub_counts

        if category_text.group(1).startswith("No"):
            return sum_counts, sub_counts

        if category == "Matched Findings":
            counts = re.findall(r"^\b\d+\b(?=\.)", category_text.group(1))
            if len(counts) > 0:
                sum_counts = int(counts[0])
            return sum_counts, sub_counts

        else:

            sub_categories = [s.split(" ", 1)[0] + " " for s in self.sub_categories]

            matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))


            if len(matches) == 0:
                matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
                sub_categories = [
                    f"({i})" + " " for i in range(1, len(self.sub_categories) + 1)
                ]

            for position, sub_category in enumerate(sub_categories):

                for match in range(len(matches)):
                    if matches[match].startswith(sub_category):

                        count = re.findall(r"(?<=: )\b\d+\b(?=\.)", matches[match])
                        if len(count) > 0:

                            sub_counts[position] = int(count[0])
            return sum(sub_counts), sub_counts

    def parse_error_sentences(self, response, category):
        pass


        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )
        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, response, re.DOTALL)
        sub_category_dict_sentences = {}
        for sub_category in self.sub_categories:
            sub_category_dict_sentences[sub_category] = []

        if not category_text:
            return sub_category_dict_sentences
        if category_text.group(1).startswith("No"):
            return sub_category_dict_sentences

        if category == "Matched Findings":
            return (
                category_text.group(1).rsplit(":", 1)[-1].rsplit(".", 1)[-1].split(";")
            )

        matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

        if len(matches) == 0:
            matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
            self.sub_categories = [
                f"({i})" + " " for i in range(1, len(self.sub_categories) + 1)
            ]

        for position, sub_category in enumerate(self.sub_categories):

            for match in range(len(matches)):
                if matches[match].startswith(sub_category):

                    sentences_list = (
                        matches[match].rsplit(":", 1)[-1].split(".", 1)[-1].split(";")
                    )
                    sub_category_dict_sentences[self.sub_categories[position]] = (
                        sentences_list
                    )

        return sub_category_dict_sentences

    def compute_sentences(self, response):

        return self.parse_error_sentences(response, self.categories[0])

    def get_representative_sentences(self, responses):
        list_sentences = []
        for i in responses:
            sentences = self.compute_sentences(i)
            list_sentences.append(sentences)

        dict_sentences = flatten_values_lists_of_list_dicts_to_dict(list_sentences)

        result_sentences_dict = {}

        for i in self.sub_categories:
            sentences = dict_sentences[i]
            sentences = [i for i in sentences if i.strip() != ""]
            _, sentences_of_largest_cluster = compute_largest_cluster(sentences)
            result_sentences_dict[i] = sentences_of_largest_cluster

        return result_sentences_dict

    def compute_accuracy(self, responses):
        pass


        counts = []
        for response in responses:
            _, sig_errors = self.parse_error_counts(response, self.categories[0])
            counts.append(sig_errors)

        counts = np.array(counts)

        dict_acc = {}
        for i in range(len(self.sub_categories)):
            error_counts = counts[:, i]

            accuracy = np.mean(error_counts == 0)
            dict_acc[self.sub_categories[i]] = accuracy

        return dict_acc

    def compute_summary(self):
        pass


        print("Computing summary ...")
        representative_sentences = self.get_representative_sentences(self.completions)
        accuracies = self.compute_accuracy(self.completions)

        summary = f"[Summary]: Green average {np.mean(self.green_scores)} and standard variation {np.std(self.green_scores)} \n [Clinically Significant Errors Analyses]: <accuracy>. <representative error>\n\n (a) False report of a finding in the candidate: {accuracies[self.sub_categories[0]]}. \n {representative_sentences[self.sub_categories[0]]} \n\n (b) Missing a finding present in the reference: {accuracies[self.sub_categories[1]]}. \n {representative_sentences[self.sub_categories[1]]} \n\n (c) Misidentification of a finding's anatomic location/position: {accuracies[self.sub_categories[2]]}. \n {representative_sentences[self.sub_categories[2]]} \n\n (d) Misassessment of the severity of a finding: {accuracies[self.sub_categories[3]]}. \n {representative_sentences[self.sub_categories[3]]} \n\n (e) Mentioning a comparison that isn't in the reference: {accuracies[self.sub_categories[4]]}. \n {representative_sentences[self.sub_categories[4]]} \n\n (f) Omitting a comparison detailing a change from a prior study: {accuracies[self.sub_categories[5]]}. {representative_sentences[self.sub_categories[5]]}."

        print(summary)


def compute(model_name, refs, hyps, output_dir=".", file_name=''):
    chat_template = "{% for message in messages %}\n{% if message['from'] == 'human' %}\n{{ '<|user|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'system' %}\n{{ '<|system|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'gpt' %}\n{{ '<|assistant|>\n'  + message['value'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map={"": "cuda:{}".format(torch.cuda.current_device())},
        torch_dtype=torch.float16,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_eos_token=True,
        use_fast=True,
        trust_remote_code=True,
        padding_side="left",
    )
    tokenizer.chat_template = chat_template
    tokenizer.pad_token = tokenizer.eos_token

    inferer = Inferer(
        dataset=[refs, hyps],
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        model_name=file_name,
        batch_size=1,
    )

    t = time.time()

    inferer.infer()

    t = time.time() - t
    print("Seconds per example: ", t / len(refs))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python -m green_score.green "
            "<refs_json> <hyps_json> [output_dir] [file_name] [model_name]"
        )

    refs_json_path = sys.argv[1]
    hyps_json_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "."
    file_name = sys.argv[4] if len(sys.argv) > 4 else ""
    model_name = (
        sys.argv[5] if len(sys.argv) > 5 else "StanfordAIMI/GREEN-radllama2-7b"
    )

    refs, hyps = load_report_pairs(refs_json_path, hyps_json_path)

    if file_name == "":
        file_name = os.path.splitext(os.path.basename(hyps_json_path))[0]

    compute(model_name, refs, hyps, output_dir=output_dir, file_name=file_name)
