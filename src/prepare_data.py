# Cleans Lama Trex dataset
# Creates set of objects per relation, and stores this as JSON file
# REL -> list of o_neg objects

import logging
import wandb
import json, os, sys, re
import pandas as pd
import numpy as np
import time

import datasets
from datasets import load_dataset, Dataset
from pathlib import Path
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from transformers import set_seed
import language_tool_python
from utils import (
    trex_preprocessing,
    hypernymy_preprocessing,
    # hyponymy_postprocessing,
    popQA_preprocessing,
    # popQA_postprocessing,
    create_paraphrase_df,
    classify,
    boxplots,
    init_wandb,
    parse_args,
    get_df_overview_stats,
)

# add parent folder to path to import ProbLM
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from dev.GenLM import GenLM
from src.utils.logging import configure_logging


def get_s_o_true(i, df_rel):
    o = df_rel.iloc[i]["obj_label"]
    s = df_rel.iloc[i]["sub_label"]

    return s, o


def get_o_neg(r, df):
    df_rel = df[df["template"] == r]
    o_neg_set = df_rel["obj_label"].unique().tolist()

    return o_neg_set


def get_s_o_true(i, df_rel):
    o = df_rel.iloc[i]["obj_label"]
    s = df_rel.iloc[i]["sub_label"]

    return s, o


def fill_template(r, s, o):
    # fill into template, and remove leading whitespace pefore punctuation

    template = r.replace("[X]", s).replace("[Y]", o).replace(" .", ".").strip()

    return template


def create_s_o_permutations(
    o_neg_sets_sampled,
    df_processed,
    paraphrased_templates,
    r2id,
    args,
):
    # create sets of r,s,o and r,s,{o_neg}
    # add r_s_id (instance id): key where each original s,o pair comes from
    df_processed["r_s_id"] = list(range(len(df_processed)))

    # create permutations of s, o, + s, o_neg pairs

    data = []
    for instance in df_processed["r_s_id"].tolist():
        (
            s,
            o,
            r_orig,
        ) = (
            df_processed.iloc[instance]["sub_label"],
            df_processed.iloc[instance]["obj_label"],
            df_processed.iloc[instance]["template"],
        )

        if s == None or o == None or r_orig == None:
            # skip cases like that
            logging.info(f"INFO: None values for s, o, r_orig: {s}, {o}, {r_orig}")
            continue

        orig_template_id = r2id[
            r_orig
        ]  # refers to the original relation template, the paraphrase was created from
        o_neg = [o_ for o_ in o_neg_sets_sampled[r_orig] if o_ != o]
        if len(o_neg) != len(o_neg_sets_sampled[r_orig]):
            logging.info(
                f"INFO: true O is included in o_neg for r:{r_orig}: {o_neg_sets_sampled[r_orig]}; o={o}"
            )

        # get the paraphrased template corresponding to the relation id, r_id to fill sequences into
        for para_id, r_paraphrased in enumerate(paraphrased_templates[r_orig]):
            # print(f"Original + Paraphrased templates for r: {r_paraphrased}")
            # print(f"All parpahrse templates for r: {paraphrased_templates[r_orig]}")
            # print(f"All keys: {paraphrased_templates.keys()}")

            # True SOR sequence
            true_sequence = fill_template(r_paraphrased, s, o)
            data.append(
                [
                    instance,
                    0,
                    s,
                    o,
                    true_sequence,
                    "pos",
                    r_orig,
                    orig_template_id,
                    r_paraphrased,
                    para_id,
                ]
            )  # permutation id 0 = true sequence

            # False SOR sequences, for each o_neg
            for i, o_neg_ in enumerate(o_neg):
                false_sequence = fill_template(r_paraphrased, s, o_neg_)
                data.append(
                    [
                        instance,
                        i + 1,
                        s,
                        o_neg_,
                        false_sequence,
                        "neg",
                        r_orig,
                        orig_template_id,
                        r_paraphrased,
                        para_id,
                    ]
                )

    return data


def correct_grammar_batch(seqs):
    # for a list of sequences
    tool = language_tool_python.LanguageTool("en-US")

    corrected_seqs = []
    for seq in tqdm(seqs, desc="Grammer checking with n>1 jobs"):
        matches = tool.check(seq)
        corrected_seq = language_tool_python.utils.correct(seq, matches)
        corrected_seqs.append(corrected_seq)

    return corrected_seqs


def correct_grammar(seq, tool):
    # for single sequence
    matches = tool.check(seq)
    corrected_seq = language_tool_python.utils.correct(seq, matches)

    return corrected_seq


def few_shot_examples(df_row, df):
    # get the original relation id of the given row
    original_relation_id = df_row["orig_relation_id"]
    r_s_id = df_row["r_s_id"]
    paraphrase_id = df_row["paraphrase_id"]

    demos = df[
        (df["orig_relation_id"] == original_relation_id)
        & (df["label"] == "pos")
        & (df["r_s_id"] != r_s_id)
    ]
    demos_same_paraphrase = df[
        (df["orig_relation_id"] == original_relation_id)
        & (df["label"] == "pos")
        & (df["r_s_id"] != r_s_id)
        & (df["paraphrase_id"] == paraphrase_id)
    ]

    demo_ids = demos.index.to_list()
    demo_same_paraphrase_ids = demos_same_paraphrase.index.to_list()

    return demo_ids, demo_same_paraphrase_ids


def main(args):
    OUT_PATH = Path(args.out_path)
    RUN_NAME = OUT_PATH.name  # name of lowest folder
    OUT_PATH.mkdir(exist_ok=True)
    dataset_unsampled_PATH = OUT_PATH / "unsampled.hf5"
    dataset_processed = OUT_PATH / "cleaned.hf5"

    configure_logging(path=OUT_PATH, mode="prepare_data")
    args.cpu_count = os.cpu_count()
    logging.info(f"Number of CPUs: {args.cpu_count}")

    init_wandb(args, mode="prepare_data")

    set_seed(args.seed)  # torch, numpy, random, etc.

    if args.LM_postprocessing == True and args.grammar_postprocessing == True:
        raise ValueError("Only one postprocessing method can be used at a time")

    if dataset_processed.exists():
        logging.info(f"Loading filtered / sampled dataset from {dataset_processed}")
        df_processed = pd.read_hdf(dataset_processed)

        with open(OUT_PATH / "o_neg_sets.json") as f:  # sampled o_neg sets
            o_neg_sets_sampled = json.load(f)

        paraphrase_df = pd.read_hdf(OUT_PATH / "paraphrase_df.hf5")

        if args.s_contexts:
            with open(args.s_contexts) as f:
                s_contexts = json.load(f)  # 'subject': context
        else:
            s_contexts = None

    else:
        s_contexts = None

        if args.dataset_name == "trex":
            df_unsampled = trex_preprocessing(dataset_unsampled_PATH)
            paraphrase_df = pd.read_hdf(args.paraphrase_templates)

            # for sake of completeness: store the paraphrase dataset in same format
            paraphrase_df.to_hdf(OUT_PATH / "paraphrase_df.hf5", key="df", mode="w")

            # full preprocessing requires downloaded pararel templates # TODO integrate better
            # paraphrase_df = create_paraphrase_df(
            #     df_unsampled,
            #     type="trex",
            #     template_PATH=template_PATH,
            #     OUT_PATH=args.paraphrase_templates,
            # )

        elif args.dataset_name == "hypernymy":
            df_unsampled = hypernymy_preprocessing(
                hypernymy_PATH=OUT_PATH.parent / "hypernymy_examples.json"
            )

            paraphrase_df = create_paraphrase_df(
                df_unsampled,
                type="hypernymy",
                template_PATH=args.paraphrase_templates,
                OUT_PATH=OUT_PATH / "paraphrase_df.hf5",
            )

        elif args.dataset_name == "PopQA":
            dataset = load_dataset("akariasai/PopQA")["test"]
            df = pd.DataFrame(dataset)

            df_unsampled = popQA_preprocessing(args.paraphrase_templates, df_dataset=df)

            paraphrase_df = create_paraphrase_df(
                df_unsampled,
                type="PopQA",
                template_PATH=args.paraphrase_templates,
                OUT_PATH=OUT_PATH / "paraphrase_df.hf5",
            )
        else:
            raise ValueError(f"Unknown dataset: {args.dataset_name}")

        df_unsampled.to_hdf(dataset_unsampled_PATH, key="df", mode="w")

        # save overview of the unsampled dataset
        unsampled_stats = get_df_overview_stats(df_unsampled)
        unsampled_stats.to_csv(OUT_PATH / "unsampled_stats.csv")

        if args.s_contexts:  # TODO can be deleted completely
            with open(args.s_contexts) as f:
                s_contexts = json.load(f)  # 'subject': context

            subjects_df = df_unsampled["sub_label"].unique().tolist()
            s_contexts_labels = list(s_contexts.keys())

            for s in subjects_df:
                assert s in s_contexts_labels, f"Subject {s} of df not in s_contexts"

        # store the o_neg set for each relation, get it from the unsampled full dataset
        original_relations = df_unsampled["template"].unique().tolist()
        logging.info(f"Number of relations: {len(original_relations)}")

        o_neg_sets, n_o_neg_sets = {}, []
        all_o_neg = []
        for r in original_relations:
            o_neg_set = get_o_neg(r, df_unsampled)
            n_o_neg_sets.append(len(o_neg_set))
            o_neg_sets[r] = o_neg_set
            all_o_neg.extend(o_neg_set)
        logging.info(f"Number of objects in o_neg set (full): {n_o_neg_sets}")

        # store the o_neg sets as json
        with open(OUT_PATH / "o_neg_sets_full.json", "w") as f:
            json.dump(o_neg_sets, f)
            logging.info(f"Stored o_neg sets in {OUT_PATH / 'o_neg_sets_full.json'}")

        if args.equal_n_o:
            # get largest common number of o_neg objects
            args.num_o = min(n_o_neg_sets)  # 100

        o_neg_sets_sampled = {}
        for r in o_neg_sets.keys():
            if len(o_neg_sets[r]) > args.num_o:
                o_neg_sets_sampled[r] = list(
                    np.random.choice(o_neg_sets[r], args.num_o, replace=False)
                )
            else:
                missing = args.num_o - len(o_neg_sets[r])
                missing_o = list(np.random.choice(all_o_neg, missing, replace=False))
                o_neg_sets_sampled[r] = o_neg_sets[r] + missing_o

            logging.info(
                f"Sampled {len(o_neg_sets_sampled[r])} objects for relation: {r}"
            )

        with open(OUT_PATH / "o_neg_sets.json", "w") as f:
            json.dump(o_neg_sets_sampled, f)
            logging.info(
                f"Stored sampled o_neg sets (n={args.num_o}) in {OUT_PATH / 'o_neg_sets.json'}"
            )

        logging.info(f"Filtering dataset...")

        # sample n examples (S) per template
        min_instances_per_r = df_unsampled["template"].value_counts().min()
        wandb.run.summary["n_instances_per_r"] = min_instances_per_r

        for t, template in enumerate(df_unsampled["template"].unique()):
            df_t = df_unsampled[df_unsampled["template"] == template].head(
                args.n_instances_per_r
            )

            if t == 0:
                df_processed = df_t
            else:
                df_processed = pd.concat([df_processed, df_t], axis=0)

        df_processed.to_hdf(dataset_processed, key="df", mode="w")

        # save overview of the cleaned / processed dataset
        processed_stats = get_df_overview_stats(df_processed)
        processed_stats.to_csv(OUT_PATH / "cleaned_stats.csv")

    original_relations = df_processed["template"].unique().tolist()
    logging.info(
        f'S, r instances in filtered df: {df_processed["template"].value_counts()}'
    )
    logging.info(f"Original relations: {original_relations}")
    paraphrased_templates = {}
    for r in original_relations:
        templates_paraphrased = paraphrase_df[paraphrase_df["original_template"] == r][
            "human_paraphrase_templates"
        ].to_list()  # TODO adapt for other paras
        logging.info(f"Templates paraphrased: {templates_paraphrased}")
        # key is orig relation: list of orig + paraphrased templates
        paraphrases = [r]
        for templates in templates_paraphrased:
            for t in templates:
                # ensure object is at last position
                if t[-5:] == "[Y] .":
                    paraphrases.append(t)
                elif t[-4:] == "[Y].":
                    paraphrases.append(t)
                elif t[-4:] == "[Y],":
                    paraphrases.append(t)
                elif t[-4:] == "[Y] ":
                    paraphrases.append(t)
                elif t[-3:] == "[Y]":
                    paraphrases.append(t)
                else:
                    # print(f"Template not ending with [Y]: '{t}'")
                    paraphrases.append(t)
        paraphrased_templates[r] = paraphrases

    with open(OUT_PATH / "paraphrased_templates.json", "w") as f:
        json.dump(paraphrased_templates, f, indent=4)
        logging.info(
            f"Stored paraphrased templates in {OUT_PATH / 'paraphrased_templates.json'}"
        )

    # for each relation template (equivalently phrased relation template), create permutations of s, o, + s, o_neg pairs
    r2id = {r: i for i, r in enumerate(original_relations)}

    logging.info(
        f"nb of o_neg_sets_sampled per r: {[len(v) for v in o_neg_sets_sampled.values()]}"
    )

    data = create_s_o_permutations(
        o_neg_sets_sampled,
        df_processed,
        paraphrased_templates,
        r2id=r2id,
        args=args,
    )

    # create df from complete data
    df_instance_permutations = pd.DataFrame(
        data,
        columns=[
            "r_s_id",  # was instance_id
            "o_permutation_n",  # was permutation_n
            "sub_label",
            "obj_label",
            "sequence",
            "label",
            "orig_relation_template",  # was orig_relation
            "orig_relation_id",
            "paraphrased_relation_template",  # was relation_paraphrased
            "paraphrase_id",
        ],
    )

    # save original sequences before postprocessing (ensuring grammatical correctness)
    sequences_original = df_instance_permutations["sequence"].to_list()
    df_instance_permutations["sequence_original"] = sequences_original

    # Postprocessing using T5 for PopQA and hypernymy datasets only, not Trex
    if args.dataset_name in ["PopQA", "hypernymy"]:

        # Postprocessing of sequences to make sure the are grammatically correct, for other datasets this is done in the preprocessing
        # TODO more dynamic with [a/an] in template or sth

        if args.LM_postprocessing:
            model_name = "google/flan-t5-base"
            instruction = "Make only small modifications to the following sencence to ensure that it is grammatically correct. Do not change the structure of the sentence."
            genlm = GenLM(model_name, instruction=instruction)

            logging.info(
                f'Postprocessing sequences (n={len(sequences_original)})using model: "{model_name}"'
            )
            postprocessed_sequences = genlm.batch_generate(
                sequences_original, batch_size=args.batch_size_postprocessing
            )
            assert len(postprocessed_sequences) == len(
                df_instance_permutations
            ), f"Length of postprocessed sequences: {len(postprocessed_sequences)} != {len(df_instance_permutations)}"

            df_instance_permutations["sequence"] = postprocessed_sequences

    if args.grammar_postprocessing:

        logging.info(
            f"Postprocessing sequences (n={len(sequences_original)}) using grammar checker with {args.n_jobs} jobs..."
        )

        # time the grammar correction
        start_time = time.time()

        if args.n_jobs > 1:
            chunck_size = (
                len(sequences_original) // args.n_jobs
            ) + 1  # if there is a remainder
            postprocessed_sequences = Parallel(
                n_jobs=args.n_jobs, backend="multiprocessing"
            )(
                delayed(correct_grammar_batch)(sequences_original[i : i + chunck_size])
                for i in range(0, len(sequences_original), chunck_size)
            )

            # flatten the results
            postprocessed_sequences = [
                item for sublist in postprocessed_sequences for item in sublist
            ]

        else:  # no multiprocessing
            tool = language_tool_python.LanguageTool(
                "en-US",
            )
            postprocessed_sequences = [
                correct_grammar(seq, tool)
                for seq in tqdm(sequences_original, desc="Grammar checking")
            ]

        assert len(postprocessed_sequences) == len(
            df_instance_permutations
        ), f"Length of postprocessed sequences: {len(postprocessed_sequences)} != {len(df_instance_permutations)}"

        df_instance_permutations["sequence"] = postprocessed_sequences

        end_time = time.time()
        df_instance_permutations.to_parquet(OUT_PATH / "permutations.parquet")

        logging.info(
            f"INFO: Grammar postprocessing took {end_time - start_time} seconds"
        )

    if args.n_shot_examples > 0:
        all_demo_ids, demo_same_r_ids = [], []

        demos = {"all": {}, "same_paraphrase": {}}
        for r_s in df_instance_permutations["r_s_id"].unique():
            row = df_instance_permutations[
                df_instance_permutations["r_s_id"] == r_s
            ].iloc[0]
            demo_ids, demo_same_paraphrase_ids = few_shot_examples(
                row, df_instance_permutations
            )
            demos["all"][int(r_s)] = demo_ids
            demos["same_paraphrase"][int(r_s)] = demo_same_paraphrase_ids

        with open(OUT_PATH / "demos.json", "w") as f:
            json.dump(demos, f, indent=4)
            logging.info(f"Stored few shot examples in {OUT_PATH / 'demos.json'}")

    df_instance_permutations.to_parquet(OUT_PATH / "permutations.parquet")

    logging.info(f"Len permutations.parquet: {len(df_instance_permutations)}")

    total_permutations_over_r = sum([len(v) + 1 for v in o_neg_sets_sampled.values()])

    if args.grammar_postprocessing:
        logging.info(f"Grammar postprocessing took {end_time - start_time} seconds")

    logging.info(
        f"Prepare_data finished. Saved results in {OUT_PATH / 'permutations.parquet'}"
    )
    print(
        f"Prepare_data finished. n={len(df_instance_permutations)} Saved results in {OUT_PATH / 'permutations.parquet'}"
    )
    wandb.run.summary["n_instance_permutations"] = len(df_instance_permutations)
    wandb.run.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
