import logging
import wandb
import json, os, sys, re
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import torch
from transformers import set_seed


from utils import (
    trex_preprocessing,
    hypernymy_preprocessing,
    # hyponymy_postprocessing,
    popQA_preprocessing,
    # popQA_postprocessing,
    create_paraphrase_df,
    classify,
    boxplots,
    CustomEncoder,
    init_wandb,
    parse_args,
)

# add parent folder to path to import ProbLM
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from dev.ProbLM import JointLM, ConditionalLM
from dev.GenLM import GenLM
from src.utils.logging import configure_logging


def sample_n_shot_examples(df_instance_permutations, all_demo_ids, OUT_PATH):
    # idx: r_s_id (ascending order), value: count
    r_s_counts = df_instance_permutations["r_s_id"].value_counts().sort_index()
    r_s_ids = r_s_counts.index.to_list()
    counts = r_s_counts.values

    # repeat the same selection of demos for all instances of a r,s pair
    selected_demo_ids = []
    selected_demo_ids_dict = {}  # r_s_id: selected_demo_ids
    for r_s_id, count in zip(r_s_ids, counts):
        demo_ids = all_demo_ids[str(r_s_id)]

        selected_ids = (
            np.random.choice(demo_ids, args.n_shot_examples, replace=False)
            if len(demo_ids) > args.n_shot_examples
            else demo_ids
        )
        selected_demo_ids.extend([selected_ids] * count)  # keep it flat
        selected_demo_ids_dict[str(r_s_id)] = selected_ids

    assert len(selected_demo_ids) == len(
        df_instance_permutations
    ), f"{len(selected_demo_ids)} != {len(df_instance_permutations)} (df length)"

    with open(OUT_PATH / "selected_demo_ids.json", "w") as f:
        json.dump(selected_demo_ids_dict, f, indent=4, cls=CustomEncoder)

    return selected_demo_ids_dict


def sample_n_shot_examples_negative(df_instance_permutations, all_demo_ids, OUT_PATH):
    """Select only unhelpful examples as demonstrations.

    Args:
        df_instance_permutations (_type_): _description_
        all_demo_ids (_type_): positive examples
        OUT_PATH (_type_): _description_

    Returns:
        _type_: _description_
    """
    # idx: r_s_id (ascending order), value: count

    r_s_counts = df_instance_permutations["r_s_id"].value_counts().sort_index()
    r_s_ids = r_s_counts.index.to_list()
    counts = r_s_counts.values

    # get negative examples: by removing the positive examples
    all_pos_demo_ids_flat = {
        item for sublist in all_demo_ids.values() for item in sublist
    }
    all_ids = df_instance_permutations.index.to_list()
    negative_ids = list(set(all_ids) - all_pos_demo_ids_flat)

    # repeat the same selection of demos for all instances of a r,s pair
    selected_demo_ids = []
    selected_demo_ids_dict = {}  # r_s_id: selected_demo_ids

    for r_s_id, count in zip(r_s_ids, counts):

        # get n negative examples for each r,s tuple
        selected_ids = (
            np.random.choice(negative_ids, args.n_shot_examples_negative, replace=False)
            if len(negative_ids) > args.n_shot_examples_negative
            else negative_ids
        )
        selected_demo_ids.extend([selected_ids] * count)  # keep it flat
        selected_demo_ids_dict[str(r_s_id)] = selected_ids

    assert len(selected_demo_ids) == len(
        df_instance_permutations
    ), f"{len(selected_demo_ids)} != {len(df_instance_permutations)} (df length)"

    with open(OUT_PATH / "selected_demo_ids_neg.json", "w") as f:
        json.dump(selected_demo_ids_dict, f, indent=4, cls=CustomEncoder)

    return selected_demo_ids_dict


def main(args):
    OUT_PATH = Path(args.out_path)
    RUN_NAME = OUT_PATH.name  # name of lowest folder
    OUT_PATH.mkdir(exist_ok=True)
    dataset_unsampled_PATH = OUT_PATH / "unsampled.hf5"
    dataset_processed = OUT_PATH / "cleaned.hf5"

    configure_logging(path=OUT_PATH, mode="encode")
    args.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    logging.info(f'Using device "{args.device}"')

    args.cpu_count = os.cpu_count()
    logging.info(f"Number of CPUs: {args.cpu_count}")

    if args.n_shot_examples > 0:
        assert (
            args.n_shot_examples_negative == 0
        ), "Cannot have both positive and negative examples"

    init_wandb(args, mode="encode")

    set_seed(args.seed)  # torch, numpy, random, etc.

    # Create sequence probabilties
    df_instance_permutations = pd.read_parquet(OUT_PATH / "permutations.parquet")
    logging.info(
        f"Number of sequences to create P(sequence) for: {len(df_instance_permutations)}"
    )

    # set up LM
    if args.model_name == "gpt2" or args.model_name == "openai-community/gpt2-large":
        lm = JointLM(
            context=None,
            model_name=args.model_name,
            debug=False,
        )
    elif args.model_name == "google/flan-t5-large":
        lm = ConditionalLM(
            context=None,
            model_name=args.model_name,
            debug=False,
        )
    elif args.model_name == "mistralai/Mistral-7B-Instruct-v0.2":
        lm = JointLM(
            context=None,
            model_name=args.model_name,
            debug=False,
        )

    # calculate sequence probabilties
    data = {"seq_prob": []}
    sequences = df_instance_permutations["sequence"].tolist()

    # get contexts (wikipedia abstracts or n-shot examples)
    if args.s_contexts:
        with open(args.s_contexts) as f:
            s_contexts = json.load(f)  # 'subject': context

    elif args.n_shot_examples > 0:  # with additional context
        # create N-shot examples for each r,s id
        logging.info(
            f"Using {args.n_shot_examples} shot examples from column '{args.example_column}'"
        )
        with open(OUT_PATH / "demos.json", "r") as f:
            all_demo_ids = json.load(f)
        all_demo_ids = all_demo_ids[args.example_column]  # r_s_id: list of demos

        # get selected demo ids; done for each r,s instance
        selected_demo_ids = sample_n_shot_examples(
            df_instance_permutations, all_demo_ids, OUT_PATH
        )

    elif args.n_shot_examples_negative > 0:
        # create negative N-shot examples for each r,s id
        logging.info(f"Using {args.n_shot_examples_negative} shot negative examples")
        with open(OUT_PATH / "demos.json", "r") as f:
            all_demo_ids = json.load(f)
        all_demo_ids = all_demo_ids["all"]

        # get selected demo ids done for each r,s instance
        selected_demo_ids = sample_n_shot_examples_negative(
            df_instance_permutations, all_demo_ids, OUT_PATH
        )

    for i in tqdm(
        range(0, len(sequences), args.batch_size),
        desc=f"Calculating P(sequence) for column: 'sequence'",
    ):
        if type(lm).__name__ == "JointLM":
            if args.s_contexts:  # for hypernymy wiki contexts
                subjects_batch = df_instance_permutations.iloc[i : i + args.batch_size][
                    "sub_label"
                ].tolist()
                contexts_batch = [s_contexts[s] for s in subjects_batch]

                assert len(contexts_batch) == len(
                    sequences[i : i + args.batch_size]
                ), f"contexts_batch: {len(contexts_batch)} != len Sequences: {len(sequences[i : i + args.batch_size])}"

                log_seq_prob, joint_mask, logits, rel_probs = lm.log_joint_with_context(
                    contexts_batch,
                    sequences[i : i + args.batch_size],
                )

            elif (
                args.n_shot_examples > 0 or args.n_shot_examples_negative > 0
            ):  # with additional context
                # concatenate selected demos to string for prompt:
                r_s_ids = df_instance_permutations.iloc[i : i + args.batch_size][
                    "r_s_id"
                ].tolist()
                batch_demo_ids = [selected_demo_ids[str(r_s_id)] for r_s_id in r_s_ids]

                batch_demos = []
                for ids in batch_demo_ids:
                    # get the demonstration examples from the dataframe by primary id
                    demo_strs = df_instance_permutations.loc[ids]["sequence"].to_list()
                    demo_prompt = "\n".join(demo_strs)
                    batch_demos.append(demo_prompt)

                assert len(batch_demos) == len(
                    sequences[i : i + args.batch_size]
                ), f"Demos: {len(batch_demos)} != len Sequences: {len(sequences[i : i + args.batch_size])}"

                log_seq_prob, joint_mask, logits, rel_probs = lm.log_joint_with_context(
                    batch_demos,
                    sequences[i : i + args.batch_size],
                )

            else:  # without additional context
                log_seq_prob, joint_mask, logits, rel_probs = lm.log_joint(
                    sequences[i : i + args.batch_size]
                )
        elif type(lm).__name__ == "ConditionalLM":
            raise NotImplementedError(
                "ConditionalLM not compatible with varying contexts"
            )

        # for conditional model, we use conditional probability
        data["seq_prob"].append(
            log_seq_prob.detach().cpu().numpy()
        )  # 1 value per seq x batch size

    all_log_seq_prob = np.vstack(data["seq_prob"])
    df_instance_permutations["log_seq_prob"] = all_log_seq_prob

    # save log seq probas
    df_instance_permutations.to_parquet(
        OUT_PATH / "permutations_log_seq_scores.parquet"
    )

    logging.info(
        f"encoding finished. Saved results in {OUT_PATH / 'permutations_log_seq_scores.parquet'}"
    )
    print(
        f"encoding finished. n={len(df_instance_permutations)} Saved results in {OUT_PATH / 'permutations_log_seq_scores.parquet'}"
    )
    wandb.run.summary["n_instance_permutations"] = len(df_instance_permutations)
    wandb.run.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
