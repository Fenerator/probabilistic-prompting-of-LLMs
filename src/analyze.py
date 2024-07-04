import logging
import wandb
import json, os, sys, re
import pandas as pd
import numpy as np
import pickle
import time
from scipy import stats
from pathlib import Path
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from transformers import set_seed
from scipy.special import softmax
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
    plot_roc_curve,
    plot_coverage_risk_curve,
)

pd.options.mode.chained_assignment = None  # default='warn'
# add parent folder to path to import ProbLM
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from dev.ProbLM import JointLM, ConditionalLM
from dev.GenLM import GenLM
from src.utils.logging import configure_logging


def calculate_p_t(
    df_instance_permutations: pd.DataFrame,
    max_paraphrases: int or None = None,
):
    """adds P(T) for each S,R instance, adds p_t_over_paraphrses for each S,R and all paraphrase versions to the input df

    Args:
        df_instance_permutations (_type_): _description_
        max_paraphrases (_type_, optional): _description_. Defaults to None. If None all paraphrases are considered. 0: only original template used, 1: original + 1 paraphrase template used.

    Returns:
        _type_: _description_
    """

    if max_paraphrases is None:
        max_paraphrases = df_instance_permutations["paraphrase_id"].max()

    # P(T): for each S, R_i, {o}, for each i (paraphrase) seperately
    # softmax to get probabilities, normalize over sequences of the same instance (same S, R)
    prob_seq = []
    logging.info(
        f'Calculating P(T) for {len(df_instance_permutations["r_s_id"].unique())} instances...'
    )

    # consider only up to max_paraphrases. 0: original only
    # by keeping only the relevant paraphrases: id <- max_paraphrases
    df_relevant_paraphrases = df_instance_permutations[
        df_instance_permutations["paraphrase_id"] <= max_paraphrases
    ]

    # P(T_over_paraphrases): sum of the each paraphrase's P(T)
    # if we have paraphrases of a relation, we need to sum the sequence probabilities of an S,O instance over all pos labeled sequences
    # and adaptions to fit it into the df
    # for each r
    for r_s_id in tqdm(
        df_relevant_paraphrases["r_s_id"].unique(), desc="Calculating P(T)"
    ):
        r_s_df = df_relevant_paraphrases[df_relevant_paraphrases["r_s_id"] == r_s_id]

        log_seqs = r_s_df["log_seq_prob"].to_numpy()
        o_permutation_n = r_s_df["o_permutation_n"].to_numpy()

        prob_s = softmax(log_seqs)  # P(T) for each o (with the same S,R)
        prob_seq.extend(prob_s)

        assert np.isclose(
            prob_s.sum(), 1.0
        ), f"Probabilities for S={r_s_id} and paraphrase {paraphrase_id} do not sum to 1: {prob_s.sum()}"

        for o in np.unique(
            o_permutation_n
        ):  # now S,R,O are fixed: iterate over the paraphrases of one such instance
            df_p_to_sum = r_s_df[r_s_df["o_permutation_n"] == o]

            # insert the value of P_T_over_paraphreases at original paraphrase index; not in a row of a paraphrased instance
            idx_paraphrase_0 = df_p_to_sum[
                df_p_to_sum["paraphrase_id"] == 0
            ].index.values
            assert (
                idx_paraphrase_0.shape[0] == 1
            ), f"More/less than one true sequence found for {r_s_id} and o: {o} \n {idx_paraphrase_0}"
            idx_paraphrase_0 = idx_paraphrase_0[0]

            assert (
                prob_s.shape == o_permutation_n.shape
            ), f"Shape mismatch: prob_s {prob_s.shape} != o_permutation_n {o_permutation_n.shape}"
            # select P(T)s of instances with the same S,R,O (over all paraphrases of one S,R,O)
            p_t_over_paraphrases = prob_s[o_permutation_n == o].sum()

            df_relevant_paraphrases.loc[idx_paraphrase_0, f"p_t_over_paraphrases"] = (
                p_t_over_paraphrases
            )

    all_seq_prob = np.vstack(prob_seq)
    df_relevant_paraphrases["p_t"] = all_seq_prob

    return df_relevant_paraphrases


def _calculate_template_probability(df_relevant_paraphrases, s_r_t_id):
    r_s_df = df_relevant_paraphrases[df_relevant_paraphrases["s_r_t_id"] == s_r_t_id]

    log_seqs = r_s_df["log_seq_prob"].to_numpy()
    o_permutation_n = r_s_df["o_permutation_n"].to_numpy()

    prob_s = softmax(log_seqs)  # P(T) for each o (with the same S,R)

    assert np.isclose(
        prob_s.sum(), 1.0
    ), f"Probabilities for S={r_s_id} of rel_paraphrases do not sum to 1: {prob_s.sum()}"

    return prob_s


def _sum_p_o_over_paraphrases(
    r_s_df, o, r_s_id, prob_s, o_permutation_n, df_relevant_paraphrases
):
    df_p_to_sum = r_s_df[r_s_df["o_permutation_n"] == o]  # P(o|S,t_i(R))

    # insert the value of P_T_over_paraphreases at original paraphrase index; not in a row of a paraphrased instance
    idx_paraphrase_0 = df_p_to_sum[df_p_to_sum["paraphrase_id"] == 0].index.values
    assert (
        idx_paraphrase_0.shape[0] == 1
    ), f"More/less than one true sequence found for {r_s_id} and o: {o} \n {idx_paraphrase_0}"
    idx_paraphrase_0 = idx_paraphrase_0[0]

    assert (
        prob_s.shape == o_permutation_n.shape
    ), f"Shape mismatch: prob_s {prob_s.shape} != o_permutation_n {o_permutation_n.shape}"
    # select P(T)s of instances with the same S,R,O (=> sums over all paraphrases of S,R,O)
    p_t_over_paraphrases = prob_s[o_permutation_n == o].sum()

    df_relevant_paraphrases.loc[idx_paraphrase_0, f"p_t_over_paraphrases"] = (
        p_t_over_paraphrases
    )

    return df_relevant_paraphrases


def _calculate_p_r_s(
    df_relevant_paraphrases,
    r_s_id,
):
    r_s_df = df_relevant_paraphrases[df_relevant_paraphrases["r_s_id"] == r_s_id]
    num_paraphrase_templates = r_s_df["paraphrase_id"].unique().shape[0]
    o_permutation_n = r_s_df["o_permutation_n"].to_numpy()

    # NEW: marginalize out paraphrase templates
    p_t_s_r = r_s_df["p_t_s_r"].to_numpy()
    prob_r_s = p_t_s_r / num_paraphrase_templates  # normalize over paraphrases

    assert np.isclose(
        prob_r_s.sum(),
        1.0,
        atol=1e-18,
    ), f"Probabilities for S={r_s_id} of rel_paraphrases do not sum to 1: {prob_s.sum()}"

    for o in np.unique(
        o_permutation_n
    ):  # now S,R,O are fixed: iterate over the paraphrases of one such instance

        # prob_s: P(T) of S,R instances
        df_relevant_paraphrases = _sum_p_o_over_paraphrases(
            r_s_df, o, r_s_id, prob_r_s, o_permutation_n, df_relevant_paraphrases
        )  # P(T)_over_paraphrases

    return prob_r_s, df_relevant_paraphrases


def calculate_instance_probability(
    df_instance_permutations: pd.DataFrame,
    max_paraphrases: int or None = None,
    n_jobs: int = 8,
):
    """adds P(T) for each S,R instance, adds p_t_over_paraphrses for each S,R and all paraphrase versions to the input df

    Args:
        df_instance_permutations (_type_): _description_
        max_paraphrases (_type_, optional): _description_. Defaults to None. If None all paraphrases are considered. 0: only original template used, 1: original + 1 paraphrase template used.

    Returns:
        _type_: _description_
    """

    if max_paraphrases is None:
        max_paraphrases = df_instance_permutations["paraphrase_id"].max()
        logging.info(f"Considering all paraphrase templates: Max = {max_paraphrases}")

    # P(T): for each S, R_i, {o}, for each i (paraphrase) seperately
    # softmax to get probabilities, normalize over sequences of the same instance (same S, R)

    # consider only up to max_paraphrases. 0: original only
    # by keeping only the relevant paraphrases: id <- max_paraphrases
    df_relevant_paraphrases = df_instance_permutations[
        df_instance_permutations["paraphrase_id"] <= max_paraphrases
    ]

    # P(T_over_paraphrases): sum of the each paraphrase's P(T)
    # if we have paraphrases of a relation, we need to sum the sequence probabilities of an S,O instance over all pos labeled sequences
    # and adaptions to fit it into the df
    # for each r
    logging.info(
        f'Calculating P(T) for {len(df_instance_permutations["r_s_id"].unique())} instances and max_paraphrases {max_paraphrases} ...'
    )

    # calculate P(t,s,r) / over objects
    prob_seq_t_s_r = []
    for s_r_t_id in tqdm(
        df_relevant_paraphrases["s_r_t_id"].unique(), desc="Calculating P(T)"
    ):
        prob_template = _calculate_template_probability(
            df_relevant_paraphrases, s_r_t_id
        )
        prob_seq_t_s_r.extend(prob_template)

    all_prob_seq_t_s_r = np.vstack(prob_seq_t_s_r)
    df_relevant_paraphrases["p_t_s_r"] = all_prob_seq_t_s_r

    # Calculate prob. per r, s instance; and sum probas of the same object over paraphrases
    prob_seq = []
    for r_s_id in tqdm(
        df_relevant_paraphrases["r_s_id"].unique(), desc="Calculating P(T)"
    ):

        prob_r_s, df_relevant_paraphrases = (
            _calculate_p_r_s(  # P(T) and P(T)_over_paraphrases # HERE
                df_relevant_paraphrases, r_s_id
            )
        )
        prob_seq.extend(prob_r_s)

    all_seq_prob = np.vstack(prob_seq)

    df_relevant_paraphrases["p_t"] = all_seq_prob

    return df_relevant_paraphrases


def calculate_ranking(df_instance_permutations):
    """Ranks each o of {o},S,R based on P(T) over paraphrases.

    Args:
        df_instance_permutations (pd.DataFrame): df containing P(T) over paraphrases (p_t_over_paraphrases) column.

    Returns:
        pd.DataFrame: with an added `rank_o` column , containing the ranking of o based on P(T) over paraphrases.
    """
    for r_s_id in tqdm(
        df_instance_permutations["r_s_id"].unique(),
        desc="Calculating ranking of o",
    ):
        # relevant probabilities over paraphrases are saved for paraphrase id 0 only
        df_r_s = df_instance_permutations[
            (df_instance_permutations["r_s_id"] == r_s_id)
            & (df_instance_permutations["paraphrase_id"] == 0)
        ]

        # argsort to get ranking
        p_t_over_p = df_r_s["p_t_over_paraphrases"].to_numpy()
        ascending_indices = np.argsort(p_t_over_p)
        descending_indices = ascending_indices[::-1]
        ranking = np.empty_like(descending_indices)
        ranking[descending_indices] = (
            np.arange(len(descending_indices)) + 1
        )  # from 1, ... , n

        # insert it into the full df, starting at the row where o_permutation_n == 0
        idx_o_0 = df_r_s[
            df_r_s["o_permutation_n"] == 0
        ].index.values  # insert the value at original paraphrase index
        assert (
            idx_o_0.shape[0] == 1
        ), f"More/less than one true sequence found for {r_s_id} and o: {o} \n {idx_o_0}"
        idx_o_0 = idx_o_0[0]

        # insert ranking into the df
        df_instance_permutations.loc[
            idx_o_0 : idx_o_0 + len(ranking) - 1, f"rank_o"
        ] = ranking

    return df_instance_permutations


def calculate_ranking_per_paraphrase(df_instance_permutations, metric="log_seq_prob"):
    """Ranks each o of {o},S,t(R)_i based on seq. probability. (e.g. 'log_seq_prob' column)

    Args:
        df_instance_permutations (pd.DataFrame): df

    Returns:
        pd.DataFrame: with an added `rank_o|t(r)_s` column , containing the ranking of o based on P(T) over paraphrases.
    """
    for r_s_id in tqdm(
        df_instance_permutations["r_s_id"].unique(),
        desc="Calculating ranking per template of o using seq. prob.",
    ):
        # relevant probabilities over paraphrases are saved for paraphrase id 0 only
        df_r_s = df_instance_permutations[
            (df_instance_permutations["r_s_id"] == r_s_id)
        ]

        for paraphrase_id in df_r_s["paraphrase_id"].unique():
            df_r_s_p = df_r_s[df_r_s["paraphrase_id"] == paraphrase_id]

            # argsort to get ranking
            seq_prob = df_r_s_p[metric].to_numpy()
            ascending_indices = np.argsort(seq_prob)
            descending_indices = ascending_indices[::-1]
            ranking = np.empty_like(descending_indices)
            ranking[descending_indices] = np.arange(len(descending_indices)) + 1

            # insert it into the full df, starting at the row where o_permutation_n == 0

            idx_o_0 = df_r_s_p.index.values
            idx_o_0 = idx_o_0[0]

            # insert ranking into the df
            df_instance_permutations.loc[
                idx_o_0 : idx_o_0 + len(ranking) - 1, f"rank_o|{metric},r_s_t(r)_i"
            ] = ranking

    return df_instance_permutations


def main(args):
    OUT_PATH = Path(args.out_path)
    RUN_NAME = OUT_PATH.name  # name of lowest folder
    OUT_PATH.mkdir(exist_ok=True)
    dataset_unsampled_PATH = OUT_PATH / "unsampled.hf5"
    dataset_processed = OUT_PATH / "cleaned.hf5"

    configure_logging(path=OUT_PATH, mode="analyze")

    args.cpu_count = os.cpu_count()
    logging.info(f"Number of CPUs: {args.cpu_count}")
    classification_thresholds = set(args.classification_thresholds)
    classification_thresholds.add(0.0)
    args.classification_thresholds = sorted(list(classification_thresholds))

    init_wandb(args, mode="analyze")

    set_seed(args.seed)  # torch, numpy, random, etc.

    # Read files from previous steps
    df_instance_permutations = pd.read_parquet(
        OUT_PATH / "permutations_log_seq_scores.parquet"
    )

    with open(OUT_PATH / "o_neg_sets.json") as f:  # sampled o_neg sets
        o_neg_sets_sampled = json.load(f)

    with open(OUT_PATH / "paraphrased_templates.json", "r") as f:
        paraphrased_templates = json.load(f)

    # add s_r_t_id to df (unique identifier for each S,R,t instance)
    df_instance_permutations["s_r_t_id"] = np.nan
    id_counter = 0
    for r_s_id in df_instance_permutations["r_s_id"].unique():
        for paraphrase_id in df_instance_permutations[
            df_instance_permutations["r_s_id"] == r_s_id
        ]["paraphrase_id"].unique():
            df_instance_permutations.loc[
                (df_instance_permutations["r_s_id"] == r_s_id)
                & (df_instance_permutations["paraphrase_id"] == paraphrase_id),
                "s_r_t_id",
            ] = id_counter
            id_counter += 1

    # calculate P(t) for each instance; over paraphrased templates
    start_time2 = time.time()
    df_instance_permutations = calculate_instance_probability(
        df_instance_permutations,
        n_jobs=args.n_jobs,
    )
    run_time_2 = time.time() - start_time2
    logging.info(f"Calculating P(T) took {run_time_2} seconds")

    df_instance_permutations = calculate_ranking(df_instance_permutations)
    df_instance_permutations = calculate_ranking_per_paraphrase(
        df_instance_permutations, metric="p_t_s_r"
    )

    # save P(T) scores in df in hf5 format
    df_instance_permutations.to_parquet(
        OUT_PATH / "permutations_scores_p_t_all.parquet"
    )

    logging.info(
        f'Saved P(T) scores in {OUT_PATH / "permutations_scores_p_t_all.parquet"}'
    )

    # ANALYSIS:
    df_processed = pd.read_hdf(OUT_PATH / "cleaned.hf5")
    original_relations = df_processed["template"].unique().tolist()

    # ANALYSIS: get stats for all variants of paraphrase and objects thresholds
    # create df per args.max_n_paraphrases
    # do this only if the number of paraphrases exist

    num_paraphrase_templates = [
        len(paraphrased_templates[r]) for r in original_relations
    ]

    logging.info(f"nb of paraphrased templates per r: {num_paraphrase_templates}")

    # 1: original + 1 paraphrase
    n_paraphrases_thresholds = set(args.max_n_paraphrases)
    n_paraphrases_thresholds.update(
        [0, max(num_paraphrase_templates)]
    )  # 0: original only, and all available paraphrases

    max_n_paraphrases_to_consider = set()
    for n in list(n_paraphrases_thresholds):
        if n == -1:  # consider every single n_paraphrases
            all_p_thresholds = [i for i in range(max(num_paraphrase_templates))]
            max_n_paraphrases_to_consider.update(all_p_thresholds)
        elif n <= max(num_paraphrase_templates):  # consider only selected paraphrases
            max_n_paraphrases_to_consider.add(n)
    args.max_n_paraphrases = max_n_paraphrases_to_consider
    logging.info(f"Max max_n_paraphrases_to_consider: {max_n_paraphrases_to_consider}")
    print(f"max_n_paraphrases_to_consider: {max_n_paraphrases_to_consider}")

    n_o_per_relation = [len(v) for v in o_neg_sets_sampled.values()]
    max_common_n_o = min(n_o_per_relation)

    max_o_to_consider = set()
    max_o_to_consider.add(
        max(n_o_per_relation)
    )  # considers max. num of objects available for o_neg

    for n in args.max_n_objects:
        if n == -1:
            all_num_o = [i for i in range(max_common_n_o + 1)]
            max_o_to_consider.update(all_num_o)
        elif n == -2:
            max_o_to_consider.add(max(n_o_per_relation))
        elif n <= args.num_o:
            max_o_to_consider.add(n)
    args.max_n_objects = max_o_to_consider
    logging.info(f"Max n_o thresholds considered: {max_o_to_consider}")
    print(f"Max n_o thresholds considered: {max_o_to_consider}")

    # save the hyperparameters and arguments used in this run
    with open(OUT_PATH / "config.json", "w") as f:
        json.dump(args.__dict__, f, indent=4, cls=CustomEncoder)

    logging.info(f"ARGS: {args.__dict__}")

    df_by_max_p = {}
    df_classification_results_by_max_p = {}
    data = []
    df_classification_results_all_p = pd.DataFrame()
    # max_o: limit the number of o_neg that are considered in the calcualtion of P(T) and P(T)_over_paraphrases
    for max_p in sorted(list(max_n_paraphrases_to_consider)):
        for max_o in max_o_to_consider:  # 0=original only, 1=original + 1 o_neg
            df_instance_permutations_max_o = df_instance_permutations[
                df_instance_permutations["o_permutation_n"] <= max_o
            ]

            # subsets df_instance_permutations to max_p paraphrases
            df_instance_permutations_max_p = calculate_instance_probability(
                df_instance_permutations_max_o,
                max_paraphrases=max_p,
                n_jobs=args.n_jobs,
            )
            df_instance_permutations_max_p = calculate_ranking(
                df_instance_permutations_max_p
            )

            df_instance_permutations_max_p = calculate_ranking_per_paraphrase(
                df_instance_permutations_max_p, metric="p_t_s_r"
            )

            # True vs. False Classification and Selective Prediction
            # for each max para individually: classify and get stats
            df_classification_results = classify(  # HERE
                df_instance_permutations_max_p,
                o_neg_sets_sampled,
                paraphrased_templates,
                thresholds=args.classification_thresholds,  # 0 is included
                max_paraphrases=max_p,
                max_o=max_o,
            )

            df_classification_results_all_p = pd.concat(
                [df_classification_results_all_p, df_classification_results],
                ignore_index=True,
            )
            df_by_max_p[max_p] = df_instance_permutations_max_p

            df_classification_results_by_max_p[max_p] = df_classification_results

    df_classification_results_all_p.to_parquet(
        OUT_PATH / "classification_results_all_thresholds.parquet"
    )
    df_classification_results_all_p.to_csv(
        OUT_PATH / "classification_results_all_thresholds.csv"
    )

    with open(OUT_PATH / "df_by_max_p.pickle", "wb") as f:
        pickle.dump(df_by_max_p, f)
    with open(OUT_PATH / "df_classification_results_by_max_p.pickle", "wb") as f:
        pickle.dump(df_classification_results_by_max_p, f)

    # Plots
    PLOT_PATH = OUT_PATH / "plots"
    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    for max_p in max_n_paraphrases_to_consider:
        # Boxplots of P(T) by subject
        fig = boxplots(
            df_by_max_p[max_p],
            y="rank_o",
            x="sub_label",
            title=f"{RUN_NAME}: rank by subjects, n_paraphrases={max_p}",
            out_path=PLOT_PATH / f"rank_true_neg_boxplots_subjects_n_para_{max_p}.html",
        )

        fig2 = boxplots(
            df_by_max_p[max_p],
            y="rank_o",
            x="obj_label",
            title=f"{RUN_NAME}: rank by objects, n_paraphrases={max_p}",
            out_path=PLOT_PATH / f"rank_true_neg_boxplots_objects_n_para_{max_p}.html",
        )

        # Boxplots of P(T) by relation
        n_para = 0
        fig3 = boxplots(
            df_by_max_p[max_p],
            y="p_t_over_paraphrases",
            x="sub_label",
            title=f"{RUN_NAME}: P(T) by subjects, n_paraphrases={max_p}",
            out_path=PLOT_PATH / f"p_true_neg_boxplots_subjects_n_para_{max_p}.html",
        )

        fig4 = boxplots(
            df_by_max_p[max_p],
            y="p_t_over_paraphrases",
            x="obj_label",
            title=f"{RUN_NAME}: rank by objects, n_paraphrases={max_p}",
            out_path=PLOT_PATH / f"p_true_neg_boxplots_objects_n_para_{max_p}.html",
        )


    logging.info(f"analyze finished.")

    logging.info(
        f"analyze finished. Saved results in {OUT_PATH / 'classification_results_all_thresholds.csv'}"
    )
    print(
        f"analyze finished. n={len(df_classification_results_all_p)}. Saved results in {OUT_PATH / 'classification_results_all_thresholds.cvs'}"
    )

    wandb.run.summary["n_instance_permutations"] = len(df_instance_permutations)
    wandb.run.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
