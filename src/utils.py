import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import json, string
import nltk
from nltk.tokenize import word_tokenize
from pathlib import Path
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from scipy.stats import entropy
from sklearn.preprocessing import LabelEncoder
import pickle
import wandb
import copy
import argparse

# nltk.download("punkt")


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, list):
            return list(obj)
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def init_wandb(args, mode):

    if args.disable_wandb:
        wandb.init(mode="disabled")

    else:
        args_nice = vars(copy.deepcopy(args))
        wandb.init(
            project="mai",
            notes=f"{args.wandb_run_name}",
            tags=[
                mode,
                args.wandb_run_name,
                args.dataset_name,
            ],
            entity="piconda",
            config=args_nice,
        )


def get_data_permutations(run_name, BASE_PATH):
    OUT_PATH = BASE_PATH / run_name

    df_stats = pd.read_parquet(
        OUT_PATH / "classification_results_all_thresholds.parquet"
    )
    df_instance_permutations = pd.read_parquet(
        OUT_PATH / "permutations_scores_p_t_all.parquet"
    )

    return df_stats, df_instance_permutations


def classify_pt(
    df_r,
    df_r_pos,
    threshold,
):
    """
    df_r: already filtered by original relation template; and num_paraphrases
    df_r_pos: already filtered by pos label and original relation template
    thresholds: list of int, counts number of paraphrases with P(T) > threshold
    each line has unique subject, with the true object
    """

    # P(T) over paraphrases
    total_pos = len(df_r_pos)
    total = len(df_r)

    classification_result = len(df_r[df_r["p_t_over_paraphrases"] > threshold]) / total

    classification_result_given_pos = (
        len(df_r_pos[df_r_pos["p_t_over_paraphrases"] > threshold]) / total_pos
    )

    return classification_result, classification_result_given_pos


def calculate_macro_avg(
    df_stats: pd.DataFrame, run_name: str, dataset: str, model_name: str
) -> pd.DataFrame:
    """Calculates macro averages over relations for a given run. Returns results as a DataFrame.

    Args:
        df_stats (pd.DataFrame): _description_
        run_name (str): _description_

    Returns:
        pd.DataFrame: containing results
    """

    data = []
    data_counts = []
    w = df_stats["weight_r"].to_numpy()

    for column in df_stats.columns.to_list():
        if column.startswith("p_t"):  # all data columns
            # weighted average
            values = df_stats[column].to_numpy()
            macro_avg = (values * w).sum()
            macro_avg = macro_avg * 100  # to percentage

            data.append([run_name, dataset, model_name, column, macro_avg])

        else:  # calculate sums of weight / number of subjects, objects column
            if column != "relation":
                values = df_stats[column].to_numpy()
                sum = values.sum()
                data_counts.append([run_name, dataset, model_name, column, sum])

    df_macro_avg = pd.DataFrame(
        data, columns=["run_name", "dataset", "model", "metric", "macro_avg (%)"]
    )

    df_sums = pd.DataFrame(
        data_counts, columns=["run_name", "dataset", "model", "metric", "sum_over_rs"]
    )
    len_df = len(df_stats)
    df_stats.insert(loc=0, column="run_name", value=[run_name] * len_df)
    df_stats.insert(loc=1, column="dataset", value=[dataset] * len_df)
    df_stats.insert(loc=2, column="model", value=[model_name] * len_df)

    return df_macro_avg, df_sums, df_stats


def _permutation_stats(
    df,
    df_r,
):
    """_summary_

    Args:
        df (_type_): df subsetted by num paraphrases
        df_r (_type_): df subsetted by num paraphrases and relation

    Returns:
        _type_: _description_
    """
    # number of different objects (for a given relation)
    n_objs = df_r["obj_label"].nunique()

    # number of different subjects (for a given relation)
    n_subjs = df_r["r_s_id"].nunique()

    # number of paraphrases for r
    n_para = df_r["paraphrased_relation_template"].nunique()

    # weighting factor for macro average
    weight_r = len(df_r) / len(df)

    # n instances of a relation
    n_instances = len(df_r)

    stats = [
        n_objs,
        n_subjs,
        n_para,
        n_instances,
    ]
    return stats


def classify(
    df,
    obj_per_rel,
    paraphrased_templates,
    thresholds,
    max_paraphrases: int,
    max_o,
):
    """_summary_

    Args:
        df (pd.DataFrame): dataframe already subsetted by the number of paraphrases relevant to consider + num_relevant objects
        obj_per_rel (_type_): _description_
        paraphrased_templates (_type_): _description_
        thresholds (_type_): _description_
        max_paraphrases (_type_): _description_

    Returns:
        pd.DataFrame
    """

    classification_data, stats_data = [], []

    # Classify all P(T) and all rank_o|p_t_s_r,r_s_t(r)_i (ranks each parpahrase template separatly); in one go
    stats = _permutation_stats(df, df)  # over all relations combined
    for t in thresholds:
        results_all = evaluate_classifier(  #  for all relations # HERE
            df,
            threshold=t,
            relation="all_relations",
            max_paraphrases=max_paraphrases,
            max_o=max_o,
        )

        classification_data.extend(results_all)
        stats_data.extend([stats])

    # Classify P(T) per individual relation
    all_relations = df["orig_relation_template"].unique()
    for r in all_relations:
        df_r = df[df["orig_relation_template"] == r]

        # evaluate binary classification based on P(T) over paraphrases for each relation

        # Dataset statistics
        stats = _permutation_stats(df, df_r)
        for t in thresholds:
            # accuracy, f1, recall, precision, auc (based on P(T))
            binary_classification_results = evaluate_classifier(  #  for each relation; for T(r): one value per r,s,o
                df_r,
                threshold=t,
                relation=r,
                max_paraphrases=max_paraphrases,
                max_o=max_o,
            )

            classification_data.extend(binary_classification_results)
            stats_data.extend([stats])

            # for r_s_id seperatly
            all_r_s_ids = df_r["r_s_id"].unique()
            for r_s_id in all_r_s_ids:
                df_r_s = df_r[df_r["r_s_id"] == r_s_id]
                stats = _permutation_stats(df, df_r_s)
                result = evaluate_classifier(  #  for all relations
                    df_r_s,
                    threshold=t,
                    relation=r,
                    max_paraphrases=max_paraphrases,
                    max_o=max_o,
                    r_s_id=r_s_id,
                )
                classification_data.extend(result)
                stats_data.extend([stats])

    data = []
    for c_data, s_data in zip(classification_data, stats_data):
        data.extend([c_data + s_data])

    df_classification_results = pd.DataFrame(data, columns=df_stats_columns)

    return df_classification_results


def hist_of_all_p_t_values(
    df_pos,
    title,
    p_t_sure=None,
    p_t_very_sure=None,
    n_objs=None,
    n_subjs=None,
    n_para=None,
    weight_r=None,
):
    # histogram of p_t values
    fig = px.histogram(df_pos, x="p_t", nbins=100, title=title)
    if p_t_sure is not None:
        fig.add_annotation(
            x=0.8, y=200, text=f"P(t)>0.5: {p_t_sure:.3f}", showarrow=False, yshift=10
        )
        fig.add_annotation(
            x=0.8,
            y=200,
            text=f"P(t)>0.75: {p_t_very_sure:.3f}",
            showarrow=False,
            yshift=-10,
        )
        fig.add_annotation(
            x=0.8, y=200, text=f"n_objs: {n_objs}", showarrow=False, yshift=-30
        )
        fig.add_annotation(
            x=0.8, y=200, text=f"n_subjs: {n_subjs}", showarrow=False, yshift=-50
        )
        fig.add_annotation(
            x=0.8, y=200, text=f"n_para: {n_para}", showarrow=False, yshift=-70
        )

    return fig


def stacked_p_t_plot(df, rel, S_id, store=False):
    df_1rel = df[(df["orig_relation_template"] == rel) & (df["r_s_id"] == S_id)]
    orig_template = df_1rel.iloc[0]["orig_relation_template"]
    S = df_1rel.iloc[0]["sub_label"]
    O_true = df_1rel[df_1rel["label"] == "pos"]["obj_label"].iloc[0]

    all_p_t = df_1rel["p_t"].to_numpy()
    sum_all_p_t = sum(all_p_t)

    assert np.isclose(
        sum_all_p_t, 1
    ), f"rel: {rel}, S: {S_id}: sum_all_p_t: {sum_all_p_t} not close to 1"

    fig = px.bar(
        df_1rel,
        x=df_1rel["obj_label"],
        y="p_t",
        title=f"P(T) for '{orig_template}', X = {S}, Y_true = {O_true}",
        hover_data=["paraphrased_relation_template", "p_t"],
        color="paraphrase_id",
    )

    if store:
        fig.write_html(OUT_PATH / RUN_NAME / "P_t_distribution_{rel}_{S}.html")

    return fig, df_1rel


def hist_of_all_p_t_values(
    df_pos,
    title,
    p_t_sure=None,
    p_t_very_sure=None,
    n_objs=None,
    n_subjs=None,
    n_para=None,
    weight_r=None,
):
    # histogram of p_t values
    fig = px.histogram(df_pos, x="p_t", nbins=100, title=title)
    if p_t_sure is not None:
        fig.add_annotation(
            x=0.8, y=200, text=f"P(t)>0.5: {p_t_sure:.3f}", showarrow=False, yshift=10
        )
        fig.add_annotation(
            x=0.8,
            y=200,
            text=f"P(t)>0.75: {p_t_very_sure:.3f}",
            showarrow=False,
            yshift=-10,
        )
        fig.add_annotation(
            x=0.8, y=200, text=f"n_objs: {n_objs}", showarrow=False, yshift=-30
        )
        fig.add_annotation(
            x=0.8, y=200, text=f"n_subjs: {n_subjs}", showarrow=False, yshift=-50
        )
        fig.add_annotation(
            x=0.8, y=200, text=f"n_para: {n_para}", showarrow=False, yshift=-70
        )
    fig.show()


def trex_preprocessing(dataset_unsampled_PATH):
    print(f"Loading Trex dataset...")
    dataset = load_dataset("lama", "trex")["train"]
    print(f"Creating dataframe from dataset obj...")
    df = pd.DataFrame(dataset)
    print(f"Len hf dataset: {len(df)}")

    # clean the data
    # keep only relations: 'N-1', '1-1', ignoring 'N-M'
    df_unsampled = df[df["type"].isin(["N-1", "1-1"])]

    # drop duplicate rows, keep only first occurence of an UUID
    df_unsampled = df_unsampled.drop_duplicates()
    df_unsampled = df_unsampled.drop_duplicates(subset="uuid", keep="first")

    # save the unsampled dataset
    df_unsampled.to_hdf(dataset_unsampled_PATH, key="df", mode="w")

    return df_unsampled


def hypernymy_preprocessing(hypernymy_PATH):
    """creates the original relation template, and true S,O pairs.

    Args:
        hypernymy_PATH (JSON file): _description_

    Returns:
        pd.DataFrame: _description_
    """

    with open(hypernymy_PATH, "r") as f:
        hypernymy_examples = json.load(f)

    # OBJ = Y Hyper Fruit
    # SUBJ = X Hypo Bananas
    # Template: [X] is a [Y] .
    # Example: Diamond is a type of gem.

    hypernymy_examples["country"]

    data = []
    for hypernym, hyponyms in hypernymy_examples.items():
        for hyponym in hyponyms:
            template = f"[X] is a [Y] ."

            data.append([hyponym, hypernym, template])

    df_unsampled = pd.DataFrame(data, columns=["sub_label", "obj_label", "template"])

    return df_unsampled


def popQA_preprocessing(popQA_template_Path, df_dataset):
    """creates the original relation template, and true S,O pairs.

    Args:
        hypernymy_PATH (JSON file path): _description_
        df_dataset (pd.DataFrame): raw hf dataset as a dataframe

    Returns:
        pd.DataFrame: _description_
    """

    print(f"PARAPHRASE TEMPLATE PATH: {popQA_template_Path}")
    with open(popQA_template_Path, "r") as f:
        templates = json.load(f)

    data = []
    for r in templates.keys():
        df_relation = df_dataset[df_dataset["prop"] == r]
        subjs = df_relation["subj"].to_list()
        objs = df_relation["obj"].to_list()
        orig_templates = [templates[r][0]] * len(subjs)

        for s, o, t in zip(subjs, objs, orig_templates):
            data.append([s, o, t])

    df_unsampled = pd.DataFrame(data, columns=["sub_label", "obj_label", "template"])

    return df_unsampled


def create_paraphrase_df(df, type="trex", template_PATH=None, OUT_PATH=None):
    data = []
    if type == "trex":  # requires the original dataset from pararel
        relation_ids = df["predicate_id"].unique()  # is np array

        for r_id in relation_ids:
            df_r = df[df["predicate_id"] == r_id]
            template = df_r["template"].unique()

            assert len(template) == 1, f">1 template for {r_id}: {template}"

            type_r = df_r["type"].iloc[0]
            data.append([r_id, template[0], type_r])

        paraphrase_df = pd.DataFrame(
            data, columns=["predicate_id", "original_template", "r_type"]
        )
        PARAREL_HUMAN_TEMPLATES = Path(
            "/Users/dug/Downloads/pararel-main/data/pattern_data/graphs_json"
        )
        para_rel_patterns = read_pararel_paraphrases(PARAREL_HUMAN_TEMPLATES)

        # integrate pararel data into paraphrase_df
        data = []
        for predictate_id in paraphrase_df["predicate_id"].to_list():
            try:
                data.append(
                    para_rel_patterns[predictate_id][1:]
                )  # first is the original template
            except KeyError:
                data.append(None)
                print(f"No paraphrases for {predictate_id}")

        paraphrase_df["human_paraphrase_templates"] = data

    elif type == "PopQA":
        with open(template_PATH, "r") as f:
            templates = json.load(f)

        data = []
        for r in templates.keys():
            data.append([templates[r][0], templates[r][1:]])

        paraphrase_df = pd.DataFrame(
            data, columns=["original_template", "human_paraphrase_templates"]
        )

    elif type == "hypernymy":
        with open(template_PATH, "r") as f:
            templates = json.load(f)

        data = []
        for r in templates.keys():
            data.append([templates[r][0], templates[r][1:]])

        paraphrase_df = pd.DataFrame(
            data, columns=["original_template", "human_paraphrase_templates"]
        )

    paraphrase_df.to_hdf(OUT_PATH, key="df")

    return paraphrase_df


def read_pararel_paraphrases(PARAREL_HUMAN_TEMPLATES):
    patterns = {}
    for f in list(Path.iterdir(PARAREL_HUMAN_TEMPLATES)):
        with open(f) as in_file:
            r_data = []
            for line in in_file:
                r_data.append(json.loads(line))

        paraphrases = [x["pattern"] for x in r_data]
        patterns[f.stem] = paraphrases

    return patterns


def scatterplots(df, y, x, title, out_path=False):
    fig = px.scatter(
        df,
        y=y,
        x=x,
        color="label",
        text="obj_label",
        hover_data=df.columns,
        title=title,
    )
    fig.update_traces(textposition="top center")
    if out_path:
        fig.write_html(out_path)

    return fig


def boxplots(df, y, x, title, out_path=False):
    fig = px.box(
        df,
        y=y,
        x=x,
        color="label",
        points="all",
        hover_data=["obj_label"],
        title=title,
    )

    if out_path:
        fig.write_html(out_path)

    return fig


def _get_classification_results(
    labels: list,
    predictions_binary: np.array,
) -> dict:
    cm = confusion_matrix(
        labels,
        predictions_binary,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()
    # Calculate Precision, Recall, F1, Accuracy
    # and handle scalar division warnings
    if tp == 0 and fp == 0:
        precision = np.nan
    else:
        precision = tp / (tp + fp)

    if tp == 0 and fn == 0:
        recall = np.nan
    else:
        recall = tp / (tp + fn)

    if fp == 0 and fn == 0:
        fpr = np.nan
    else:
        fpr = fp / (fp + tn)

    if precision == 0 and recall == 0:
        f1 = np.nan
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    if tp == 0 and tn == 0 and fp == 0 and fn == 0:
        accuracy = np.nan
    else:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    total = tn + fp + fn + tp

    return precision, recall, f1, accuracy, tp, tn, fp, fn, fpr, total


def evaluate_classifier(  # HERE
    df: pd.DataFrame,
    threshold: float,
    relation: str,
    max_paraphrases: int,
    max_o: int,
    r_s_id: int = None,
) -> dict:
    """Evaluates predictions of a binary classifier.

    Args:
        predictions (np.array): P(T) over paraphrases; dimension: (n,)
        predictions_argmax (np.array): binary predictions based on argmax of 'rank_o'; dimension: (n,) or rank_o|p_t_s_r,r_s_t(r)_i
        labels (list): ground truth for each data point, normally list of labels with ['pos', 'neg']; but can be any binary labels
        df_r: already filtered by original relation template; and num_paraphrases; contains only instances with P(T) over paraphrases
        df_r_pos: already filtered by pos label and original relation template; contains only instances P(T) over paraphrases
        each line has unique subject, with the true object
        threshold (list): thresholds for binary classification

    Returns:
        dict: metric: value
    """
    results = []
    labels_pp = df["label"].to_list()

    df_global = df[df["paraphrase_id"] == 0]
    labels_global = df_global["label"].to_list()
    for el in set(labels_pp).union(set(labels_global)):
        if el not in ["pos", "neg"]:
            raise AssertionError(
                f"Labels are not in format 'neg' / 'pos': {set(labels)}"
            )

    # ALL: per para
    labels_pp = np.array(
        [1 if l == "pos" else 0 for l in labels_pp]
    )  # 1 = pos, 0 = neg
    predictions_pp = df["p_t_s_r"].to_numpy()
    predictions_argmax_pp = df["rank_o|p_t_s_r,r_s_t(r)_i"].to_numpy()
    predictions_argmax_pp = np.where(predictions_argmax_pp == 1, 1, 0)
    # SELECTIVE: filtering per para
    predictions_argmax_binary_pp_selective = predictions_argmax_pp[
        predictions_pp > threshold
    ]
    labels_pp_selective = labels_pp[predictions_pp > threshold]
    predictions_pp_binary_selective = np.ones_like(labels_pp_selective)

    # ALL: global
    labels_global = np.array([1 if l == "pos" else 0 for l in labels_global])
    predictions_p_o_global = df_global["p_t_over_paraphrases"].to_numpy()
    predictions_argmax_global = df_global["rank_o"].to_numpy()
    predictions_argmax_global = np.where(predictions_argmax_global == 1, 1, 0)
    # SELECTIVE: filtering global
    selected_probs = predictions_p_o_global[predictions_p_o_global > threshold]
    predictions_argmax_binary_global_selective = predictions_argmax_global[
        predictions_p_o_global > threshold
    ]
    labels_global_selective = labels_global[predictions_p_o_global > threshold]
    predictions_global_binary_selective = np.ones_like(labels_global_selective)

    # Metrics that are threshold "independent":
    if threshold == 0.0:
        # AUROC, AUPRC
        if (
            len(set(labels_global).union(set(labels_pp))) == 1
        ):  # AUC is not defined for single class
            fpr, tpr, roc_thresholds, auc = np.nan, np.nan, np.nan, np.nan
            fpr_argmax, tpr_argmax, roc_thresholds_argmax, auc_argmax = (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )
        else:
            # for P(O) / global
            (
                fpr_by_threshold_p_o_global,
                tpr_by_threshold_p_o_global,
                roc_thresholds_p_o_global,
            ) = roc_curve(labels_global, predictions_p_o_global)
            roc_thresholds_p_o_global = roc_thresholds_p_o_global.astype(np.float64)
            auc_p_o_global = roc_auc_score(labels_global, predictions_p_o_global)

            # for rank / global
            (
                fpr_by_threshold_argmax_global,
                tpr_by_threshold_argmax_global,
                roc_thresholds_argmax_global,
            ) = roc_curve(labels_global, predictions_argmax_global)
            auc_argmax_global = roc_auc_score(labels_global, predictions_argmax_global)

            # for rank / pp
            (
                fpr_by_threshold_argmax_pp,
                tpr_by_threshold_argmax_pp,
                roc_thresholds_argmax_pp,
            ) = roc_curve(labels_pp, predictions_argmax_pp)
            auc_argmax_pp = roc_auc_score(labels_pp, predictions_argmax_pp)

    else:
        fpr_by_threshold_p_o_global = np.nan
        tpr_by_threshold_p_o_global = np.nan
        roc_thresholds_p_o_global = np.nan
        auc_p_o_global = np.nan
        fpr_by_threshold_argmax_global = np.nan
        tpr_by_threshold_argmax_global = np.nan
        roc_thresholds_argmax_global = np.nan
        auc_argmax_global = np.nan
        fpr_by_threshold_argmax_pp = np.nan
        tpr_by_threshold_argmax_pp = np.nan
        roc_thresholds_argmax_pp = np.nan
        auc_argmax_pp = np.nan

    # P(O) selective prediction global: considering predictions only for P(T) > threshold; rest is ignored # OK
    coverage_abs_global = sum(predictions_global_binary_selective)
    coverage_rel = coverage_abs_global / len(labels_global)
    (
        precision_selective_global,
        recall_selective_global,
        f1_selective_global,
        accuracy_selective_global,
        tp_selective_global,
        tn_selective_global,
        fp_selective_global,
        fn_selective_global,
        fpr_selective_global,
        total_selective_global,
    ) = _get_classification_results(
        labels_global_selective, predictions_global_binary_selective
    )

    # P(O) selective prediction PP: considering predictions only for P(T) > threshold; rest is ignored

    coverage_abs_pp = sum(predictions_pp_binary_selective)
    coverage_rel_pp = coverage_abs_pp / len(labels_pp)

    (
        precision_selective_pp,
        recall_selective_pp,
        f1_selective_pp,
        accuracy_selective_pp,
        tp_selective_pp,
        tn_selective_pp,
        fp_selective_pp,
        fn_selective_pp,
        fpr_selective_pp,
        total_selective_pp,
    ) = _get_classification_results(
        labels_pp_selective, predictions_pp_binary_selective
    )

    # RANK_selective / global: classify predictions based on argmax of rank_o
    (
        precision_argmax_selective_global,
        recall_argmax_selective_global,  # precision_argmax_global
        f1_argmax_selective_global,
        accuracy_argmax_selective_global,
        tp_argmax_selective_global,
        tn_argmax_selective_global,
        fp_argmax_selective_global,
        fn_argmax_selective_global,
        fpr_argmax_selective_global,
        total_argmax_selective_global,
    ) = _get_classification_results(
        labels_global_selective, predictions_argmax_binary_global_selective
    )

    # RANK_selective / pp
    (
        precision_argmax_selective_pp,
        recall_argmax_selective_pp,
        f1_argmax_selective_pp,
        accuracy_argmax_selective_pp,  # precision_argmax_pp
        tp_argmax_selective_pp,
        tn_argmax_selective_pp,
        fp_argmax_selective_pp,
        fn_argmax_selective_pp,
        fpr_argmax_selective_pp,
        total_argmax_selective_pp,
    ) = _get_classification_results(
        labels_pp_selective, predictions_argmax_binary_pp_selective
    )

    # RANK_overall / pp
    (
        precision_argmax_overall_pp,
        recall_argmax_overall_pp,
        f1_argmax_overall_pp,
        accuracy_argmax_overall_pp,
        tp_argmax_overall_pp,
        tn_argmax_overall_pp,
        fp_argmax_overall_pp,
        fn_argmax_overall_pp,
        fpr_argmax_overall_pp,
        total_argmax_overall_pp,
    ) = _get_classification_results(labels_pp, predictions_argmax_pp)

    # RANK_overall / global
    (
        precision_argmax_overall_global,
        recall_argmax_overall_global,
        f1_argmax_overall_global,
        accuracy_argmax_overall_global,
        tp_argmax_overall_global,
        tn_argmax_overall_global,
        fp_argmax_overall_global,
        fn_argmax_overall_global,
        fpr_argmax_overall_global,
        total_argmax_overall_global,
    ) = _get_classification_results(labels_global, predictions_argmax_global)

    # Combine all results
    results.append(
        [
            relation,
            r_s_id,
            max_paraphrases,
            max_o,
            threshold,
            coverage_abs_global,
            coverage_rel,
            precision_selective_global,
            recall_selective_global,
            f1_selective_global,
            accuracy_selective_global,
            tp_selective_global,
            tn_selective_global,
            fp_selective_global,
            fn_selective_global,
            fpr_selective_global,
            total_selective_global,
            coverage_abs_pp,
            coverage_rel_pp,
            precision_selective_pp,
            recall_selective_pp,
            f1_selective_pp,
            accuracy_selective_pp,
            tp_selective_pp,
            tn_selective_pp,
            fp_selective_pp,
            fn_selective_pp,
            fpr_selective_pp,
            total_selective_pp,
            auc_p_o_global,
            fpr_by_threshold_p_o_global,
            tpr_by_threshold_p_o_global,
            roc_thresholds_p_o_global,
            precision_argmax_selective_global,
            recall_argmax_selective_global,  # precision_argmax_global
            f1_argmax_selective_global,
            accuracy_argmax_selective_global,
            tp_argmax_selective_global,
            tn_argmax_selective_global,
            fp_argmax_selective_global,
            fn_argmax_selective_global,
            fpr_argmax_selective_global,
            total_argmax_selective_global,
            fpr_by_threshold_argmax_global,
            tpr_by_threshold_argmax_global,
            roc_thresholds_argmax_global,
            auc_argmax_global,
            precision_argmax_selective_pp,
            recall_argmax_selective_pp,
            f1_argmax_selective_pp,
            accuracy_argmax_selective_pp,  # precision_argmax_pp
            tp_argmax_selective_pp,
            tn_argmax_selective_pp,
            fp_argmax_selective_pp,
            fn_argmax_selective_pp,
            fpr_argmax_selective_pp,
            total_argmax_selective_pp,
            fpr_by_threshold_argmax_pp,
            tpr_by_threshold_argmax_pp,
            roc_thresholds_argmax_pp,
            auc_argmax_pp,
            precision_argmax_overall_pp,
            recall_argmax_overall_pp,
            f1_argmax_overall_pp,
            accuracy_argmax_overall_pp,
            tp_argmax_overall_pp,
            tn_argmax_overall_pp,
            fp_argmax_overall_pp,
            fn_argmax_overall_pp,
            fpr_argmax_overall_pp,
            total_argmax_overall_pp,
            precision_argmax_overall_global,
            recall_argmax_overall_global,
            f1_argmax_overall_global,
            accuracy_argmax_overall_global,
            tp_argmax_overall_global,
            tn_argmax_overall_global,
            fp_argmax_overall_global,
            fn_argmax_overall_global,
            fpr_argmax_overall_global,
            total_argmax_overall_global,
        ]
    )

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="data preparation for exp_3_set_proba")

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["trex", "hypernymy", "PopQA"],
        default="trex",
        help="Original dataset name (trex, hypernymy)",
    )

    parser.add_argument(
        "--s_contexts",
        type=str,
        default=None,
        help="Path to context texts per subject for hypernymy dataset",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output path folder",
    )

    parser.add_argument(
        "--paraphrase_templates",
        type=str,
        required=True,
        help="Path to paraphrase template json file, or hf5 file for trex.",
    )

    parser.add_argument(
        "--n_instances_per_r",
        type=int,
        default=100,
        help="Number of instances (S) per relation type",
    )

    parser.add_argument(
        "--equal_n_o",
        action="store_true",
        help="Equal number of o_neg objects per relation",
    )
    parser.add_argument(
        "--num_o", type=int, default=100, help="Maximum number of o_neg objects"
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=8,
        help="Number of jobs for multiprocessing",
    )

    parser.add_argument(
        "--grammar_postprocessing",
        action="store_true",
        help="Activate grammar postprocessing of permutated sequences",
    )

    parser.add_argument(
        "--LM_postprocessing",
        action="store_true",
        help="Use T5 model for postprocessing of permutated sequences",
    )

    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable wandb logging",
    )

    parser.add_argument(
        "--wandb_run_name",
        type=str,
        required=True,
    )

    # encode
    parser.add_argument(
        "--example_column",
        type=str,
        choices=["all", "same_paraphrase"],
        default="demos_same_paraphrase",
        help="Column name in the dataframe that contains the demonstration examples. Only used when n_shot_examples > 0.",
    )
    parser.add_argument(
        "--n_shot_examples",
        type=int,
        default=0,
        help="Number of demonstration examples to sample per s,r instance. 0 to disable.",
    )

    parser.add_argument(
        "--n_shot_examples_negative",
        type=int,
        default=0,
        help="Number of negative demonstration examples to sample per relation. 0 to disable.",
    )

    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")

    # analyze
    parser.add_argument(
        "--max_n_paraphrases",
        nargs="+",
        type=int,
        default=[0],
        help="Create P(T) results separatly, considering each int as the maximum number of paraphrases to consider",
    )
    parser.add_argument(
        "--max_n_objects",
        nargs="+",
        type=int,
        default=[-2],
        help="Create P(T) results separatly, considering each int as the maximum number of objects to consider per subject",
    )

    parser.add_argument(
        "--classification_thresholds",
        nargs="+",
        type=float,
        default=[0.1, 0.25, 0.5, 0.75, 0.9],
        help="Create P(T) results separatly, considering each int as the maximum number of paraphrases to consider",
    )

    args = parser.parse_args()

    return args


def get_df_overview_stats(df, permutations_df=False):
    # group by template
    if permutations_df:
        relation_column = "orig_relation_template"
    else:
        relation_column = "template"

    relation_templates = df[relation_column].unique()

    # sum number of objects, number of subjects per template, count labels
    data = []
    for relation in relation_templates:
        df_template = df[df[relation_column] == relation]
        if permutations_df:
            n_pos = len(df_template[df_template["label"] == "pos"])
            n_neg = len(df_template[df_template["label"] == "neg"])

            o_neg_size = df_template[df_template["label"] == "neg"][
                "o_permutation_n"
            ].nunique()

        else:
            n_pos = len(df_template)
            n_neg = 0
            o_neg_size = 0

        n_objects = df_template["obj_label"].nunique()
        n_subjects = df_template["sub_label"].nunique()

        total_instances = len(df_template)
        data.append(
            [relation, n_objects, n_subjects, o_neg_size, n_pos, n_neg, total_instances]
        )

    data_df = pd.DataFrame(
        data,
        columns=[
            "relation",
            "\#o",
            "\#s",
            "$|O^-|$",
            "\#pos",
            "\#neg",
            "total",
        ],
    )

    # data_df.loc["total"] = data_df.sum(numeric_only=True)
    # data_df.loc["total", "relation"] = "all relations"

    return data_df


def plot_roc_curve(df_auc, out_path=None, use_legend=False):
    tprs = df_auc["tpr_by_threshold_global"].to_list()
    fprs = df_auc["fpr_by_threshold_global"].to_list()
    aucs = df_auc["auc_global"].to_list()
    relations = df_auc["relation"].to_list()
    datasets = df_auc["dataset"].to_list()
    models = df_auc["model"].to_list()
    max_paraphrases = df_auc["max_paraphrases"].to_list()
    max_os = df_auc["max_o"].to_list()
    thresholds = df_auc["threshold"].to_list()

    if use_legend:
        legends = df_auc["legend"].to_list()
    else:
        legends = models

    fig = go.Figure()

    for fpr, tpr, auc, relation, max_p, max_o, threshold, dataset, model, legend in zip(
        fprs,
        tprs,
        aucs,
        relations,
        max_paraphrases,
        max_os,
        thresholds,
        datasets,
        models,
        legends,
    ):
        if use_legend:
            name = f"{legend}, auc: {auc:.3f}"
        else:
            name = f"{model}, {dataset}, max_p: {max_p}, auc: {auc:.3f})"

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=name,
            )
        )

    fig.update_layout(
        title="",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend_title="Configurations",
    )

    if out_path:
        fig.write_html(str(out_path) + ".html")
        pio.write_image(fig, str(out_path) + ".pdf")

    return fig


def plot_coverage_risk_curve(df, risk_name, out_path=None):
    coverages, risks, thresholds, titles = [], [], [], []
    for r in df["relation"].unique():
        df_r = df[df["relation"] == r]

        for max_p in df_r["max_paraphrases"].unique():
            df_r_p = df_r[df_r["max_paraphrases"] == max_p]

            for max_o in df_r_p["max_o"].unique():
                df_r_p_o = df_r_p[df_r_p["max_o"] == max_o]

                coverage = df_r_p_o["coverage_rel"].to_numpy()
                risk = df_r_p_o[risk_name].to_numpy()
                threshold = df_r_p_o["threshold"].to_numpy()
                title = [f"{r}, {max_p}, {max_o}, {threshold}"]

                coverages.append(coverage)
                risks.append(risk)
                thresholds.append(threshold)
                titles.append(title)

    fig = go.Figure()
    for i in range(len(coverages)):
        fig.add_trace(
            go.Scatter(
                x=coverages[i], y=risks[i], mode="lines+markers", name=str(titles[i])
            )
        )
        fig.update_layout(
            title=f"coverage vs {risk_name}",
            xaxis_title="coverage",
            yaxis_title=risk_name,
            legend_title="relations",
        )

    if out_path:
        fig.write_html(str(out_path) + ".html")
        pio.write_image(fig, str(out_path) + ".pdf")

    return fig


def plot_coverage_risk_curve_2(
    df,
    risk_name,
    coverage_name,
    mode="overall",
    plot_title="",
    out_path=None,
    legend=False,
    r_s_id=None,
    plot_all_p_only=False,
):
    # for GLOBAL RISK COverage curve
    coverages, risks, thresholds, titles, aucs, data, max_ps = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for r in df["run_name"].unique():
        print(f"Plotting run: {r}")
        df_r = df[df["run_name"] == r]
        dataset = df_r["dataset"].unique()[0]
        run_attribute = df_r["run_attributes"].unique()[0]
        model = df_r["model"].unique()[0]

        if mode == "by_relation":
            print(f"Plotting by relation")
            for relation in df_r["relation"].unique():
                df_rel = df_r[df_r["relation"] == relation]

                for max_p in df_rel["max_paraphrases"].unique():
                    df_r_p = df_rel[df_rel["max_paraphrases"] == max_p]

                    coverage = df_r_p[coverage_name].to_numpy()
                    risk = df_r_p[risk_name].to_numpy()
                    threshold = df_r_p["threshold"].to_numpy()

                    # avoid risk values of nan for auc calculation:
                    while np.isnan(risk[-1]):
                        risk = risk[:-1]
                        coverage = coverage[:-1]
                        threshold = threshold[:-1]

                    coverage_desc = np.sort(coverage)[::-1]
                    assert np.allclose(
                        coverage_desc, coverage
                    ), f"Coverage is not sorted in descending order."

                    auc_value = auc(
                        coverage,
                        risk,
                    )  # x array (coverage) can be arbitrary spaced, but needs to be sorted ascending

                    aucs.append(auc_value)

                    coverages.append(coverage)
                    risks.append(risk)
                    thresholds.append(threshold)
                    max_ps.append(max_p)

                    if legend:
                        title_legend = df_r_p["legend"].unique()[0]
                        title = f"{title_legend}, auc: {auc_value:.3f}"
                    else:
                        title = f"{dataset}/{relation} {run_attribute}, #p: {max_p}, auc: {auc_value:.3f}"
                    titles.append(title)

                    data.append(
                        [
                            dataset,
                            model,
                            run_attribute,
                            relation,
                            r_s_id,
                            max_p,
                            risk_name,
                            auc_value,
                        ]
                    )

        else:
            for max_p in df_r["max_paraphrases"].unique():
                df_r_p = df_r[df_r["max_paraphrases"] == max_p]

                # avoid risk = 0 for auc calculation:
                coverage = df_r_p[coverage_name].to_numpy()[:-1]
                risk = df_r_p[risk_name].to_numpy()[:-1]
                threshold = df_r_p["threshold"].to_numpy()[:-1]

                if np.isnan(risk[-1]):
                    risk = risk[:-1]
                    coverage = coverage[:-1]
                    threshold = threshold[:-1]

                coverage_desc = np.sort(coverage)[::-1]
                assert np.allclose(
                    coverage_desc, coverage
                ), f"Coverage is not sorted in descending order."

                auc_value = auc(
                    coverage,
                    risk,
                )  # x array (coverage) can be arbitrary spaced, but needs to be sorted ascending

                aucs.append(auc_value)

                coverages.append(coverage)
                risks.append(risk)
                thresholds.append(threshold)
                max_ps.append(max_p)

                if legend:
                    title_legend = df_r_p["legend"].unique()[0]
                    title = f"{title_legend}, auc: {auc_value:.3f}"
                else:
                    title = (
                        f"{dataset} {run_attribute}, #p: {max_p}, auc: {auc_value:.3f}"
                    )
                titles.append(title)

                data.append(
                    [
                        dataset,
                        model,
                        run_attribute,
                        "all_relations",
                        max_p,
                        risk_name,
                        auc_value,
                    ]
                )

    fig = go.Figure()
    for i in range(len(coverages)):
        if plot_all_p_only:
            if max_ps[i] > 0:  # plot only with more than 0 paraphrase
                fig.add_trace(
                    go.Scatter(
                        x=coverages[i],
                        y=risks[i],
                        mode="lines+markers",
                        name=str(titles[i]),
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=coverages[i],
                    y=risks[i],
                    mode="lines+markers",
                    name=str(titles[i]),
                )
            )

    fig.update_layout(
        title=plot_title,
        xaxis_title="coverage",
        yaxis_title=risk_name,
        legend_title="relations",
        font_family="Serif",
        font_size=12,
        yaxis_title_text="selective precision",
    )

    if out_path:
        fig.write_html(str(out_path) + ".html")
        pio.write_image(fig, str(out_path) + ".pdf")
        print(str(out_path) + ".pdf")

    return fig, data


def create_df_probability_vis(
    df_instance_permutations, r_s_id, PLOT_PATH=False, template_column="p_t_s_r"
):

    df_example = df_instance_permutations[
        (df_instance_permutations["r_s_id"] == r_s_id)
    ]

    # replace [X] and [Y] with [s] and [o]
    paraphrased_templates = df_example["paraphrased_relation_template"].to_list()
    df_example["paraphrased_relation_template"] = [
        paraphrased_templates[i].replace("[X]", "[s]").replace("[Y]", "[o]")
        for i in range(len(paraphrased_templates))
    ]

    df_example["rank"] = df_example["rank_o|p_t_s_r,r_s_t(r)_i"].to_numpy()

    df_example["p"] = df_example[template_column].to_numpy()

    df_global = df_example[
        df_example["paraphrase_id"] == 0
    ]  # where P(T) over paraprhases is calculated
    subject = df_global["sub_label"].unique()[0]
    df_global["paraphrased_relation_template"] = (
        r"$\frac{1}{H}\sum_{h=1}^{H} P(o|t(r)_h, s)$"
    )
    df_global["p"] = df_global["p_t_over_paraphrases"].to_numpy()
    df_global["rank"] = df_global["rank_o"].to_numpy()

    # ignore too low values for better visualization, and subset to only a few object labels
    # df_global = df_global[df_global["paraphrase_id"].isin([0, 1])]
    # df_global = df_global[df_global["obj_label"].isin(['politician', 'surgeon', 'lawyer', 'farmer', 'dentist', 'model'])]
    # df_demo = df_demo[df_demo["paraphrase_id"].isin([0, 1])]
    # df_demo = df_demo[df_demo["obj_label"].isin(['politician', 'surgeon', 'lawyer', 'farmer', 'dentist', 'model'])]

    # concatenate the dataframes
    df_vis = pd.concat([df_example, df_global], ignore_index=True)

    fig = px.scatter(
        df_vis,
        x="p",
        y="paraphrased_relation_template",
        color="obj_label",
        text="label",
        title="",
        hover_data=["obj_label"],
    )
    fig.update_traces(textposition="top center")

    fig.update_layout(
        xaxis_title_text=rf"$P(o|t(r), s=\text{{{subject}}})$",  # yaxis label
        yaxis_title_text="",
        legend_title_text=rf"$O$",
        font_family="Serif",
        font_size=12,
        # margin_l=5, margin_t=5, margin_b=5, margin_r=5
        # width =1000, height=500,
    )

    if PLOT_PATH:
        pio.write_image(
            fig,
            PLOT_PATH,
            width=1.5 * 600,
            height=2.0 * 600,
        )
        print(PLOT_PATH)

    return df_vis, fig


def calculate_entropies(
    run_names, dataset_per_run, model_per_run, pk_metrics, BASE_PATH
):
    data = []
    for i in range(len(run_names)):
        _, df_instance_permutations = get_data_permutations(run_names[i], BASE_PATH)
        dataset = dataset_per_run[i]
        model = model_per_run[i]
        relations = list(
            df_instance_permutations["orig_relation_template"].unique()
        ) + ["all_relations"]

        for relation in relations:
            if relation == "all_relations":
                df_r = df_instance_permutations
            else:
                df_r = df_instance_permutations[
                    df_instance_permutations["orig_relation_template"] == relation
                ]

            for pk_metric in pk_metrics:
                if pk_metric == "p_t_over_paraphrases":
                    pk = df_r[~df_r["p_t_over_paraphrases"].isna()][
                        "p_t_over_paraphrases"
                    ].to_numpy()
                else:
                    pk = df_r[pk_metric].to_numpy()
                H = entropy(pk)
                data.append([model, dataset, relation, pk_metric, H])

    df_entropy = pd.DataFrame(
        data, columns=["model", "dataset", "relation", "pk_metric", "entropy"]
    )

    return df_entropy


df_stats_columns = [
    "relation",
    "r_s_id",
    "max_paraphrases",
    "max_o",
    "threshold",
    "coverage_abs_global",
    "coverage_rel_global",
    "precision_selective_global",
    "recall_selective_global",
    "f1_selective_global",
    "accuracy_selective_global",
    "tp_selective_global",
    "tn_selective_global",
    "fp_selective_global",
    "fn_selective_global",
    "fpr_selective_global",
    "total_selective_global",
    "coverage_abs_pp",
    "coverage_rel_pp",
    "precision_selective_pp",
    "recall_selective_pp",
    "f1_selective_pp",
    "accuracy_selective_pp",
    "tp_selective_pp",
    "tn_selective_pp",
    "fp_selective_pp",
    "fn_selective_pp",
    "fpr_selective_pp",
    "total_selective_pp",
    "auc_p_o_global",
    "fpr_by_threshold_p_o_global",
    "tpr_by_threshold_p_o_global",
    "roc_thresholds_p_o_global",
    "precision_argmax_selective_global",
    "recall_argmax_selective_global",
    "f1_argmax_selective_global",
    "accuracy_argmax_selective_global",
    "tp_argmax_selective_global",
    "tn_argmax_selective_global",
    "fp_argmax_selective_global",
    "fn_argmax_selective_global",
    "fpr_argmax_selective_global",
    "total_argmax_selective_global",
    "fpr_by_threshold_argmax_global",
    "tpr_by_threshold_argmax_global",
    "roc_thresholds_argmax_global",
    "auc_argmax_global",
    "precision_argmax_selective_pp",
    "recall_argmax_selective_pp",
    "f1_argmax_selective_pp",
    "accuracy_argmax_selective_pp",
    "tp_argmax_selective_pp",
    "tn_argmax_selective_pp",
    "fp_argmax_selective_pp",
    "fn_argmax_selective_pp",
    "fpr_argmax_selective_pp",
    "total_argmax_selective_pp",
    "fpr_by_threshold_argmax_pp",
    "tpr_by_threshold_argmax_pp",
    "roc_thresholds_argmax_pp",
    "auc_argmax_pp",
    "precision_argmax_overall_pp",
    "recall_argmax_overall_pp",
    "f1_argmax_overall_pp",
    "accuracy_argmax_overall_pp",
    "tp_argmax_overall_pp",
    "tn_argmax_overall_pp",
    "fp_argmax_overall_pp",
    "fn_argmax_overall_pp",
    "fpr_argmax_overall_pp",
    "total_argmax_overall_pp",
    "precision_argmax_overall_global",
    "recall_argmax_overall_global",
    "f1_argmax_overall_global",
    "accuracy_argmax_overall_global",
    "tp_argmax_overall_global",
    "tn_argmax_overall_global",
    "fp_argmax_overall_global",
    "fn_argmax_overall_global",
    "fpr_argmax_overall_global",
    "total_argmax_overall_global",
    "n_objs",
    "n_subjs",
    "n_para",
    "n_instances",
]


def combine_stats_dfs(
    run_names, dataset_per_run, model_per_run, BASE_PATH, run_attributes=None
):
    # combine stats to 1 df + add dataset, model, run_name
    for i in range(len(run_names)):
        df_stats, df_instance_permutations = get_data(run_names[i], BASE_PATH)
        df_stats["dataset"] = dataset_per_run[i]
        df_stats["model"] = model_per_run[i]
        df_stats["run_name"] = run_names[i]

        max_p = df_stats["max_paraphrases"].to_list()
        df_stats["max_paraphrases"] = [20 if m > 0 else 0 for m in max_p]

        if run_attributes:
            df_stats["run_attributes"] = run_attributes[i]
        else:
            df_stats["run_attributes"] = ""

        # concat all stats
        if i == 0:
            df_all_stats = df_stats
        else:
            df_all_stats = pd.concat([df_all_stats, df_stats])

    return df_all_stats


def save_plot(fig, BASE_PATH, filename="figure"):
    fig.update_layout(
        font_family="Serif", font_size=12, yaxis_title_text="selective precision"
    )
    pio.write_image(fig, BASE_PATH / f"{filename}.pdf", width=3 * 300, height=1.5 * 300)
    fig.write_html(BASE_PATH / f"{filename}.html")
    print(f"Saved to {BASE_PATH} / {filename}.pdf")


def get_data(run_name, BASE_PATH):
    OUT_PATH = BASE_PATH / run_name

    df_stats = pd.read_parquet(
        OUT_PATH / "classification_results_all_thresholds.parquet"
    )
    df_instance_permutations = pd.read_parquet(
        OUT_PATH / "permutations_scores_p_t_all.parquet"
    )

    # paraphrase_df = pd.read_hdf(OUT_PATH / "paraphrase_df.hf5")
    # df_processed = pd.read_hdf(OUT_PATH / "cleaned.hf5")
    # df_unsampled = pd.read_hdf(OUT_PATH / "unsampled.hf5")

    # with open(OUT_PATH / "paraphrased_templates.json", "r") as f:
    #     paraphrased_templates = json.load(f)

    # with open(OUT_PATH / "o_neg_sets.json", "r") as f:
    #     o_neg_sets = json.load(f)

    # return df_stats, paraphrase_df, df_instance_permutations, df_processed, df_unsampled, paraphrased_templates, o_neg_sets

    return df_stats, df_instance_permutations


def subject_overview2(
    df_stats,
    df_instance_permutations,
    threshold=0.5,
    metric="precision_selective_global",
):
    # stats for s,o pairs
    df_stats_s = df_stats[
        (~df_stats["relation"].str.contains("all_relations"))
        & (~df_stats["r_s_id"].isna())
    ]
    df_stats_s = df_stats_s[
        (df_stats_s["threshold"] == threshold) & (df_stats_s["max_paraphrases"] > 0)
    ]

    # get top=1 predictions according to rank
    df_rank = df_instance_permutations[df_instance_permutations["rank_o"] == 1]
    df_rank["argmax_o"] = df_rank["obj_label"]
    df_rank = df_rank[["r_s_id", "argmax_o"]]

    # keep only Positive r,s,o combinations
    permutations_pos = df_instance_permutations[
        (df_instance_permutations["label"] == "pos")
        & (df_instance_permutations["paraphrase_id"] == 0)
    ]
    # df_stats_s = df_stats_s.sort_values(
    #     by=["precision_selective_global"], ascending=False
    # )

    # merge by r_s_id to get rank_o
    permutations_pos = permutations_pos.merge(df_rank, on="r_s_id", how="left")
    assert len(permutations_pos) == len(
        df_rank
    ), f"Not all r_s_id are merged: len(df_rank)={len(df_rank)}, len(permutations_pos)={len(permutations_pos)}"

    # associate r_s_id with subject and relation
    merged_stats = df_stats_s.merge(permutations_pos, on="r_s_id", how="left")
    assert len(merged_stats) == len(df_stats_s) and len(permutations_pos) == len(
        df_stats_s
    ), f"Not all r_s_id are merged: len(df_stats_s)={len(df_stats_s)}, len(merged_stats)={len(merged_stats)}"

    # set nan values to 0
    merged_stats.fillna(0, inplace=True)

    # add a label for correct and incorrect knowledge
    values = merged_stats[metric].to_list()
    merged_stats["category"] = [
        "Knowledge is Correct" if v == 1.0 else "Knowledge is Incorrect" for v in values
    ]
    merged_stats["s, o"] = merged_stats["sub_label"] + ", " + merged_stats["obj_label"]

    return merged_stats


def subject_overview(
    df_stats,
    df_instance_permutations,
    threshold=0.5,
):

    covered = df_instance_permutations[
        df_instance_permutations["p_t_over_paraphrases"] > threshold
    ]
    covered["s, o"] = (
        df_instance_permutations["sub_label"]
        + ", "
        + df_instance_permutations["obj_label"]
    )
    permutations_pos = df_instance_permutations[
        (df_instance_permutations["label"] == "pos")
        & (df_instance_permutations["paraphrase_id"] == 0)
    ]
    permutations_pos["true_o"] = permutations_pos["obj_label"]
    permutations_pos = permutations_pos[["r_s_id", "true_o"]]

    # merge by r_s_id to get ground truth objec
    covered = covered.merge(permutations_pos, on="r_s_id", how="left")
    obj_labels = covered["obj_label"].to_list()
    true_os = covered["true_o"].to_list()
    covered["category"] = [
        (
            "Knowledge is Correct"
            if true_os[i] == obj_labels[i]
            else "Knowledge is Incorrect"
        )
        for i in range(len(obj_labels))
    ]

    return covered


def convert_permutations_for_plotting(df):
    """_summary_

    Args:
        df (_type_): df_instance_permutations type

    Returns:
        _type_: _description_
    """

    # replace [X] and [Y] with [s] and [o] in relation templates
    paraphrased_templates = df["paraphrased_relation_template"].to_list()
    df["paraphrased_relation_template"] = [
        paraphrased_templates[i].replace("[X]", "[s]").replace("[Y]", "[o]")
        for i in range(len(paraphrased_templates))
    ]
    orig_templates = df["orig_relation_template"].to_list()
    df["orig_relation_template"] = [
        orig_templates[i].replace("[X]", "[s]").replace("[Y]", "[o]")
        for i in range(len(orig_templates))
    ]

    # make obj labels more readable
    obj_label = df["obj_label"].to_list()
    labels = df["label"].to_list()
    labels = ["+" if o == "pos" else "-" for o in labels]
    df["obj_label"] = [
        rf"$\text{{{o}}} \in O^{{{l}}}$" for o, l in zip(obj_label, labels)
    ]

    return df


def rename_metrics(df):
    # rename columns
    # metric names
    if "coverage_rel_global" in df.columns:
        df.rename(columns={"coverage_rel_global": "coverage"}, inplace=True)
    if "precision_selective_global" in df.columns:
        df.rename(
            columns={"precision_selective_global": "selective precision"}, inplace=True
        )
    if "precision_argmax_overall_global" in df.columns:
        df.rename(
            columns={"precision_argmax_overall_global": "argmax precision"},
            inplace=True,
        )

    return df


def convert_for_pdf(df, para_expl=True):
    """_summary_

    Args:
        df (_type_): df of type stats / or for tables
    """

    # replace gpt-l with GPT-2-L
    if "model" in df.columns:
        models = df["model"].to_list()
        df["model"] = [
            i.replace("gpt-l", "GPT-2-L").replace("mistral-7B", "Mistral-7B-I")
            for i in models
        ]

    # all or 0 instead of max_paraphrases
    try:
        num_para = df["max_paraphrases"].to_list()
        if not para_expl:
            num_para = ["all" if x > 0 else "0" for x in num_para]
        df["#p"] = num_para
    except KeyError:
        print(f"Df columns: {df.columns}")
        pass

    # relations
    if "paraphrased_relation_template" in df.columns:
        paraphrased_templates = df["paraphrased_relation_template"].to_list()
        df["paraphrased_relation_template"] = [
            paraphrased_templates[i].replace("[X]", "S").replace("[Y]", "O")
            for i in range(len(paraphrased_templates))
        ]
    if "orig_relation_template" in df.columns:
        orig_templates = df["orig_relation_template"].to_list()
        df["orig_relation_template"] = [
            orig_templates[i].replace("[X]", "S").replace("[Y]", "O")
            for i in range(len(orig_templates))
        ]
    if "relation" in df.columns:
        relations = df["relation"].to_list()
        df["relation"] = [
            relations[i].replace("[X]", "S").replace("[Y]", "O").replace(" .", ".")
            for i in range(len(relations))
        ]

    # dataset names
    if "dataset" in df.columns:
        datasets = df["dataset"].to_list()
        df["dataset"] = [
            i.replace("hypernymy", "Hypernymy").replace("trex", "TRex")
            for i in datasets
        ]

    # rename columns
    # metric names
    if "max_p" in df.columns:
        df.rename(columns={"max_p": r"\#p"}, inplace=True)

    if "incorrect -> correct" in df.columns:
        df.rename(
            columns={"incorrect -> correct": f"incorrect $\rightarrow$ correct"},
            inplace=True,
        )
    if f"Rel. incorrect -> correct" in df.columns:
        df.rename(
            columns={
                f"% incorrect -> correct": f"Rel. incorrect $\rightarrow$ correct"
            },
            inplace=True,
        )
    if "correct -> incorrect" in df.columns:
        df.rename(
            columns={"correct -> incorrect": f"correct $\rightarrow$ incorrect"},
            inplace=True,
        )

    if f"Rel. correct -> incorrect" in df.columns:
        df.rename(
            columns={"correct -> incorrect": f"Rel. correct $\rightarrow$ incorrect"},
            inplace=True,
        )
    if f"Rel. no change" in df.columns:
        df.rename(columns={"Rel. no change": "$\%$ no change"}, inplace=True)

    # coverage_rel_global
    # precision_selective_global
    # precision_argmax_selective_global

    # avoid rounding by using string formatting
    # threshold

    # by using int formatting
    if "\#o" in df.columns:
        df["\#o"] = df["\#o"].astype(np.int32)
    if "\#s" in df.columns:
        df["\#s"] = df["\#s"].astype(np.int32)
    if "total" in df.columns:
        df["total"] = df["total"].astype(np.int32)
    if "\#pos" in df.columns:
        df["\#pos"] = df["\#pos"].astype(np.int32)
        print(f"using int for \#pos")
    if "\#neg" in df.columns:
        df["\#neg"] = df["\#neg"].astype(np.int32)
        print(f"Using int for \#neg")
    if "#para" in df.columns:
        df["\#para"] = df["\#para"].astype(np.int32)
    if "#p" in df.columns:
        try:
            df["\#p"] = df["\#p"].astype(np.int32)
        except KeyError:
            pass

    if "$|O^-|$" in df.columns:
        df["$|O^-|$"] = df["$|O^-|$"].astype(np.int32)

    return df


def find_differences_2_runs(
    run_names,
    dataset_per_run,
    model_per_run,
    run_attributes,
    BASE_PATH,
    threshold,
    metric="precision_selective_global",
):
    # combine stats the two runs
    for r, run_name in enumerate(run_names):
        df_stats, df_instance_permutations = get_data(run_name, BASE_PATH)

        stats_s = subject_overview2(
            df_stats,
            df_instance_permutations,
        )

        stats_s["run_attributes"] = run_attributes[r]

        if r == 0:
            stats_s_all = stats_s
        else:
            stats_s_all = pd.concat([stats_s_all, stats_s], axis=0)

    # find differences between s, and s + context in terms of precision (category)
    # different results
    diff = []
    new_correct = []
    new_incorrect = []

    stats_s_all = stats_s_all.sort_values(by=["r_s_id"], ascending=True)

    for r_s_id in stats_s_all["r_s_id"].unique():
        df = stats_s_all[stats_s_all["r_s_id"] == r_s_id]
        assert (
            len(df) == 2
        ), f"Error: {len(df)} rows for r_s_id: {r_s_id}, should be 2. Run names {run_names}"

        run_1 = df[df["run_attributes"] == run_attributes[0]]
        run_2 = df[df["run_attributes"] == run_attributes[1]]

        categories = run_1["category"].to_list() + run_2["category"].to_list()
        attributes = (
            run_1["run_attributes"].to_list() + run_2["run_attributes"].to_list()
        )
        r_s = run_1["r_s_id"].to_list() + run_2["r_s_id"].to_list()

        assert (
            attributes[0] != attributes[1]
        ), f"Error: {attributes[0]} == {attributes[1]}"
        assert (
            (attributes[1] == "")
            or (attributes[1] == "0-shot")
            or (attributes[1] == "3-shot" and attributes[0] == "3-shot-neg")
        ), f"second run_arg is not standard: {attributes}, {categories}, {r_s} "  # other one is with context

        if (
            categories[0] != categories[1]
        ):  # correct with context, wrong without context or vice versa
            diff.extend([1, 1])

            # correct with context, wrong without context
            if (
                categories[0] == "Knowledge is Correct"
                and categories[1] == "Knowledge is Incorrect"
            ):
                new_correct.extend([True, True])

            # correct without context, incorrect with context
            elif (
                categories[0] == "Knowledge is Incorrect"
                and categories[1] == "Knowledge is Correct"
            ):
                new_correct.extend([False, False])

        elif categories[0] == categories[1]:
            diff.extend([0, 0])
            new_correct.extend([np.nan, np.nan])

    diff = np.array(diff)
    new_correct = np.array(new_correct)

    stats_s_all["difference"] = diff
    stats_s_all["new_correct"] = new_correct

    # calculate stats
    correct_new = new_correct[new_correct == True].shape[0] / 2
    incorrect_new = new_correct[new_correct == False].shape[0] / 2
    no_change = len(diff) / 2 - diff.sum() / 2
    total = len(diff) / 2

    assert (
        total == correct_new + incorrect_new + no_change
    ), f"Error: {total} != {correct_new} + {incorrect_new} + {no_change}"

    data = [
        run_names[0],
        dataset_per_run[0],
        model_per_run[0],
        run_attributes[0],
        correct_new,
        incorrect_new,
        no_change,
        total,
    ]

    stats_plot = stats_s_all[stats_s_all["run_attributes"] == run_attributes[0]]

    old_argmax_o = stats_s_all[stats_s_all["run_attributes"] == run_attributes[1]][
        "argmax_o"
    ].to_list()

    # get the old argmax_o prediction:
    stats_plot["old_argmax_o"] = old_argmax_o

    # rename difference from 1, 0 to different, same
    stats_plot["difference_str"] = [
        "Different" if x == 1 else "Same" for x in stats_plot["difference"]
    ]

    return stats_s_all, stats_plot, data
