import pytest
from exp_3_set_proba.utils import create_df_probability_vis
from main import (
    rougeL_score,
    bert_score,
    charf_score,
    exact_match,
    calculate_similarity_metrics,
)
import numpy as np
from scipy.special import softmax

# SETTINGS
r_s_id = 1


def test_probability_dists(df_instance_permutations):
    df_vis, _ = create_df_probability_vis(
        df_instance_permutations,
        r_s_id=r_s_id,
        PLOT_PATH=False,
        template_column="p_t_s_r",
    )

    # Check probabilities are normalized over paraphrased_relation_template and r_s_id (sum over paraphrases)
    for column in df_vis["paraphrased_relation_template"].unique():
        sum_column = df_vis[df_vis["paraphrased_relation_template"] == column][
            "p"
        ].sum()

        assert np.allclose(
            sum_column, 1.0, atol=1e-12
        ), f"T(r) = {column}: Sum is not 1: {sum_column}"


def test_summing_over_paraphrases(df_instance_permutations):
    # check the summing of one object over all paraphrased_relation_template is 1
    df_test = df_instance_permutations[df_instance_permutations["r_s_id"] == r_s_id]

    test_objects = df_test["obj_label"].unique()[:5]

    for test_object in test_objects:
        p_t_over_paraphrases = df_test[
            (df_test["obj_label"] == test_object) & (df_test["paraphrase_id"] == 0)
        ]["p_t_over_paraphrases"]

        p_per_paraphrase_template_normalized = df_test[
            df_test["obj_label"] == test_object
        ]["p_t"]

        summed = np.sum(p_per_paraphrase_template_normalized)

        assert np.allclose(
            summed, p_t_over_paraphrases, atol=1e-12
        ), f"Sum is not equal for test obj: {test_object}: {summed}, {p_t_over_paraphrases}"
