import pytest
import evaluate
from sacrebleu import CHRF
from main import parse_args
import pandas as pd
import os
from pathlib import Path

HOME_PATH = os.path.expanduser("~/")
OUT_PATH = Path(f"{HOME_PATH}/Py/MAI_Codebase/exp_3_set_proba/")  # TODO change this
RUN_NAME = "small_test_2_3_new"
# RUN_NAME = "PopQA_test_2000_50_mistral7B" # TODO remove


@pytest.fixture(scope="session")
def score_data():
    continuations_flat = [
        "Time",
        "ABC",
        "CBD",
        "CBS",
        "Matthew",
        "Drew",
        "Mann",
        "Manning",
    ]
    labels_flat = [
        ["CBS"],
        ["CBS"],
        ["CBS"],
        ["CBS"],
        ["Manning"],
        ["Manning"],
        ["Manning"],
        ["Manning"],
    ]

    return continuations_flat, labels_flat


@pytest.fixture(scope="session")
def metrics():
    rouge_l_metric = evaluate.load("rouge")
    bert_score_metric = evaluate.load("bertscore", device="mps")
    charf_metric = CHRF()  # native implementation

    return rouge_l_metric, bert_score_metric, charf_metric


@pytest.fixture(scope="session")
def data():
    contexts = [
        "The TGIF comedy Family Matters for the 1997-98 season was originally aired by",
        "The first choice in the NFL draft of 1998 was",
    ]
    continuations = [["Time", "ABC", "Fox"], ["Matthew", "Drew", "Mike"]]
    labels = ["CBS", "Manning"]

    return contexts, continuations, labels


@pytest.fixture(scope="session")
def df_instance_permutations():
    df_instance_permutations = pd.read_parquet(
        OUT_PATH / RUN_NAME / "permutations_scores_p_t_all.parquet"
    )
    return df_instance_permutations
