import pytest

from main import (
    rougeL_score,
    bert_score,
    charf_score,
    exact_match,
    calculate_similarity_metrics,
)
import numpy as np


def test_rouge_score(score_data, metrics):

    continuations, labels = score_data
    rouge_l_metric, bert_score_metric, charf_metric = metrics

    rougeL, rouge1, rouge2, rougeLsum = rougeL_score(
        rouge_l_metric, continuations, labels
    )

    assert len(rougeL) == len(labels)

    assert np.allclose(rouge1, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], atol=1e-1)
    assert np.allclose(rouge2, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-1)
    assert np.allclose(rougeL, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], atol=1e-1)
    assert np.allclose(rougeLsum, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], atol=1e-1)


def test_bert_score(score_data, metrics):
    continuations, labels = score_data
    rouge_l_metric, bert_score_metric, charf_metric = metrics

    precision, recall, f1 = bert_score(bert_score_metric, continuations, labels)

    assert len(f1) == len(labels)

    assert np.allclose(
        f1,
        [
            0.993033766746521,
            0.9943046569824219,
            0.9900882840156555,
            0.9999998807907104,
            0.9941731095314026,
            0.9949090480804443,
            0.9955456852912903,
            0.9999999403953552,
        ],
        atol=1e-6,
    )


def test_charf_score(score_data, metrics):
    continuations, labels = score_data
    rouge_l_metric, bert_score_metric, charf_metric = metrics

    score = charf_score(charf_metric, continuations, labels)

    assert len(score) == len(labels)

    assert np.allclose(
        score,
        [
            0.0,
            22.22222222222222,
            38.888888888888886,
            100.0,
            7.5396825396825395,
            0.0,
            48.56912535268037,
            100.0,
        ],
        atol=1e-6,
    )


def test_em_score(score_data):
    continuations, labels = score_data
    score = exact_match(continuations, labels)

    assert len(score) == len(labels)
    assert np.allclose(
        score,
        [
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        atol=1e-6,
    )


def test_all_metrics(data):
    # test all metrics together with the pipeline
    contexts, continuations, labels = data

    scores, detailed_scores = calculate_similarity_metrics(
        continuations,
        labels,
        contexts,
        device="mps",
        num_generations=len(continuations[0]),
    )

    assert scores["bert_score_f1"].shape == (2,)  # averages: contexts x 1
    assert np.allclose(scores["bert_score_f1"], [0.9940296, 0.99508778], atol=1e-3)
    assert np.allclose(scores["charf"], [7.40740741, 5.11739418], atol=1e-3)

    assert detailed_scores["bert_score_f1"].shape == (2, 3)  # contexts x continuations
    assert np.allclose(
        detailed_scores["bert_score_f1"],
        [[0.99303377, 0.99430466, 0.99475038], [0.99417311, 0.99490905, 0.99618119]],
        atol=1e-3,
    )
    assert np.allclose(
        detailed_scores["charf"],
        [[0.0, 22.22222222, 0.0], [7.53968254, 0.0, 7.8125]],
        atol=1e-3,
    )
