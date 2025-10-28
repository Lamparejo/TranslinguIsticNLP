"""Tests for the REBEL relation extraction helpers."""

import pytest

from src.nlp.rebel_pipeline import parse_rebel_output, split_sentences


@pytest.mark.parametrize(
    "generated, expected",
    [
        (
            "<triplet> <subj> Apple Inc. </subj> <rel> founded by </rel> <obj> Steve Jobs </obj>",
            [("Apple Inc.", "founded by", "Steve Jobs")],
        ),
        (
            (
                "<triplet> <subj> Tim Cook </subj> <rel> is CEO of </rel> <obj> Apple Inc. </obj> "
                "<triplet> <subj> Apple Inc. </subj> <rel> headquartered in </rel> <obj> Cupertino </obj>"
            ),
            [
                ("Tim Cook", "is CEO of", "Apple Inc."),
                ("Apple Inc.", "headquartered in", "Cupertino"),
            ],
        ),
        ("", []),
    ],
)
def test_parse_rebel_output(generated, expected):
    assert parse_rebel_output(generated) == expected


def test_split_sentences_uses_spacy_sentencizer():
    text = "Apple lançou um novo produto. O evento ocorreu em Cupertino."
    sentences = list(split_sentences(text))
    assert sentences == [
        "Apple lançou um novo produto.",
        "O evento ocorreu em Cupertino.",
    ]
