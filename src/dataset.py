from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
from datasets import Dataset


STUDY_TOKEN = "<STUDY>"
QUERY_TOKEN = "<QUERY>"
ANSWER_TOKEN = "<ANSWER>"
STOP_TOKEN = "<STOP>"

COLUMNS = ["input", "output", "query", "gold_answer", "target", "relation", "episode_id"]


def get_dataset(
    dataset_type: str,
    train_path: str = "data/train_episodes_7b.jsonl",
    test_path: str = "data/test_episodes_7b.jsonl",
    dev_per_relation: int = 0,
    subsample_train: Optional[int] = None,
    seed: int = 42,
    print_info: bool = True,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Build the train, dev, and test splits for the analogy task.

    Args:
        dataset_type: Training condition, ``"base"`` or ``"meta"``. This selects
            whether the ``<STUDY>`` support set is included in the model input.
        train_path: Path to the training-episode ``.jsonl`` file.
        test_path: Path to the test-episode ``.jsonl`` file (disjoint relations).
        dev_per_relation: Number of episodes per relation to hold out of the
            training data to form an in-distribution validation set.
        subsample_train: If set, cap the number of training episodes per relation
            (low-data regime). If ``None``, all training data is used. Applied
            after the dev set is carved out.
        seed: Seed for the deterministic dev carve / subsampling / shuffle.
        print_info: If ``True``, print split statistics.

    Returns:
        ``(train, dev, test)`` as HuggingFace ``Dataset`` objects. ``dev`` shares
        relations with ``train`` (in-distribution); ``test`` uses held-out
        relations (out-of-distribution generalisation).
    """
    if dataset_type not in ("base", "meta"):
        raise ValueError("dataset_type must be 'base' or 'meta'")

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    train_df, dev_df = _carve_dev(train_df, dev_per_relation, seed)
    if subsample_train:
        train_df = _subsample_per_relation(train_df, subsample_train, seed)

    # Format inputs/outputs for the chosen condition.
    for df in (train_df, dev_df, test_df):
        _format_input_output(df, dataset_type)

    _check_relation_disjointness(train_df, test_df)
    if print_info:
        _print_dataset_info(train_df, dev_df, test_df)

    train = Dataset.from_pandas(train_df[COLUMNS], preserve_index=False).shuffle(seed=seed)
    dev = Dataset.from_pandas(dev_df[COLUMNS], preserve_index=False)
    test = Dataset.from_pandas(test_df[COLUMNS], preserve_index=False)
    return train, dev, test


def _format_input_output(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """Add ``input``, ``output``, ``gold_answer`` and ``episode_id`` columns in place.

    The training label (``output``) is the canonical answer, i.e. the first
    entry of ``target``. Loss is computed only on the span after ``<ANSWER>``.
    """
    # Canonical answer used as the supervised label.
    df["gold_answer"] = df["target"].apply(lambda t: t[0] if isinstance(t, list) else t)

    if dataset_type == "meta":
        df["input"] = (
            f"{STUDY_TOKEN} " + df["study"].astype(str)
            + f" ; {QUERY_TOKEN} " + df["query"].astype(str)
        )
    else:  # base: no support set
        df["input"] = f"{QUERY_TOKEN} " + df["query"].astype(str)

    # Leading space before ANSWER_TOKEN guarantees the loss-mask boundary is
    # found in both conditions (in base, the query text precedes it).
    df["output"] = f" {ANSWER_TOKEN} " + df["gold_answer"].astype(str) + f" {STOP_TOKEN}"

    df["episode_id"] = [f"{r}_{i}" for i, r in enumerate(df["relation"])]
    return df


def _carve_dev(
    train_df: pd.DataFrame, dev_per_relation: int, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out ``dev_per_relation`` episodes per relation as validation data."""
    if dev_per_relation <= 0:
        return train_df.reset_index(drop=True), train_df.iloc[0:0].copy()

    dev_parts: List[pd.DataFrame] = []
    for _, group in train_df.groupby("relation"):
        k = min(dev_per_relation, len(group))
        dev_parts.append(group.sample(n=k, random_state=seed))

    dev_df = pd.concat(dev_parts).reset_index(drop=True)
    train_df = train_df.drop(index=pd.concat(dev_parts).index).reset_index(drop=True)
    return train_df, dev_df


def _subsample_per_relation(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Cap the number of episodes per relation at ``k`` (low-data regime)."""
    parts = [
        group.sample(n=min(k, len(group)), random_state=seed)
        for _, group in df.groupby("relation")
    ]
    return pd.concat(parts).reset_index(drop=True)


def _check_relation_disjointness(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Warn if any relation appears in both train and test (leakage)."""
    overlap = set(train_df["relation"]) & set(test_df["relation"])
    if overlap:
        import warnings

        warnings.warn(
            f"Train and test share relations (leakage): {sorted(overlap)}. "
            "Generalisation is measured across disjoint relations.",
            stacklevel=2,
        )


def _print_dataset_info(
    train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    """Print per-split size and relation statistics."""
    print("-" * 50)
    for name, df in [("Train", train_df), ("Dev", dev_df), ("Test", test_df)]:
        rels = sorted(df["relation"].unique())
        print(f"{name} size: {len(df)} | relations ({len(rels)}): {rels}")
    train_rels = set(train_df["relation"])
    test_rels = set(test_df["relation"])
    print(f"Held-out (test-only) relations: {sorted(test_rels - train_rels)}")
    print("-" * 50)
