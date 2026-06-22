from __future__ import annotations


def get_system_prompt(dataset: str) -> str:
    """Return the system prompt used for analogy completion.

    This is shared between API evaluation and local prompting baselines.

    Args:
        dataset: Either "base" or "meta".

    Returns:
        The system prompt string.
    """
    if dataset == "meta":
        return (
            "You are tasked with solving word analogies. You are given:\n"
            "1. A set of example pairs of the form 'A is to B', preceded by the token <STUDY>. All example pairs share the same hidden relation.\n"
            "2. A query of the form 'C is to', preceded by the token <QUERY>.\n\n"
            "Infer the relation that holds between the pairs in the study examples, then apply that same relation to the query to determine the single term that completes the analogy.\n\n"
            "Provide your answer in exactly this format:\n"
            "### Answer: term"
        )

    if dataset == "base":
        return (
            "You are tasked with solving word analogies. You are given a query of the form 'C is to', preceded by the token <QUERY>.\n\n"
            "Determine the single term that completes the analogy.\n\n"
            "Provide your answer in exactly this format:\n"
            "### Answer: term"
        )

    raise ValueError("dataset must be 'base' or 'meta'")
