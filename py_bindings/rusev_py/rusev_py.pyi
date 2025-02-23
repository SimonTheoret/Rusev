def classification_report(
    y_true: list[str],
    y_pred: list[str],
    zero_division: str,
    suffix: bool,
    scheme: str | None = None,
    sample_weight: list[float] | None = None,
) -> dict[str, dict[str, float]]: ...
