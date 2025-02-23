# TODO: Create stubs for rusev (lib.rs)
# TODO: Verify return type
# TODO: Reread documentation

from py_bindings import (
    classification_report as classification_report_inner
)


def classification_report(
    y_true: list[list[str]],
    y_pred: list[list[str]],
    digits=2,
    suffix=False,
    output_dict=False,
    mode=None,
    sample_weight: list[float] | None = None,
    zero_division: str = "ReplaceBy1",
    scheme: str | None = None,
) -> dict[str, dict[str, float]]:  # type: ignore
    """Entry point of the Rusev library. Build a text report showing the main
    classification metrics.

    Args:
        y_true : 2d array. Ground truth (correct) target values.

        y_pred : 2d array. Estimated targets as returned by a classifier.

        digits : int. Number of digits for formatting output floating point values.

        output_dict : bool(default=False). If True, return output as dict else str.

        mode : str, [None (default), `strict`].
            This parameter is left as is to not break compatibility, but does
            nothing. The mode is automatically selected based on the the value
            of the `scheme` argument. If `scheme` is `None`, the function
            assumes we are in `default` mode. If scheme is one the provided
            scheme, such as `IOB1`, `IOE1`, `IOB2`, `IOE2`, `BILOU`, `IOBES`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. M
        zero_division : "ReplaceBy1", "ReplaceBy0", or "ReturnError". Defaults to "ReplaceBy1".
            Sets the value to return when there is a zero division:
               - recall: when there are no positive labels
               - precision: when there are no positive predictions
               - f-score: both

            If set to "ReplaceBy0", this acts as 0. When passing the arguments,
            capitalization is ignored.

        scheme : Token, [IOB2, IOE2, IOBES]

        suffix : bool, False by default.

    Returns:
        report : dict. Summary of the precision, recall, F1 score for each class.

    Examples:
        >>> from seqeval.metrics import classification_report
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> print(classification_report(y_true, y_pred))
                     precision    recall  f1-score   support
        <BLANKLINE>
               MISC       0.00      0.00      0.00         1
                PER       1.00      1.00      1.00         1
        <BLANKLINE>
          micro avg       0.50      0.50      0.50         2
          macro avg       0.50      0.50      0.50         2
       weighted avg       0.50      0.50      0.50         2
        <BLANKLINE>
    """
    classification_report_inner(
        y_true,
        y_pred,
        digits,
        suffix,
        output_dict,
        mode,
        sample_weight,
        zero_division,
        scheme,
    )
