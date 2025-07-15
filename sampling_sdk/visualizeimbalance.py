"""
Module for Visualization imbalance method of sampling.
"""

import math
from collections import Counter
from numpy import log
import plotly.graph_objects as go


def shanon(seq):
    """
    Calculate normalized Shannon entropy of a sequence.

    Args:
        seq (iterable): Input sequence.

    Returns:
        float or str: Normalized entropy (0–1), or 'Not Applicable' if undefined.
    """

    n = len(seq)
    classes = [(clas, float(count)) for clas, count in Counter(seq).items()]
    k = len(classes)

    h = -sum([(count / n) * log((count / n)) for clas, count in classes])
    result = h / log(k)
    if math.isnan(result):
        result = "Not Applicable"
    return round(result, 3)


def plot_fun(column_name, df, col_type):
    """
    Generate a plot for a given column based on its type.

    Args:
        column_name (str): Column to plot.
        df (DataFrame): Input DataFrame.
        col_type (str): Column type ('categorical', 'continuous', etc.).

    Returns:
        Figure: Plotly figure object.
    """

    color = ["#1f77b4"]

    if col_type in ("categorical", "catcont"):
        trace = go.Bar(
            x=df[column_name].value_counts().keys().tolist(),
            y=df[column_name].value_counts().tolist(),
            marker_color=color[0],
        )
    else:
        trace = go.Histogram(x=df[column_name], name=column_name, marker_color=color[0])

    data = [trace]
    layout = go.Layout(
        barmode="overlay",
        xaxis=dict(title=column_name),
        yaxis=dict(title="Count"),
        bargap=0.25,
    )

    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        {"plot_bgcolor": "rgba(0,0,0,0)", "paper_bgcolor": "rgba(0,0,0,0)"}
    )

    if col_type in ("categorical", "catcont"):
        fig.update_xaxes(
            title_text=column_name,
            showline=True,
            linewidth=1,
            linecolor="black",
            type="category",
        )
    else:
        fig.update_xaxes(
            title_text=column_name, showline=True, linewidth=1, linecolor="black"
        )

    fig.update_yaxes(title_text="Count", showline=True, linewidth=1, linecolor="black")

    return fig


def structure_table(shannon_entropy, column_name):
    """
    Create a summary dictionary with entropy and interpretation.

    Args:
        shannon_entropy (float): Calculated Shannon entropy.
        column_name (str): Name of the column.

    Returns:
        dict: Dictionary with variable, entropy, and interpretation.
    """

    interpretation = ""
    if shannon_entropy > 0.9:
        interpretation = "Highly Balanced"
    elif shannon_entropy < 0.8:
        interpretation = "Imbalanced"
    else:
        interpretation = "Balanced"
    result = {}
    result["Variable"] = column_name
    result["Shannon Entropy"] = shannon_entropy
    result["Interpretation"] = interpretation
    return result
