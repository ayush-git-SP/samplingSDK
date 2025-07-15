"""module for the sampling subsection of Sampling """

import random
import traceback
from collections import OrderedDict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from .HelperScripts import get_summary_stat


def preference_maker(rs, count):
    """
    Return a list of user-selected colors up to a specified count.

    Args:
        rs (iterable): Iterable of colors.
        count (int): Number of colors to select.

    Returns:
        list: List of selected colors.
    """
    lst = [] * count
    for col in rs:
        lst.append(col)
        if len(lst) == count:
            break
    return lst


def plot_hist(df, column_name, name, color, col_type):
    """
    Plot a histogram for a given DataFrame column.

    Args:
        df (pd.DataFrame): Data to plot.
        column_name (str): Column name for x-axis.
        name (str): Plot title.
        color (str): Color of the histogram bars.
        col_type (str, optional): Column type, e.g., 'categorical' or 'continuous'. Defaults to 'continuous'.

    Returns:
        plotly.graph_objects.Figure: Histogram figure.
    """
    fig1 = go.Figure(
        layout=dict(
            xaxis=dict(title=column_name.capitalize()), yaxis=dict(title="Count")
        )
    )
    fig1.add_trace(go.Histogram(x=df[column_name], name=name, marker_color=color))

    fig1.update_layout(
        {
            "plot_bgcolor": "rgba(0,0,0,0)",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "title": {"text": name, "x": 0.5, "xanchor": "center"},
        }
    )

    if col_type == "categorical" or col_type == "catcont":
        fig1.update_xaxes(
            title_text=column_name.capitalize(),
            showline=True,
            linewidth=1,
            linecolor="black",
            type="category",
        )
    else:
        fig1.update_xaxes(
            title_text=column_name.capitalize(),
            showline=True,
            linewidth=1,
            linecolor="black",
        )
    fig1.update_yaxes(title_text="Count", showline=True, linewidth=1, linecolor="black")
    fig1.layout.legend = dict(
        x=0.4,
        y=-0.2,
        traceorder="normal",
        orientation="h",
        font=dict(
            size=12,
        ),
    )
    return fig1


def sampling_compare(df, test_df, train_df, column_name, color=None, col_type=None):
    """
    Compare population, train, and test samples with summary stats and histograms.

    Args:
        df (pd.DataFrame): Original dataset.
        test_df (pd.DataFrame): Test sample.
        train_df (pd.DataFrame): Train sample.
        column_name (str): Column to analyze.
        color (list, optional): List of colors for plots (default provided).
        col_type (str, optional): Column type for plotting.

    Returns:
        tuple: (list of plotly figures, summary DataFrame)
    """
    if not color or len(color) < 3:
        color = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    dframe1 = get_summary_stat(df, column_name, col_type)
    test_summ_stat = get_summary_stat(test_df, column_name, col_type)
    train_summ_stat = get_summary_stat(train_df, column_name, col_type)
    lis = []
    for key, pop_val in dframe1.items():
        lis.append(
            OrderedDict(
                {
                    "Attribute": key,
                    "Population": pop_val,
                    "TrainSample": train_summ_stat[key],
                    "TestSample": test_summ_stat[key],
                }
            )
        )

    listdf = pd.DataFrame(lis, None)
    fig1 = plot_hist(df, column_name, "Before Sampling", color[0], col_type)
    fig2 = plot_hist(train_df, column_name, "After Sampling Train", color[1], col_type)
    fig3 = plot_hist(test_df, column_name, "After Sampling Test", color[2], col_type)

    return [fig1, fig2, fig3], listdf


def make_rand(df, column_name, ratio, lst, action, col_type):
    """
    Perform random sampling and return figures, train/test splits, and summary.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Column to sample on.
        ratio (float): Train set ratio.
        lst (list): List of colors for plotting.
        action (str): Action type, e.g., 'submit'.
        col_type (str): Column type for plotting.

    Returns:
        tuple: (figures, train DataFrame, test DataFrame, summary table)
    """
    if lst is None:
        lst = ["#1f77b4"]

    train, test = train_test_split(df, train_size=ratio)
    fig = []
    table1 = []
    if action == "submit":
        fig, table1 = sampling_compare(df, test, train, column_name, lst, col_type)
    return fig, train, test, table1


def makestrat(df, column_name, ratio, lst, action, col_type):
    """
    Perform stratified sampling and return figures, train/test splits, and summary.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Column to stratify on.
        ratio (float): Train set ratio.
        lst (list): List of colors for plotting.
        action (str): Action type, e.g., 'submit'.
        col_type (str): Column type for plotting.

    Returns:
        tuple: (figures, train DataFrame, test DataFrame, summary table)
    """
    cat_var = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ["object"]]
    if column_name in cat_var:
        y = df[column_name]
    else:
        try:
            est = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
            y = est.fit_transform(df[[column_name]]).flatten().astype(int)
        except Exception:
            traceback.print_exc()
            y = df[column_name]
    try:
        train, test = train_test_split(df, train_size=ratio, stratify=y)
    except Exception:
        traceback.print_exc()
        train, test = train_test_split(df, train_size=ratio)
    fig = []
    table1 = []
    if action == "submit":
        fig, table1 = sampling_compare(df, test, train, column_name, lst, col_type)
    return fig, train, test, table1


def cluster_sampling(df, column_name, cluster, lst, action, col_type):
    """
    Perform cluster sampling on the DataFrame and return sample, figure, and table.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Target column name.
        cluster (int): Number of clusters.
        lst (list): List of colors for plotting.
        action (str): Action type, e.g., 'submit' or 'save'.
        col_type (str): Column type for plotting.

    Returns:
        tuple: (figure, sampled DataFrame, summary table).
    """
    l = [x % cluster for x in range(0, len(df))]
    x = np.array(l)
    df["cluster_id"] = x
    indexes = []
    num = random.randint(0, cluster - 1)
    for i in range(0, len(df)):
        if df["cluster_id"].iloc[i] == num:
            indexes.append(i)
    cluster_sample = df.iloc[indexes]
    cluster_sample.head()
    y_train = cluster_sample[column_name]
    y = df[column_name]
    y_test = y[~y.index.isin(cluster_sample.index)]
    fig = []
    table1 = []
    if action == "submit":
        fig, table1 = sampling_compare(
            y.to_frame(),
            y_test.to_frame(),
            y_train.to_frame(),
            column_name,
            lst,
            col_type,
        )
    return fig, cluster_sample, table1


def systematic_sampling(df, column_name, ratio, lst, action, col_type):
    """
    Perform systematic sampling and return sample, figure, and table.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Target column name.
        ratio (float): Sampling ratio.
        lst (list): List of colors for plotting.
        action (str): Action type, e.g., 'submit' or 'save'.
        col_type (str): Column type for plotting.

    Returns:
        tuple: (figure, sampled DataFrame, summary table).
    """
    main_size = df[column_name].size
    smpl_size = main_size * ratio
    n = main_size / smpl_size
    indexes = np.arange(0, len(df), step=n)
    systematic_sample = df.iloc[indexes]
    y_train = systematic_sample[column_name]
    y = df[column_name]
    y_test = y[~y.index.isin(systematic_sample.index)]
    fig = []
    table1 = []
    if action == "submit":
        fig, table1 = sampling_compare(
            y.to_frame(),
            y_test.to_frame(),
            y_train.to_frame(),
            column_name,
            lst,
            col_type,
        )

    return fig, systematic_sample, table1


def sample(df, column_name, ratio, method, lst, action, col_type):
    """
    Split data by specified sampling method and return figures, tables, and samples.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Target column name.
        ratio (float or int): Sampling ratio or cluster count.
        method (int): Sampling method choice (1: random, 2: stratified, 3: cluster, 4: systematic).
        lst (list): List of colors for plotting.
        action (str): Action type, e.g., 'submit' or 'save'.
        col_type (str): Column type for plotting.

    Returns:
        tuple: Figures, sampled data, and summary table depending on method.
    """

    cat_var = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ["object"]]
    if column_name in cat_var:
        df[column_name].fillna("NOT_GIVEN", inplace=True)
    else:
        df[column_name].fillna(-1, inplace=True)

    choice = method
    if choice == 1:
        fig1, train, test, tab1 = make_rand(
            df, column_name, ratio, lst, action, col_type
        )
        return fig1, train, test, tab1
    elif choice == 2:
        fig2, train, test, tab2 = makestrat(
            df, column_name, ratio, lst, action, col_type
        )
        return fig2, train, test, tab2
    elif choice == 3:
        fig3, cs, tab3 = cluster_sampling(df, column_name, ratio, lst, action, col_type)
        return fig3, cs, tab3
    elif choice == 4:
        fig3, ss, tab3 = systematic_sampling(
            df, column_name, ratio, lst, action, col_type
        )
        return fig3, ss, tab3
