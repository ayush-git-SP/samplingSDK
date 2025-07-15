"""module for the sampling subsection of Sampling """

import warnings
import random
from collections import OrderedDict
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from .HelperScripts import get_summary_stat
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

warnings.simplefilter("ignore")


def preference_maker(rs, count):
    """This function return list of colors selected by the user."""
    lst = [] * count
    for col in rs:
        lst.append(col)
        if len(lst) == count:
            break
    return lst


def plot_hist(df, column_name, name, color, col_type):
    """This function plots histogram.

    Args:
        df : Dataframe.
        column_name (str): Name of the columns to be used for plotting.
        name (str): Name of histogram plot
        color (str): Color choice foor plot.
        column_type (str, optional): Type of column. Defaults to "continous".

    Returns:
        Returns histogram plot figure.
    """
    fig1 = go.Figure(
        layout=dict(
            xaxis=dict(title=column_name.capitalize()), yaxis=dict(title="Count")
        )
    )
    fig1.add_trace(go.Histogram(x=df[column_name], name=name, marker_color=color))

    # fig1.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
    #                    'paper_bgcolor': 'rgba(0,0,0,0)'})
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
    # Default color list if not provided or too short
    if not color or len(color) < 3:
        color = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    dframe1 = get_summary_stat(df, column_name, col_type)
    test_summ_stat = get_summary_stat(test_df, column_name, col_type)
    train_summ_stat = get_summary_stat(train_df, column_name, col_type)
    lis = []
    for key in dframe1:
        lis.append(
            OrderedDict(
                {
                    "Attribute": key,
                    "Population": dframe1[key],
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
    """This function returns figure and table for random method in sampling"""

    if lst is None:
        lst = ["#1f77b4"]

    train, test = train_test_split(df, train_size=ratio)
    fig = []
    table1 = []
    if action == "submit":
        fig, table1 = sampling_compare(df, test, train, column_name, lst, col_type)
    return fig, train, test, table1


def makestrat(df, column_name, ratio, lst, action, col_type):
    """This function return figure and table for stratified method in sampling"""
    print("Column name:", column_name)
    print("inside makestrat")
    cat_var = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ["object"]]
    if column_name in cat_var:
        y = df[column_name]
    else:
        try:
            est = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
            y = est.fit_transform(df[[column_name]]).flatten().astype(int)
        except Exception:
            y = df[column_name]
    try:
        train, test = train_test_split(df, train_size=ratio, stratify=y)
    except:
        train, test = train_test_split(df, train_size=ratio)
    fig = []
    table1 = []
    if action == "submit":
        fig, table1 = sampling_compare(df, test, train, column_name, lst, col_type)
    return fig, train, test, table1


def cluster_sampling(df, column_name, cluster, lst, action, col_type):
    """This function makes cluster of data and returns figure table for the same."""
    print("Column name:", column_name)
    print("inside cluster_sampling")
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
    This function is used for systematic sampling method
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
    """This function returns figure, table and train and test data according to the method choice provided by user.

    Args:
        df : Dataframe.
        column_name (str): Name of columns.
        ratio (str): Ratio in which train and test data has to be divided.
        method (str): Name of method.
        lst (list): List of colours.
        action (str): Type of action,

    Returns:
        Returns figure and table.
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
