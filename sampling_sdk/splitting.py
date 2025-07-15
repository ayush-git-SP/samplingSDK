"""Module for Splitting method of sampling."""

from collections import OrderedDict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .global_var import PLOT_BGCOLOR
from .HelperScripts import get_color, get_summary_stat


def plot_hist(df, column_name, name, color, col_type):
    """
    Create a histogram plot for a specified column.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Column to plot.
        name (str): Plot title or trace name.
        color (str): Color for the bars.
        col_type (str): Column type, affects x-axis formatting.

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
        {"plot_bgcolor": "rgba(0,0,0,0)", "paper_bgcolor": "rgba(0,0,0,0)"}
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


def encoded_data(data, categorical_feature):
    """
    Encode categorical columns using LabelEncoder.

    Args:
        data (pd.DataFrame): Input DataFrame.
        categorical_feature (list): List of categorical column names.

    Returns:
        Tuple:
            data_encoded (pd.DataFrame): DataFrame with encoded categorical features.
            encoders (dict): Mapping of column names to LabelEncoder objects.
            categorical_feature (list): List of encoded categorical columns.
    """
    data_encoded = data.copy()
    categorical_names = {}
    encoders = {}

    for feature in categorical_feature:
        le = LabelEncoder()
        le.fit(data_encoded[feature].astype(str))

        data_encoded[feature] = le.transform(data_encoded[feature].astype(str))

        categorical_names[feature] = le.classes_
        encoders[feature] = le

    return data_encoded, encoders, categorical_feature


def decode_dataset(data, encoders, categorical_features):
    """
    Decode previously encoded categorical columns back to original values.

    Args:
        data (pd.DataFrame): DataFrame with encoded categorical columns.
        encoders (dict): Mapping of column names to LabelEncoder objects.
        categorical_features (list): List of categorical columns to decode.

    Returns:
        pd.DataFrame: DataFrame with decoded categorical columns.
    """
    df = data.copy()
    for feat in categorical_features:
        df[feat] = encoders[feat].inverse_transform(df[feat].astype(int))
    return df


def sp(df, column_name, ratio1, ratio2, method, action, col_type):
    """
    Split data using specified method.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Column to split on.
        ratio1 (float): First split ratio.
        ratio2 (float): Second split ratio.
        method (int): 1 for random split, 2 for stratified split.
        action (str): Action mode ('submit' or 'save').
        col_type (str): Column type for plotting.

    Returns:
        Tuple: Outputs from chosen split function.
    """
    if method == 1:
        return split_data(df, column_name, ratio1, ratio2, action, col_type)
    elif method == 2:
        return strat_split(df, column_name, ratio1, ratio2, action, col_type)


def stratified_split(df, column_name, ratio):
    """
    Split data into train and test using stratification on binned column values.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Column to stratify.
        ratio (float): Test set size proportion.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test splits.
    """
    bins = np.linspace(0, len(df), 5)
    y_binned = np.digitize(df[column_name], bins)
    x_train, x_test = train_test_split(
        df, df[column_name], test_size=ratio, stratify=y_binned, random_state=42
    )
    return x_train, x_test


def stratified_split1(df, column_name, ratio):
    """
    Split DataFrame into train and test sets using stratified sampling.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Column to stratify by.
        ratio (float): Test set proportion.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test DataFrames.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        df, df[column_name], test_size=ratio, stratify=df[column_name], random_state=42
    )
    return x_train, x_test


def startified_splitting(df, column_name, ratio):
    """
    Split data with stratification; handle categorical and numerical columns differently.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Column to stratify by.
        ratio (float): Test set proportion.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test DataFrames.
    """
    cat_var = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ["object"]]
    if column_name in cat_var:
        df[column_name] = df[column_name].fillna("NOT_GIVEN")
        x_train, x_test = stratified_split1(df, column_name, ratio)
        return x_train, x_test
    else:
        df[column_name] = df[column_name].fillna(-1)
        x_train, x_test = stratified_split(df, column_name, ratio)
        return x_train, x_test


def plot_pie(df, color):
    """
    Create a pie chart from value counts of a DataFrame column.

    Args:
        df (pd.Series): Data for pie chart (categorical).
        color (list): List of three colors for pie slices.

    Returns:
        plotly.graph_objects.Figure: Pie chart figure.
    """

    colors = [color[0], color[1], color[2]]
    val_cnt = df.value_counts()
    values = val_cnt.tolist()
    labels = val_cnt.keys().tolist()
    fig1 = go.Figure(
        data=[go.Pie(labels=labels, values=values, hole=0, pull=[0.007, 0.007, 0.007])]
    )
    fig1.update_traces(marker=dict(colors=colors))
    fig1.update_layout({"plot_bgcolor": PLOT_BGCOLOR, "paper_bgcolor": PLOT_BGCOLOR})
    fig1.update_layout(
        font_family="Poppins, sans-serif",
        title_x=0.5,
        legend_title_text="Legend",
        legend={"font": {"size": 12}, "x": 0.8, "y": 0.9, "traceorder": "normal"},
        title_font_color="black",
    )
    return fig1


def plot_pie2(value1, value2, value3, color):
    """
    Generate a pie chart with three slices labeled Mean, Median, and Mode.

    Args:
        value1 (float): Value for the Mean slice.
        value2 (float): Value for the Median slice.
        value3 (float): Value for the Mode slice.
        color (list): List of three colors for the slices.

    Returns:
        plotly.graph_objects.Figure: Pie chart figure.
    """

    colors = [color[0], color[1], color[2]]
    labels = ["Mean", "Median", "Mode"]
    values = [value1, value2, value3]

    fig = go.Figure(
        data=[go.Pie(labels=labels, values=values, hole=0, pull=[0.007, 0.007, 0.007])]
    )
    fig.update_traces(marker=dict(colors=colors))
    fig.update_layout({"plot_bgcolor": PLOT_BGCOLOR, "paper_bgcolor": PLOT_BGCOLOR})
    fig.update_layout(
        font_family="Poppins, sans-serif",
        title_x=0.5,
        legend_title_text="Legend",
        legend={"font": {"size": 12}, "x": 0.8, "y": 0.9, "traceorder": "normal"},
        title_font_color="black",
    )
    return fig


def split_data(df, column_name, ratio1, ratio2, action, col_type):
    """
    Split data into train, test, and validation sets (non-stratified) and return plots and stats.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Target column for splitting.
        ratio1 (float): Train split ratio.
        ratio2 (float): Test split ratio from remaining data.
        action (str): 'submit' to display output, 'save' to return empty plots and stats.
        col_type (str): Column type for plotting ('categorical' or 'numerical').

    Returns:
        Tuple: (fig1, fig2, fig3, fig4, summary_df, train_df, test_df, validate_df)
    """

    categorical_feature = [
        key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ["object"]
    ]
    data_encoded, encoders, categorical_features = encoded_data(df, categorical_feature)

    x = data_encoded
    y = data_encoded[column_name]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=ratio1, random_state=42
    )

    x_train2, x_test2, y_train2, y_test2 = train_test_split(
        x_test, y_test, train_size=ratio2, random_state=42
    )

    train_df = decode_dataset(x_train, encoders, categorical_features)
    test_df = decode_dataset(x_train2, encoders, categorical_features)
    cv_df = decode_dataset(x_test2, encoders, categorical_features)

    if action == "save":
        return (
            [],
            [],
            [],
            [],
            [],
            pd.DataFrame(train_df),
            pd.DataFrame(test_df),
            pd.DataFrame(cv_df),
        )

    dframe1 = get_summary_stat(df, column_name, col_type)
    test_summ_stat = get_summary_stat(test_df, column_name, col_type)
    train_summ_stat = get_summary_stat(train_df, column_name, col_type)
    validate_summ_stat = get_summary_stat(cv_df, column_name, col_type)

    lis = []
    for key, pop_val in dframe1.items():
        lis.append(
            OrderedDict(
                {
                    "Attribute": key,
                    "Population": pop_val,
                    "TrainSample": train_summ_stat[key],
                    "TestSample": test_summ_stat[key],
                    "Validate Sample": validate_summ_stat[key],
                }
            )
        )
    listdf = pd.DataFrame(lis, None)

    fig1 = plot_hist(df, column_name, "Before Splitting", get_color(0), col_type)
    fig2 = plot_hist(
        train_df, column_name, "After Train Splitting", get_color(1), col_type
    )
    fig3 = plot_hist(
        test_df, column_name, "After Test Splitting", get_color(2), col_type
    )
    fig4 = plot_hist(
        cv_df, column_name, "After Validate Splitting", get_color(3), col_type
    )

    return fig1, fig2, fig3, fig4, listdf, train_df, test_df, cv_df


def strat_split(df, column_name, ratio1, ratio2, action, col_type):
    """
    Perform stratified split into train, test, and validation sets and return plots and stats.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Column to stratify on.
        ratio1 (float): Train split ratio.
        ratio2 (float): Test split ratio from remaining data.
        color (any): Unused; kept for compatibility.
        action (str): 'submit' or 'save'.
        col_type (str): Column type for plotting ('categorical' or 'numerical').

    Returns:
        Tuple: (fig1, fig2, fig3, fig4, summary_df, train_df, test_df, validate_df)
    """

    categorical_feature = [
        key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ["object"]
    ]
    data_encoded, encoders, categorical_features = encoded_data(df, categorical_feature)

    x_train, train = startified_splitting(data_encoded, column_name, ratio1)
    cv, test = startified_splitting(x_train, column_name, ratio2)

    train_df = decode_dataset(train, encoders, categorical_features)
    test_df = decode_dataset(test, encoders, categorical_features)
    cv_df = decode_dataset(cv, encoders, categorical_features)

    if action == "save":
        return [], [], [], [], [], train_df, test_df, cv_df

    dframe1 = get_summary_stat(df, column_name, col_type)
    test_summ_stat = get_summary_stat(test_df, column_name, col_type)
    train_summ_stat = get_summary_stat(train_df, column_name, col_type)
    validate_summ_stat = get_summary_stat(cv_df, column_name, col_type)

    lis = []
    for key, pop_val in dframe1.items():
        lis.append(
            OrderedDict(
                {
                    "Attribute": key,
                    "Population": pop_val,
                    "TrainSample": train_summ_stat[key],
                    "TestSample": test_summ_stat[key],
                    "Validate Sample": validate_summ_stat[key],
                }
            )
        )
    listdf = pd.DataFrame(lis, None)

    fig1 = plot_hist(df, column_name, "Before Splitting", get_color(0), col_type)
    fig2 = plot_hist(
        train_df, column_name, "After Train Splitting", get_color(1), col_type
    )
    fig3 = plot_hist(
        test_df, column_name, "After Test Splitting", get_color(2), col_type
    )
    fig4 = plot_hist(
        cv_df, column_name, "After Validate Splitting", get_color(3), col_type
    )

    return fig1, fig2, fig3, fig4, listdf, train_df, test_df, cv_df
