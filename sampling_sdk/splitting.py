"""Module for Splitting method of sampling."""
from .HelperScripts import (get_summary_stat)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import warnings
from collections import OrderedDict
from sampling_sdk.HelperScripts import get_color
warnings.simplefilter("ignore")

PLOT_BGCOLOR = 'rgba(0,0,0,0)'

def plot_hist(df, column_name, name, color, col_type):
    """This function plots histogram.

    Args:
        df : Dataframe.
        column_name (str): Name of the columns to be used for plotting.
        name (str): Name of histogram plot
        color (str): Color choice foor plot.
        column_type (str, optional): Type of column.

    Returns:
        Returns histogram plot figure.
    """
    fig1 = go.Figure(
        layout=dict(
            xaxis=dict(title=column_name.capitalize()),
            yaxis=dict(title='Count')
        ))
    fig1.add_trace(go.Histogram(
        x=df[column_name],
        name=name,
        marker_color=color
    ))

    fig1.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                       'paper_bgcolor': 'rgba(0,0,0,0)'})
    if col_type == "categorical" or col_type == "catcont":
        fig1.update_xaxes(title_text=column_name.capitalize(
        ), showline=True, linewidth=1, linecolor='black', type='category')
    else:
        fig1.update_xaxes(title_text=column_name.capitalize(),
                          showline=True, linewidth=1, linecolor='black')
    fig1.update_yaxes(title_text='Count', showline=True,
                      linewidth=1, linecolor='black')
    fig1.layout.legend = dict(
        x=0.4,
        y=-0.2,
        traceorder='normal',
        orientation="h",
        font=dict(
            size=12, ))
    return fig1


def encoded_data(data, categorical_feature):
    """This function converts data into a format required for a number of information processing needs."""
    data_encoded = data.copy()
    categorical_names = {}
    encoders = {}

    # Use Label Encoder for categorical columns (including target column)
    for feature in categorical_feature:
        le = LabelEncoder()
        le.fit(data_encoded[feature].astype(str))

        data_encoded[feature] = le.transform(data_encoded[feature].astype(str))

        categorical_names[feature] = le.classes_
        encoders[feature] = le

    return data_encoded, encoders, categorical_feature

    # decoding_data


def decode_dataset(data, encoders, categorical_features):
    """This function converts back encoded data into its actual form."""
    df = data.copy()
    for feat in categorical_features:
        df[feat] = encoders[feat].inverse_transform(df[feat].astype(int))
    return df

# def sp(df, column_name, ratio1, ratio2, method, action, col_type):
#     """This function splits the data based on the method provided."""
#     if method == 1:
#         fig, fig1, fig2, fig3, listdf, train_df, test_df, cv_df = split_data(
#             df, column_name, ratio1, ratio2, lst, action, col_type)
#         return fig, fig1, fig2, fig3, listdf, train_df, test_df, cv_df
#     elif method == 2:
#         fig, fig1, fig2, fig3, listdf, train_df, test_df, cv_df = strat_split(
#             df, column_name, ratio1, ratio2, lst, action, col_type)
#         return fig, fig1, fig2, fig3, listdf, train_df, test_df, cv_df


def sp(df, column_name, ratio1, ratio2, method, action, col_type):
    """This function splits the data based on the method provided."""
    if method == 1:
        return split_data(df, column_name, ratio1, ratio2, action, col_type)
    elif method == 2:
        return strat_split(df, column_name, ratio1, ratio2, action, col_type)


def stratified_split(df, column_name, ratio):
    """This function splits data into train and test. """
    bins = np.linspace(0, len(df), 5)
    y_binned = np.digitize(df[column_name], bins)
    x_train, x_test, y_train, y_test = train_test_split(
        df, df[column_name], test_size=ratio, stratify=y_binned, random_state=42)
    return x_train, x_test

# splitting for categorical variable


def stratified_split1(df, column_name, ratio):
    """This function split data into train and test."""
    x_train, x_test, y_train, y_test = train_test_split(
        df, df[column_name], test_size=ratio, stratify=df[column_name], random_state=42)
    return x_train, x_test


def startified_splitting(df, column_name, ratio):
    """This function do splitting for categorical variable."""
    cat_var = [key for key in dict(df.dtypes) if dict(df.dtypes)[
        key] in ['object']]
    if column_name in cat_var:
        df[column_name] = df[column_name].fillna('NOT_GIVEN')
        x_train, x_test = stratified_split1(df, column_name, ratio)
        return x_train, x_test
    else:
        df[column_name] = df[column_name].fillna(-1)
        x_train, x_test = stratified_split(df, column_name, ratio)
        return x_train, x_test


def plot_pie(df, color):
    """This function generates pie plot for splitted data."""
    colors = [color[0], color[1], color[2]]
    val_cnt = df.value_counts()
    values = val_cnt.tolist()
    labels = val_cnt.keys().tolist()
    fig1 = go.Figure(
        data=[go.Pie(labels=labels, values=values, hole=0, pull=[0.007, 0.007, 0.007])])
    fig1.update_traces(marker=dict(colors=colors))
    fig1.update_layout({'plot_bgcolor': PLOT_BGCOLOR,
                       'paper_bgcolor': PLOT_BGCOLOR})
    fig1.update_layout(
        font_family="Poppins, sans-serif",
        title_x=0.5,
        legend_title_text='Legend',
        legend={"font": {'size': 12}, "x": 0.8,
                "y": 0.9, "traceorder": "normal"},
        title_font_color="black")
    return fig1


def plot_pie2(value1, value2, value3, color):
    """This function generate pie chart and split it into 3 values for their respective ratio."""
    colors = [color[0], color[1], color[2]]
    labels = ['Mean', 'Median', 'Mode']
    values = [value1, value2, value3]

    fig = go.Figure(
        data=[go.Pie(labels=labels, values=values, hole=0, pull=[0.007, 0.007, 0.007])])
    fig.update_traces(marker=dict(colors=colors))
    fig.update_layout({'plot_bgcolor': PLOT_BGCOLOR,
                      'paper_bgcolor': PLOT_BGCOLOR})
    fig.update_layout(
        font_family="Poppins, sans-serif",
        title_x=0.5,
        legend_title_text='Legend',
        legend={"font": {'size': 12}, "x": 0.8,
                "y": 0.9, "traceorder": "normal"},
        title_font_color="black")
    return fig



def split_data(df, column_name, ratio1, ratio2, action, col_type):
    """Splits data into Train/Test/Validation and returns figures and summary stats."""
    from sampling_sdk.HelperScripts import get_color

    categorical_feature = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['object']]
    data_encoded, encoders, categorical_features = encoded_data(df, categorical_feature)

    x = data_encoded
    y = data_encoded[column_name]

    train_df, x_train, y_train, y_test = train_test_split(x, y, train_size=ratio1, random_state=42)
    test_df, cv_df, y_test, y_cv = train_test_split(x_train, y_test, train_size=ratio2, random_state=42)

    train_df = decode_dataset(train_df, encoders, categorical_features)
    test_df = decode_dataset(test_df, encoders, categorical_features)
    cv_df = decode_dataset(cv_df, encoders, categorical_features)

    if action == 'save':
        return [], [], [], [], [], pd.DataFrame(train_df), pd.DataFrame(test_df), pd.DataFrame(cv_df)

    dframe1 = get_summary_stat(df, column_name, col_type)
    test_summ_stat = get_summary_stat(test_df, column_name, col_type)
    train_summ_stat = get_summary_stat(train_df, column_name, col_type)
    validate_summ_stat = get_summary_stat(cv_df, column_name, col_type)

    lis = []
    for key in dframe1:
        lis.append(OrderedDict({
            "Attribute": key,
            "Population": dframe1[key],
            "TrainSample": train_summ_stat[key],
            "TestSample": test_summ_stat[key],
            "Validate Sample": validate_summ_stat[key]
        }))
    listdf = pd.DataFrame(lis, None)

    fig1 = plot_hist(df, column_name, "Before Splitting", get_color(0), col_type)
    fig2 = plot_hist(train_df, column_name, "After Train Splitting", get_color(1), col_type)
    fig3 = plot_hist(test_df, column_name, "After Test Splitting", get_color(2), col_type)
    fig4 = plot_hist(cv_df, column_name, "After Validate Splitting", get_color(3), col_type)

    return fig1, fig2, fig3, fig4, listdf, train_df, test_df, cv_df




def strat_split(df, column_name, ratio1, ratio2, color, action, col_type):
    """
    Splits data into Train, Test, and Validation using stratified sampling.

    Args:
        df (pd.DataFrame): Input dataset.
        column_name (str): Column used for stratification.
        ratio1 (float): Ratio for training set.
        ratio2 (float): Ratio for test set (out of remaining).
        color (any): Unused, kept for compatibility.
        action (str): 'submit' or 'save'.
        col_type (str): Type of column for plotting.

    Returns:
        Tuple: figs, result table, train/test/validate dfs
    """
    categorical_feature = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['object']]
    data_encoded, encoders, categorical_features = encoded_data(df, categorical_feature)

    x_train, train = startified_splitting(data_encoded, column_name, ratio1)
    cv, test = startified_splitting(x_train, column_name, ratio2)

    train_df = decode_dataset(train, encoders, categorical_features)
    test_df = decode_dataset(test, encoders, categorical_features)
    cv_df = decode_dataset(cv, encoders, categorical_features)

    if action == 'save':
        return [], [], [], [], [], train_df, test_df, cv_df

    # Prepare summary table
    dframe1 = get_summary_stat(df, column_name, col_type)
    test_summ_stat = get_summary_stat(test_df, column_name, col_type)
    train_summ_stat = get_summary_stat(train_df, column_name, col_type)
    validate_summ_stat = get_summary_stat(cv_df, column_name, col_type)

    lis = []
    for key in dframe1:
        lis.append(OrderedDict({
            "Attribute": key,
            "Population": dframe1[key],
            "TrainSample": train_summ_stat[key],
            "TestSample": test_summ_stat[key],
            "Validate Sample": validate_summ_stat[key]
        }))
    listdf = pd.DataFrame(lis, None)

    # Use dynamic color indices
    fig1 = plot_hist(df, column_name, "Before Splitting", get_color(0), col_type)
    fig2 = plot_hist(train_df, column_name, "After Train Splitting", get_color(1), col_type)
    fig3 = plot_hist(test_df, column_name, "After Test Splitting", get_color(2), col_type)
    fig4 = plot_hist(cv_df, column_name, "After Validate Splitting", get_color(3), col_type)

    return fig1, fig2, fig3, fig4, listdf, train_df, test_df, cv_df
