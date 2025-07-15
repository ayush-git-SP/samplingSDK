"""module for the resampling subsection of Sampling"""

from collections import Counter

from numpy import log
import pandas as pd
import plotly.graph_objects as go

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, RandomOverSampler, SMOTE
from imblearn.under_sampling import (
    EditedNearestNeighbours,
    NearMiss,
    NeighbourhoodCleaningRule,
    OneSidedSelection,
    RandomUnderSampler,
    TomekLinks,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from .HelperScripts import dtypes_changes, get_color


def encoded_data(data, categorical_feature):
    """
    Encode categorical columns using LabelEncoder.

    Args:
        data (pd.DataFrame): Input data.
        categorical_feature (list): List of categorical column names.

    Returns:
        Tuple[pd.DataFrame, dict, list]: Encoded DataFrame, encoders, and categorical features.
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
        categorical_features (list): Columns to decode.

    Returns:
        pd.DataFrame: Decoded DataFrame.
    """
    df = data.copy()
    for feat in categorical_features:
        df[feat] = encoders[feat].inverse_transform(df[feat].astype(int))
    return df


def shanon(seq):
    """
    Calculate Shannon entropy normalized by log of number of classes.

    Args:
        seq (iterable): Input sequence.

    Returns:
        float: Normalized Shannon entropy.
    """
    n = len(seq)
    classes = [(clas, float(count)) for clas, count in Counter(seq).items()]
    k = len(classes)

    H = -sum([(count / n) * log((count / n)) for clas, count in classes])
    return H / log(k)


def rand_under_sample(df, column_name):
    """
    Perform random undersampling to balance the dataset.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Target column.

    Returns:
        pd.DataFrame: Resampled dataset.
    """
    df = dtypes_changes(df)
    rus = RandomUnderSampler(sampling_strategy="auto")
    x_resample, y_resample = rus.fit_resample(df, df[column_name])
    return x_resample


def nearmiss_under(df, column_name):
    """
    Perform NearMiss undersampling to balance the dataset.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Target column.

    Returns:
        pd.DataFrame: Resampled dataset.
    """
    categorical_feature = [
        key for key in dict(df.dtypes) if dict(df.dtypes)[key] == "object"
    ]
    for col in df.columns:
        if col not in categorical_feature:
            df[col].fillna(-1, inplace=True)
    df = dtypes_changes(df)
    nm = NearMiss(version=3)
    x_resample, y_resample = nm.fit_resample(df, df[column_name])
    return x_resample


def tomeklink(df, column_name):
    """
    Perform Tomek Links undersampling to clean overlapping samples.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Target column.

    Returns:
        pd.DataFrame: Resampled dataset.
    """
    categorical_feature = [
        key for key in dict(df.dtypes) if dict(df.dtypes)[key] == "object"
    ]
    for col in df.columns:
        if col not in categorical_feature:
            df[col].fillna(-1, inplace=True)
    df = dtypes_changes(df)
    tl = TomekLinks()
    x_resample, y_resample = tl.fit_resample(df, df[column_name])
    return x_resample


def edited_nearest_neighbour(df, column_name):
    """
    Perform Edited Nearest Neighbours undersampling to clean dataset.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Target column.

    Returns:
        pd.DataFrame: Resampled dataset.
    """
    categorical_feature = [
        key for key in dict(df.dtypes) if dict(df.dtypes)[key] == "object"
    ]
    for col in df.columns:
        if col not in categorical_feature:
            df[col].fillna(-1, inplace=True)
    df = dtypes_changes(df)
    enn = EditedNearestNeighbours()
    x_resample, y_resample = enn.fit_resample(df, df[column_name])
    return x_resample


def one_sided_selection(df, column_name):
    """
    Perform One-Sided Selection undersampling to balance the dataset.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Target column.

    Returns:
        pd.DataFrame: Resampled dataset.
    """
    categorical_feature = [
        key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ["object"]
    ]

    for col in df.columns:
        if col not in categorical_feature:
            df[col].fillna(-1, inplace=True)

    df = dtypes_changes(df)
    oss = OneSidedSelection(random_state=42)
    x_resample, y_resample = oss.fit_resample(df, df[column_name])
    return x_resample


def neighbourhood_cleaning_rule(df, column_name):
    """
    Apply Neighbourhood Cleaning Rule undersampling to balance the dataset.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Target column.

    Returns:
        pd.DataFrame: Resampled dataset.
    """
    categorical_feature = [
        key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ["object"]
    ]
    for col in df.columns:
        if col not in categorical_feature:
            df[col].fillna(-1, inplace=True)
    df = dtypes_changes(df)
    ncr = NeighbourhoodCleaningRule()
    x_resample, y_resample = ncr.fit_resample(df, df[column_name])
    return x_resample


def rand_over_sample(df, column_name):
    """
    Perform random oversampling to balance the dataset.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Target column.

    Returns:
        pd.DataFrame: Resampled dataset.
    """

    df = dtypes_changes(df)
    ros = RandomOverSampler(sampling_strategy="auto")
    x_resample, y_resample = ros.fit_resample(df, df[column_name])
    return x_resample


def smote(df, column_name):
    """
    Apply SMOTE oversampling to balance the dataset.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Target column.

    Returns:
        pd.DataFrame: Resampled dataset.
    """

    categorical_feature = [
        key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ["object"]
    ]
    for col in df.columns:
        if col not in categorical_feature:
            df[col].fillna(-1, inplace=True)
    df = dtypes_changes(df)
    sm = SMOTE(k_neighbors=NearestNeighbors(n_neighbors=5, algorithm="brute"))
    x_resample, y_resample = sm.fit_resample(df, df[column_name])
    return x_resample


def borderline_smote(df, column_name):
    """
    Apply Borderline-SMOTE oversampling to balance the dataset.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Target column.

    Returns:
        pd.DataFrame: Resampled dataset.
    """
    categorical_feature = [
        key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ["object"]
    ]
    for col in df.columns:
        if col not in categorical_feature:
            df[col].fillna(-1, inplace=True)
    df = dtypes_changes(df)
    bs = BorderlineSMOTE(random_state=42)
    x_resample, y_resample = bs.fit_resample(df, df[column_name])
    return x_resample


def adasyn(df, column_name):
    """
    Apply ADASYN oversampling to balance the dataset.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Target column.

    Returns:
        pd.DataFrame: Resampled dataset.
    """

    categorical_feature = [
        key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ["object"]
    ]
    for col in df.columns:
        if col not in categorical_feature:
            df[col].fillna(-1, inplace=True)
    df = dtypes_changes(df)
    ada = ADASYN(random_state=42)
    x_resample, y_resample = ada.fit_resample(df, df[column_name])
    return x_resample


def smoteen(df, column_name):
    """
    Apply SMOTE-ENN combined sampling to balance the dataset.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Target column.

    Returns:
        pd.DataFrame: Resampled dataset.
    """
    categorical_feature = [
        key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ["object"]
    ]
    for col in df.columns:
        if col not in categorical_feature:
            df[col].fillna(-1, inplace=True)
    df = dtypes_changes(df)
    sme = SMOTEENN(random_state=42)
    x_resample, y_resample = sme.fit_resample(df, df[column_name])
    return x_resample


def smotetomek(df, column_name):
    """
    Apply SMOTE-Tomek combined sampling to balance the dataset.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Target column.

    Returns:
        pd.DataFrame: Resampled dataset.
    """
    categorical_feature = [
        key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ["object"]
    ]
    for col in df.columns:
        if col not in categorical_feature:
            df[col].fillna(-1, inplace=True)
    df = dtypes_changes(df)
    smt = SMOTETomek(random_state=42)
    x_resample, y_resample = smt.fit_resample(df, df[column_name])
    return x_resample


def undersample(df, column_name, method):
    """
    Perform undersampling on the dataset based on the chosen method.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Target column for sampling.
        method (int): Undersampling method selector (1-6).

    Returns:
        pd.DataFrame: Resampled dataset.
    """
    choice = method

    if choice == 1:
        x_resample = rand_under_sample(df, column_name)
    elif choice == 2:
        x_resample = nearmiss_under(df, column_name)
    elif choice == 3:
        x_resample = tomeklink(df, column_name)
    elif choice == 4:
        x_resample = edited_nearest_neighbour(df, column_name)
    elif choice == 5:
        x_resample = one_sided_selection(df, column_name)
    elif choice == 6:
        x_resample = neighbourhood_cleaning_rule(df, column_name)
    return x_resample


def combisample(df, column_name, method):
    """
    Perform combined over- and undersampling based on the chosen method.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Target column for sampling.
        method (int): Combined sampling method selector (1-2).

    Returns:
        pd.DataFrame: Resampled dataset.
    """
    choice = method
    if choice == 1:
        x_resample = smoteen(df, column_name)
    elif choice == 2:
        x_resample = smotetomek(df, column_name)
    return x_resample


def oversample(df, column_name, method):
    """
    Perform oversampling on the dataset based on the chosen method.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Target column for sampling.
        method (int): Oversampling method selector (1-4).

    Returns:
        pd.DataFrame: Resampled dataset.
    """

    choice = method
    if choice == 1:
        x_resample = rand_over_sample(df, column_name)
    elif choice == 2:
        x_resample = smote(df, column_name)
    elif choice == 3:
        x_resample = boderline_smote(df, column_name)
    elif choice == 4:
        x_resample = adasyn(df, column_name)
    return x_resample


def plot_fun(column_name, df, stat, index=0, col_type=None):
    """
    Create a histogram or bar plot for a column based on its type.

    Args:
        column_name (str): Column to plot.
        df (pd.DataFrame): Data source.
        stat (str): Label for the plot (e.g., "Before", "After").
        index (int, optional): Color index. Defaults to 0.
        col_type (str, optional): Column type to decide plot style.

    Returns:
        plotly.graph_objects.Figure: The plot figure.
    """
    color = get_color(index)

    col_data = df[column_name].dropna()

    if col_data.empty:
        return go.Figure()

    if col_type in ("categorical", "catcont", "string"):
        counts = col_data.value_counts()
        trace = go.Bar(
            x=counts.index.tolist(),
            y=counts.values.tolist(),
            marker_color=color,
            name=stat,
        )
    else:
        trace = go.Histogram(x=col_data, name=stat, marker_color=color)

    fig = go.Figure(data=[trace])
    fig.update_layout(
        title=f"{stat} Distribution",
        xaxis=dict(title=column_name),
        yaxis=dict(title="Count"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        barmode="overlay",
    )

    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        type=(
            "category" if col_type in ("categorical", "catcont", "string") else "linear"
        ),
    )
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black")

    return fig


def compare_and_return(df, df_resampled, column_name, lst, col_type):
    """
    Generate before/after plots and summary tables for resampling comparison.

    Args:
        df (pd.DataFrame): Original data.
        df_resampled (pd.DataFrame): Resampled data.
        column_name (str): Column to analyze.
        lst (list): List of categories or colors.
        col_type (str): Column type.

    Returns:
        tuple: (plot_before, plot_after, before_table, after_table, df_resampled)
    """

    categorical_feature = [
        key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ["object"]
    ]

    plot_before = plot_fun(column_name, df, "Before", index=0, col_type=col_type)
    plot_after = plot_fun(
        column_name, df_resampled, "After", index=1, col_type=col_type
    )

    if column_name in categorical_feature or df[column_name].nunique() < 15:
        x = df[column_name].value_counts()
        before_table = pd.DataFrame()
        before_table["count"] = x
        l = [str(t) for t in before_table.index]
        before_table.insert(0, column_name, l)

        x = df_resampled[column_name].value_counts()
        after_table = pd.DataFrame()
        after_table["count"] = x
        l = [str(t) for t in after_table.index]
        after_table.insert(0, column_name, l)

        return plot_before, plot_after, before_table, after_table, df_resampled

    else:
        x = df[column_name].describe()
        before_table = pd.DataFrame(x)
        l = [str(t) for t in before_table.index]
        before_table.insert(0, "Attribute", l)

        x = df_resampled[column_name].describe()
        after_table = pd.DataFrame(x)
        l = [str(t) for t in after_table.index]
        after_table.insert(0, "Attribute", l)

        return plot_before, plot_after, before_table, after_table, df_resampled


def resample(df, column_name, method, method2, lst, action, col_type):
    """
    Perform resampling on the data with specified method and return results.

    Args:
        df (pd.DataFrame): Input dataset.
        column_name (str): Target column.
        method (int): Resampling type (1=undersample, 2=oversample, 3=combined).
        method2 (str): Specific resampling technique.
        lst (list): Color or category list.
        action (str): 'submit' or other action.
        col_type (str): Column type.

    Returns:
        tuple: Plots, tables, and resampled DataFrame.
    """
    categorical_feature = [
        key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ["object"]
    ]
    (data_encoded, encoders, categorical_features) = encoded_data(
        df, categorical_feature
    )

    choice = method
    if choice == 1:
        x_resample = undersample(
            data_encoded,
            column_name,
            method2,
        )
    elif choice == 2:
        x_resample = oversample(data_encoded, column_name, method2)
    elif choice == 3:
        x_resample = combisample(data_encoded, column_name, method2)

    df_resampled = decode_dataset(x_resample, encoders, categorical_features)
    if action == "submit":
        (plot_before, plot_after, before_table, after_table, df_resampled) = (
            compare_and_return(df, df_resampled, column_name, lst, col_type)
        )
        return plot_before, plot_after, before_table, after_table, df_resampled
    return [], [], [], [], df_resampled
