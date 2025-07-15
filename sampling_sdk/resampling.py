"""module for the resampling subsection of Sampling"""

from .HelperScripts import dtypes_changes

from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
import plotly.graph_objects as go
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from numpy import log
from collections import Counter
import pandas as pd
import warnings

from .HelperScripts import get_color

warnings.simplefilter("ignore")


def encoded_data(data, categorical_feature):
    """
    This function converts data into a format required for a number of information processing needs.
    """
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


def decode_dataset(data, encoders, categorical_features):
    """This function converts back encoded data into its actual form."""
    df = data.copy()
    for feat in categorical_features:
        df[feat] = encoders[feat].inverse_transform(df[feat].astype(int))
    return df


def shanon(seq):
    """This function returns shanon entropy."""
    n = len(seq)
    classes = [(clas, float(count)) for clas, count in Counter(seq).items()]
    k = len(classes)

    H = -sum([(count / n) * log((count / n)) for clas, count in classes])
    return H / log(k)


def rand_under_sample(df, column_name):
    """This function returns result for random undersampling operation."""
    df = dtypes_changes(df)
    rus = RandomUnderSampler(sampling_strategy="auto")
    x_resample, y_resample = rus.fit_resample(df, df[column_name])
    return x_resample


def nearmiss_under(df, column_name):
    """This function is for performing near miss algorithm for balancing the dataset in undersampling operation."""
    categorical_feature = [
        key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ["object"]
    ]

    for col in df.columns:
        if col not in categorical_feature:
            df[col].fillna(-1, inplace=True)
    df = dtypes_changes(df)
    nm = NearMiss(version=3)
    x_resample, y_resample = nm.fit_resample(df, df[column_name])
    # print(x_resample)
    return x_resample


# tomeklink
def tomeklink(df, column_name):
    """This function is for tomek link method in undersampling operation."""
    categorical_feature = [
        key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ["object"]
    ]

    for col in df.columns:
        if col not in categorical_feature:
            df[col].fillna(-1, inplace=True)
    df = dtypes_changes(df)
    tl = TomekLinks()
    x_resample, y_resample = tl.fit_resample(df, df[column_name])
    # print(x_resample)
    return x_resample


# edited Nearest Neighbour
def edited_nearest_neighbour(df, column_name):
    """This function is for edited nearest neighbour method in undersampling operation."""
    categorical_feature = [
        key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ["object"]
    ]

    for col in df.columns:
        if col not in categorical_feature:
            df[col].fillna(-1, inplace=True)
    df = dtypes_changes(df)
    enn = EditedNearestNeighbours()
    x_resample, y_resample = enn.fit_resample(df, df[column_name])
    return x_resample


# One Sided Selection
def one_sided_selection(df, column_name):
    """This function is for one sided selection method in undersampling operation."""
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


# Neighbourhood Cleaning Rule
def neighbourhood_cleaning_rule(df, column_name):
    """This function is for neighbourhood cleaning rule method in undersampling operation."""
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


# RandomOversampling
def rand_over_sample(df, column_name):
    """
    This function returns result for random undersampling operation.
    """

    df = dtypes_changes(df)
    ros = RandomOverSampler(sampling_strategy="auto")
    x_resample, y_resample = ros.fit_resample(df, df[column_name])
    return x_resample


# Smote
def smote(df, column_name):
    """This function is for smote method in random oversampling operation."""

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


# Boderline SMOTE
def boderline_smote(df, column_name):
    """This function is for borderline smote method in random oversampling operation."""
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


# ADASYN
def adasyn(df, column_name):
    """This function is for Adaptic Synthetic method in random oversampling operation."""

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


# SMOTEEN
def smoteen(df, column_name):
    """This function is for Smote-Enn method in random oversampling operation."""
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


# SMOTETOMEK
def smotetomek(df, column_name):
    """This function is for smote-tomek method in random oversampling operation."""
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
    """This function returns output according to the method provided in undersampling operation."""
    print("inside undersample")
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
    """This function returns output according to the method provided."""
    print("inside combisample")
    choice = method
    if choice == 1:
        x_resample = smoteen(df, column_name)
    elif choice == 2:
        x_resample = smotetomek(df, column_name)
    return x_resample


def oversample(df, column_name, method):
    """This function returns output according to the method provided in oversampling operation."""
    print("inside oversample")

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
    from sampling_sdk.HelperScripts import get_color

    color = get_color(index)
    print(
        f"[DEBUG] Plotting {stat} - Column: {column_name}, Index: {index}, Color: {color}"
    )
    print(f"[DEBUG] Data Preview: {df[column_name].head(5)}")
    print(f"[DEBUG] Column Type: {col_type}")
    print(f"[DEBUG] Non-null values in column: {df[column_name].dropna().shape[0]}")

    # Remove NaNs
    col_data = df[column_name].dropna()

    if col_data.empty:
        print("[DEBUG] Warning: No data to plot.")
        return go.Figure()

    if col_type in ("categorical", "catcont", "string"):
        # Plot Bar chart for string/categorical
        counts = col_data.value_counts()
        trace = go.Bar(
            x=counts.index.tolist(),
            y=counts.values.tolist(),
            marker_color=color,
            name=stat,
        )
    else:
        # Plot Histogram for continuous
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

    print(f"[DEBUG] Figure created for: {stat}")
    return fig


def compare_and_return(df, df_resampled, column_name, lst, col_type):
    """
    Compare and return visualizations and data tables before and after resampling.
    Args:
        df (DataFrame): The original DataFrame.
        df_resampled (DataFrame): The resampled DataFrame.
        column_name (str): The column name for comparison.
        lst (list): List of categories.
    Returns:
        tuple: A tuple containing visualizations, data tables, and the resampled DataFrame.
            The tuple contains plot_before, plot_after, before_table, after_table, and df_resampled.
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
        # for continous
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
    """Methodology for resampling"""
    print("Resampling started")
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
