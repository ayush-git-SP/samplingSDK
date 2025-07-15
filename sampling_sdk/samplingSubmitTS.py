"""
Functions for handling time series data sampling.
"""

import re
import traceback
from dateutil import parser

import pandas as pd
import plotly.express as px

from sampling_sdk.HelperScripts import get_color
from .HelperScripts import get_summary_stats


def minmax(df, column_name=0):
    """
    Get start and end dates from a DataFrame column, handling date parsing if needed.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str or int, optional): Column name or index to analyze (default is 0).

    Returns:
        pd.DataFrame: Table with 'Start Date' and 'End Date' for the given column.

    Notes:
        - Attempts to parse dates; if fails, uses original values.
        - Sorts data by the column before extracting min/max.
    """

    temp_column_name = column_name + "Temp"
    date_time_format = False
    basic_stat = {}
    try:
        df[temp_column_name] = pd.to_datetime(df[column_name])
        date_time_format = True
    except Exception:
        traceback.print_exc()
        df[temp_column_name] = df[column_name]
        date_time_format = False
    if date_time_format:
        df = df.sort_values(by=temp_column_name)
        basic_stat["End Date"] = df[column_name].iloc[-1]
        basic_stat["Start Date"] = df[column_name].iloc[0]
        df = df.drop([temp_column_name], axis=1)
        table_one = []
        for attr, val in basic_stat.items():
            table_one.append({"attribute": attr, column_name: str(val)})
        table_1 = pd.DataFrame(table_one)
        return table_1

    else:
        df = df.sort_values(by=[column_name])
        basic_stat["End Date"] = df[column_name].iloc[-1]
        basic_stat["Start Date"] = df[column_name].iloc[0]
        table_one = []
        for attr, val in basic_stat.items():
            table_one.append({"attribute": attr, column_name: str(val)})

        table_1 = pd.DataFrame(table_one)
        return table_1


def SamplingTimeseriesSplit(df, column_name, start_date, end_date):
    """
    Split time series data into in-sample and out-of-sample based on date range.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Date column name.
        start_date (str or datetime): Start date for in-sample period.
        end_date (str or datetime): End date for in-sample period.

    Returns:
        Tuple:
            data1 (pd.DataFrame): In-sample data after min-max scaling.
            data2 (pd.DataFrame): Out-of-sample data after min-max scaling.
            df_insample (pd.DataFrame): Original in-sample data.
            df_outsample (pd.DataFrame): Original out-of-sample data.

    Notes:
        - Attempts to convert the date column to datetime; if fails, processes as-is.
        - Sorts data before splitting.
    """
    temp_column_name = column_name + "Temp"
    date_time_format = False
    try:
        df[temp_column_name] = pd.to_datetime(df[column_name])
        date_time_format = True
    except Exception:
        # traceback.print_exc()
        df[temp_column_name] = df[column_name]
        date_time_format = False
    if date_time_format:
        df = df.sort_values(by=temp_column_name)
        temp_df = pd.to_datetime([start_date, end_date])
        start_date = temp_df[0]
        end_date = temp_df[1]
        lst_date_column = df[temp_column_name].tolist()
        for i, value in enumerate(lst_date_column):
            if value >= start_date:
                start = i
                break

        for i, value in reversed(list(enumerate(lst_date_column))):
            if value <= end_date:
                end = i
                break

        df = df.drop([temp_column_name], axis=1)
        df_insample = df[start : end + 1]
        df_outsample = df.drop(df.index[range(start, end)])
        data1 = minmax(df_insample, column_name)
        data2 = minmax(df_outsample, column_name)
        return data1, data2, df_insample, df_outsample
    else:
        df = df.sort_values(by=[column_name])
        lst_date_column = df[column_name].tolist()
        for i, value in enumerate(lst_date_column):
            if value >= start_date:
                start = i
                break

        for i, value in reversed(list(enumerate(lst_date_column))):
            if value <= end_date:
                end = i
                break

        df = df.drop([temp_column_name], axis=1)
        df_insample = df[start : end + 1]
        df_outsample = df.drop(df.index[range(start, end)])
        data1 = minmax(df_insample, column_name)
        data2 = minmax(df_outsample, column_name)
        return data1, data2, df_insample, df_outsample


def sampleTSsubmit(df, column_name, from_ts, to_ts, target):
    """
    Generate time series plots and summary statistics for in-sample and out-of-sample splits.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Date/time column name.
        from_ts (str or datetime): Start date for in-sample period.
        to_ts (str or datetime): End date for in-sample period.
        target (str): Target variable to analyze.

    Returns:
        Tuple:
            graphs (list): List of plotly figures for in-sample and out-of-sample data.
            res_table1 (pd.DataFrame): Summary stats for in-sample data.
            res_table2 (pd.DataFrame): Summary stats for out-of-sample data.
            summary_insample (pd.DataFrame): Detailed summary for in-sample target.
            summary_outsample (pd.DataFrame): Detailed summary for out-of-sample target.
    """

    default_colors = [get_color(i) for i in range(3)]

    res_table1, res_table2, df_insample, df_outsample = SamplingTimeseriesSplit(
        df, column_name, from_ts, to_ts
    )

    graphs = []

    fig1 = px.line(df_insample, x=column_name, y=target)
    fig1.update_traces(line_color=default_colors[0])
    fig1.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all"),
            ]
        ),
    )
    graphs.append(fig1)

    fig2 = px.line(df_outsample, x=column_name, y=target)
    fig2.update_traces(line_color=default_colors[1])
    fig2.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all"),
            ]
        ),
    )
    graphs.append(fig2)

    df_insample = df_insample.dropna(subset=[target])
    df_outsample = df_outsample.dropna(subset=[target])

    summary_insample = get_summary_stats(df_insample[target], target)
    summary_outsample = get_summary_stats(df_outsample[target], target)

    return graphs, res_table1, res_table2, summary_insample, summary_outsample


def SamplingTimeseriesSave(df, column_name, start_date, end_date):
    """
    Split DataFrame into in-sample and out-of-sample sets based on date range.

    Args:
        df (pd.DataFrame): Input data.
        column_name (str): Date column name.
        start_date (str or datetime): Start date for in-sample data.
        end_date (str or datetime): End date for in-sample data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (in-sample DataFrame, out-of-sample DataFrame).

    Notes:
        - Attempts to convert date column to datetime; if fails, processes as-is.
        - Sorts data by date before splitting.
    """

    temp_column_name = column_name + "Temp"
    date_time_format = False
    try:
        df[temp_column_name] = pd.to_datetime(df[column_name])
        date_time_format = True
    except Exception:
        traceback.print_exc()
        df[temp_column_name] = df[column_name]
        date_time_format = False
    if date_time_format:
        df = df.sort_values(by=temp_column_name)
        temp_df = pd.to_datetime([start_date, end_date])
        start_date = temp_df[0]
        end_date = temp_df[1]
        lst_date_column = df[temp_column_name].tolist()
        for i, value in enumerate(lst_date_column):
            if value >= start_date:
                start = i
                break

        for i, value in reversed(list(enumerate(lst_date_column))):
            if value <= end_date:
                end = i
                break

        df = df.drop([temp_column_name], axis=1)
        df_insample = df[start:end]
        df_outsample = df.drop(df.index[range(start, end)])
        return df_insample, df_outsample
    else:
        df = df.sort_values(by=[column_name])
        lst_date_column = df[column_name].tolist()
        for i, value in enumerate(lst_date_column):
            if value >= start_date:
                start = i
                break

        for i, value in reversed(list(enumerate(lst_date_column))):
            if value <= end_date:
                end = i
                break

        df = df.drop([temp_column_name], axis=1)
        df_insample = df[start:end]
        df_outsample = df.drop(df.index[range(start, end)])
        return df_insample, df_outsample


def convert_ddmmyyyy_to_q(date):
    """
    Convert date from 'ddmmyyyy' format to 'YYYY Qx' quarter format.

    Args:
        date (str): Input date string in 'ddmmyyyy' or other parseable format.

    Returns:
        str: Date string in 'YYYY Qx' format (e.g., '2023 Q2').
    """
    pattern = r"\d{4} Q[1-4]"
    if re.match(pattern, str(date)):
        return date
    else:
        d = parser.parse(date)
        if d.month in (1, 2, 3):
            date = f"{d.year} Q1"
        elif d.month in (4, 5, 6):
            date = f"{d.year} Q2"
        elif d.month in (7, 8, 9):
            date = f"{d.year} Q3"
        elif d.month in (10, 11, 12):
            date = f"{d.year} Q4"
        return date


def convert_from_yyyy_q(out):
    """
    Convert date string from 'YYYY Q' format to 'YYYY-MM-DD' format (start of quarter).

    Args:
        out (str): Date string in 'YYYY Qx' format (e.g., '2023 Q2').

    Returns:
        str or None: Converted date string as 'YYYY-MM-DD', or None if conversion fails.
    """
    try:
        d = str(out).split()
        year = d[0]
        quarter = re.sub("Q", "", d[1])
        month = str(((int(quarter) - 1) * 3) + 1)
        if month != "10":
            month = "0" + month
        out = year + "-" + month + "-1"
        return out
    except Exception:
        traceback.print_exc()
        raise
