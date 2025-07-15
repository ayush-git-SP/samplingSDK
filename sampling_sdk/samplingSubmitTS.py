from dateutil import parser
import re
import pandas as pd
import plotly.express as px
from .HelperScripts import (
    get_summary_stats,
)
from sampling_sdk.HelperScripts import get_color


def minmax(df, column_name=0):
    """
    Input:- dataframe , column_name(if not present default 0)
    Output:- table with minimum and maximum value present in Insample and Outsample dataframe
    description:- If date is in english format, create new column with standard format
    and do sorting otherwise use as it is
    """
    temp_column_name = column_name + "Temp"
    date_time_format = False
    basic_stat = {}
    try:
        df[temp_column_name] = pd.to_datetime(df[column_name])
        date_time_format = True
    except:
        df[temp_column_name] = df[column_name]
        date_time_format = False
    if date_time_format:
        df = df.sort_values(by=temp_column_name)
        basic_stat["End Date"] = df[column_name].iloc[-1]
        basic_stat["Start Date"] = df[column_name].iloc[0]
        df = df.drop([temp_column_name], axis=1)
        table_one = []
        for attr in basic_stat.keys():
            table_one.append({"attribute": attr, column_name: str(basic_stat[attr])})
        table_1 = pd.DataFrame(table_one)
        return table_1

    else:
        df = df.sort_values(by=[column_name])
        basic_stat["End Date"] = df[column_name].iloc[-1]
        basic_stat["Start Date"] = df[column_name].iloc[0]
        table_one = []
        for attr in basic_stat.keys():
            table_one.append({"attribute": attr, column_name: str(basic_stat[attr])})
        table_1 = pd.DataFrame(table_one)
        return table_1


def SamplingTimeseriesSplit(df, column_name, start_date, end_date):
    """
    Input:- Dataframe , column_name
    Output:- Insample, OutSample, StartingDate, ENdingDate
    Description:- If input(start_date, end_date) is in English date formate, then convert into standard format
                    sort after chnaging the date format otherwise do as it is i.e normal approach
    """
    temp_column_name = column_name + "Temp"
    date_time_format = False
    try:
        df[temp_column_name] = pd.to_datetime(df[column_name])
        date_time_format = True
    except:
        df[temp_column_name] = df[column_name]
        date_time_format = False
    if date_time_format:
        df = df.sort_values(by=temp_column_name)
        temp_df = pd.to_datetime([start_date, end_date])
        start_date = temp_df[0]
        end_date = temp_df[1]
        lst_date_column = df[temp_column_name].tolist()
        for i in range(0, len(lst_date_column)):
            if lst_date_column[i] >= start_date:
                start = i
                break
        for i in range(len(lst_date_column) - 1, 0, -1):
            if lst_date_column[i] <= end_date:
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
        for i in range(0, len(lst_date_column)):
            if lst_date_column[i] >= start_date:
                start = i
                break
        for i in range(len(lst_date_column) - 1, 0, -1):
            if lst_date_column[i] <= end_date:
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
    Input:- Dataframe, column_name, from_ts(start_date), to_ts(end_date)
    Output:- Figures, Tables
    Description:- Submit function for time series sampling that returns figures and tables.
    """
    from sampling_sdk.HelperScripts import get_color

    default_colors = [get_color(i) for i in range(3)]

    res_table1, res_table2, df_insample, df_outsample = SamplingTimeseriesSplit(
        df, column_name, from_ts, to_ts
    )

    graphs = []

    # In-sample chart
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

    # Out-sample chart
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
    Input:- Dataframe, column_name, start_date, end_date
    Output:- Return insample and outsample dataframe
    Description:- Save the insample and outsample dataframe
                if date in English date then convert into standard date format else countinue with same column
    """
    temp_column_name = column_name + "Temp"
    date_time_format = False
    try:
        df[temp_column_name] = pd.to_datetime(df[column_name])
        date_time_format = True
    except:
        df[temp_column_name] = df[column_name]
        date_time_format = False
    if date_time_format:
        df = df.sort_values(by=temp_column_name)
        temp_df = pd.to_datetime([start_date, end_date])
        start_date = temp_df[0]
        end_date = temp_df[1]
        lst_date_column = df[temp_column_name].tolist()
        for i in range(0, len(lst_date_column)):
            if lst_date_column[i] >= start_date:
                start = i
                break
        for i in range(len(lst_date_column) - 1, 0, -1):
            if lst_date_column[i] <= end_date:
                end = i
                break
        df = df.drop([temp_column_name], axis=1)
        df_insample = df[start:end]
        df_outsample = df.drop(df.index[range(start, end)])
        return df_insample, df_outsample
    else:
        df = df.sort_values(by=[column_name])
        lst_date_column = df[column_name].tolist()
        for i in range(0, len(lst_date_column)):
            if lst_date_column[i] >= start_date:
                start = i
                break
        for i in range(len(lst_date_column) - 1, 0, -1):
            if lst_date_column[i] <= end_date:
                end = i
                break
        df = df.drop([temp_column_name], axis=1)
        df_insample = df[start:end]
        df_outsample = df.drop(df.index[range(start, end)])
        return df_insample, df_outsample


def convert_ddmmyyyy_to_q(date):
    """Convert Date format from ddmmyyyy to YYYY Q
    Args:
        date (string): input date of type ddmmyyyy
    Returns:
        date (string): return converted date fromat to YYYY Q
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
    """Convert date from YYYY Q to YYYY MM DD
    Returns:
        date(string): date convert to desired fromat
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
    except:
        return None


def convert_from_yyyy_q(out):
    """Convert date from YYYY Q to YYYY MM DD

    Returns:
        date(string): date convert to desired fromat
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
    except:
        return None
