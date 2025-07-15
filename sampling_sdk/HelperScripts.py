"""
This module provides functions for various data processing tasks including time zone handling,
progress bar updates, and interaction with cloud storage services.
The module imports necessary modules such as pytz, json, BytesIO, List, numpy, pandas, re, bs4,
and pyarrow.dataset.
"""

import base64
import json
import re
import traceback
from typing import List

import bs4
import numpy as np
import pandas as pd
from pandas.arrays import IntegerArray

from .global_var import (
    dtype_conv,
    int_,
    string_,
    date_,
    INT,
    schema,
    date,
    OPEN_SANS,
)

config = {"displayModeBar": True}
features = []


def fig_to_b64(fig):
    """Convert a Plotly figure to a base64-encoded PNG image."""
    img_bytes = fig.to_image(format="png")
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return b64


def get_column_type(df, column_name, categorical_threshold):
    """
    Determine the type of a dataframe column based on dtype and cardinality.

    Args:
        df (pd.DataFrame): Input dataframe.
        column_name (str): Column name.
        categorical_threshold (int): Threshold of unique values to consider categorical.

    Returns:
        str: One of "catcont", "continuous", "timestamp", "categorical", or "string".
    """
    dtype = str(df[column_name].dtype).lower()
    if dtype in INT:
        if df[column_name].nunique() <= categorical_threshold:
            column_type = "catcont"  # Categorical with continuous-like values
        else:
            column_type = "continuous"
    elif dtype in date:
        column_type = "timestamp"
    else:
        if df[column_name].nunique() <= categorical_threshold:
            column_type = "categorical"
        else:
            column_type = "string"
    return column_type


def get_summary_stat(df, column_name, col_type):
    """
    Calculate summary statistics for a given column.

    Args:
        df (pd.DataFrame): Input dataframe.
        column_name (str): Column name.
        col_type (str): Column type (categorical, continuous, catcont, string).

    Returns:
        dict: Dictionary of summary statistics.
    """
    basic_stat = {}

    if col_type in ("categorical", "catcont", "string"):
        basic_stat["Count"] = float(df[column_name].count())
        basic_stat["Most Frequent Observation"] = str(df[column_name].mode().iloc[0])
        basic_stat["Unique"] = float(df[column_name].nunique())
    else:
        series = df[column_name]
        basic_stat["Count"] = float(series.count())
        basic_stat["Min"] = float(series.min())
        basic_stat["Mean"] = float(series.mean())
        basic_stat["Median"] = float(series.median())
        basic_stat["Max"] = float(series.max())
        basic_stat["Range"] = float(series.max() - series.min())
        basic_stat["Variance"] = float(series.var())
        basic_stat["Standard Deviation"] = float(series.std())
        mean = series.mean()
        basic_stat["Coefficient of Variation"] = (
            float(series.std() / mean) if mean != 0 else None
        )
        basic_stat["Skewness"] = float(series.skew())
        basic_stat["Kurtosis"] = float(series.kurtosis())
        basic_stat["5%"] = float(series.quantile(0.05))
        basic_stat["95%"] = float(series.quantile(0.95))

    return basic_stat


def update_chart_modebar(fig, chart_title):
    """
    Enhances the accessibility and interactivity of a Plotly figure's modebar.

    This function generates JavaScript code to:
    - Add accessible labels (`aria-label`) to modebar buttons with the chart title.
    - Enable keyboard navigation through the modebar using Tab and Enter keys.
    - Apply custom focus outlines for accessibility.
    - Handle relayout and mousedown events for updated accessibility behavior.

    Args:
        fig: plotly.graph_objects.Figure
            The Plotly figure to enhance.
        chart_title: str
            The title of the chart to append to button labels.

    Returns:
        fig: plotly.graph_objects.Figure
            The same figure object (unchanged).

    Note:
        This function currently returns the figure unchanged.
        The JavaScript `code` is generated as a string but not embedded or executed automatically.
    """
    code = f"""
    document.addEventListener("DOMContentLoaded", function () {{
        function updateModebarButtons() {{
            const modebarGroups = document.querySelectorAll(".modebar-group");

            modebarGroups.forEach((group) => {{
                const buttons = group.querySelectorAll("a");

                buttons.forEach((button) => {{
                    const originalTitle = button.getAttribute("data-title");
                    const newCustomValue = `${{originalTitle}} {chart_title}`;

                    button.setAttribute("aria-label", newCustomValue);
                    button.setAttribute("role", "button");
                    button.setAttribute("tabindex", "0");
                }});
            }});
        }}

        document.addEventListener("plotly_relayout", updateModebarButtons);
        updateModebarButtons();

        const modebarButtons = document.querySelectorAll(".modebar-btn");
        const imageContainer = document.getElementById("image-container");
        let currentIndex = -1;

        function focusModebarButton(index) {{
            if (currentIndex !== -1) {{
                modebarButtons[currentIndex].blur();
                modebarButtons[currentIndex].style.outline = "none";
            }}

            if (index < 0 || index >= modebarButtons.length) {{
                currentIndex = -1;
                return false;
            }}

            currentIndex = index;
            modebarButtons[currentIndex].focus();
            modebarButtons[currentIndex].style.outline = "1px solid #272D55";
            announce(modebarButtons[currentIndex].getAttribute("aria-label"));
            return true;
        }}

        document.addEventListener("keydown", function (event) {{
            if (modebarButtons.length === 0) return;

            if (event.key === "Tab") {{
                if (currentIndex === -1) {{
                    if (event.shiftKey) {{
                        focusImageContainer();
                    }}
                    return;
                }}

                event.preventDefault();
                const success = focusModebarButton(
                    event.shiftKey ? currentIndex - 1 : currentIndex + 1
                );

                if (!success) {{
                    currentIndex = -1;
                    document.activeElement.blur();
                }}
            }} else if (event.key === "Enter" && currentIndex !== -1) {{
                event.preventDefault();
                const focusedButton = modebarButtons[currentIndex];
                if (focusedButton) {{
                    focusedButton.click();
                }}
            }}
        }});

        document.addEventListener("mousedown", function (event) {{
            if (!event.target.closest(".modebar-btn") && currentIndex !== -1) {{
                modebarButtons[currentIndex].blur();
                modebarButtons[currentIndex].style.outline = "none";
                currentIndex = -1;
            }}
        }});

        modebarButtons.forEach((button) => {{

            button.addEventListener("focus", () => {{
                announce(button.getAttribute("aria-label"));
            }});

            button.addEventListener("keydown", (event) => {{
                if (event.key === "Enter") {{
                    event.preventDefault();
                    button.click();
                }}
            }});
        }});

        function focusImageContainer() {{
            if (currentIndex !== -1) {{
                modebarButtons[currentIndex].blur();
                modebarButtons[currentIndex].style.outline = "none";
            }}
            currentIndex = -1;
            imageContainer.focus();
            imageContainer.style.outline = "1px solid #272D55";
        }}

        imageContainer.setAttribute("tabindex", "0");
        imageContainer.setAttribute("role", "img");
        imageContainer.addEventListener("focus", () => {{
            announce("Image focused");
        }});
    }});
    """
    return fig


def update_chart_html(txt, chart_id):
    """
    Update the HTML code of a Plotly chart to embed it as a PNG image.

    Args:
        txt (str): Original HTML code containing the chart.
        chart_id (str): Identifier of the chart.

    Returns:
        str: Updated HTML code with image embedding or original if no match found.
    """
    soup = bs4.BeautifulSoup(txt, "html.parser")
    x = re.search(r"Plotly.newPlot.*\)", str(soup))
    if x:
        val = (
            str(x.group())
            + '.then(data=>{Plotly.toImage(data, {format:"png", height: 360, width: 640}).then(data =>'
            " {window.parent.postMessage({base64 : data, chartId :"
            + str(chart_id)
            + '}, "*");});})'
        )
        n_html = re.sub(r"Plotly.newPlot.*\)", val, str(soup))
        return n_html
    return txt


def update_chart_svg(txt, chartId):
    """
    Update the font and embed JavaScript to export Plotly SVG charts as base64 images.

    Args:
        txt (str): Original SVG HTML code.
        chartId (str or int): Chart identifier.

    Returns:
        str: Modified HTML with embedded JavaScript for exporting SVG as base64.
    """
    soup = bs4.BeautifulSoup(txt, "lxml")
    new_tag = soup.new_tag("script")
    new_tag.string = (
        r"<![CDATA[function getDataURL(svgXml) {var svgString = new XMLSerializer().serializeToString(svgXml);svgString= svgString.replace(/[\u00A0-\u2666]/g, function(c) {return '&#' + c.charCodeAt(0) + ';';}); var decoded = decodeURIComponent(encodeURIComponent(svgString));var base64 = btoa(decoded);var imgSource = `data:image/svg+xml;base64,${base64}`;return(imgSource);} window.onload= function(e) {setTimeout(() => {const svg = document.querySelector('svg');let data = getDataURL(svg);window.parent.postMessage({ base64: data, chartId: "
        + str(chartId)
        + " }, '*');}, 1000);}]]>"
    )

    soup.html.body.svg.insert(1, new_tag)
    svg = soup.svg.extract()
    tag = svg.script
    tag["type"] = "text/javascript"
    svg_html = (
        '<html><head><meta charset="utf-8"/>'
        '<style>body { font-family: "Poppins", sans-serif !important; }</style>'
        '<link href="https://fonts.googleapis.com/css?family=Poppins:300,400,500,700&display=swap" rel="stylesheet"/>'
        "<style> *::-webkit-scrollbar { width: 5px; height: 0.3rem; } *:hover::-webkit-scrollbar { width: 5px; height: 8px; } *::-webkit-scrollbar-track { } *::-webkit-scrollbar-thumb { background: #aecdfc; border-radius: 10px; opacity: 0.5; } *::-webkit-scrollbar-track { border-radius: 10px; } </style>"
        "</head><body><div>"
        '<script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: "local"};</script>'
        + str(svg)
        + "</div></body></html>"
    )
    return svg_html


def update_chart3(fig, static_chart=False, chart_title=None):
    """
    Apply custom styling to a Plotly figure and optionally generate a static PNG image.

    Args:
        fig (go.Figure): Plotly figure to update.
        static_chart (bool): Whether to generate a static image (PNG).
        chart_title (str, optional): Title of the chart.

    Returns:
        tuple: (Updated figure, PNG image bytes or None)
    """

    fig.update_layout(
        font_family=OPEN_SANS,
        title={"x": 0.5, "font_size": 14},
        legend=dict(
            orientation="v",
            font=dict(family=OPEN_SANS, size=12, color="#222222"),
            yanchor="top",
            y=1.5,
            xanchor="right",
            x=1,
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        modebar=dict(bgcolor="rgba(34,34,34,0.6)"),
    )

    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        title_font=dict(family=OPEN_SANS, size=14, color="#222222"),
        tickfont=dict(family=OPEN_SANS, size=12, color="#222222"),
        showgrid=False,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        title_font=dict(family=OPEN_SANS, size=14, color="#222222"),
        tickfont=dict(family=OPEN_SANS, size=12, color="#222222"),
        showgrid=False,
    )

    fig2 = None
    if static_chart:
        try:
            fig2 = fig.to_image(format="png")
        except Exception:
            traceback.print_exc()
            raise
    else:
        try:
            fig = update_chart_modebar(fig, chart_title)
        except Exception:
            traceback.print_exc()
            raise

    return fig, fig2


def get_charts(fig, chart_id, static_chart, chart_title):
    """
    Prepare chart figure in SVG or HTML format with additional updates.

    Args:
        fig: Plotly figure object.
        chart_id (str): Identifier for the chart.
        static_chart (bool): Flag to determine SVG or HTML output.
        chart_title (str): Title of the chart.

    Returns:
        tuple: Updated figure and base64 representation.
    """
    fig.update_layout(font_family="Poppins, sans-serif")
    fig, fig2 = update_chart3(fig, static_chart, chart_title)
    b64 = fig_to_b64(fig2)
    if static_chart:
        fig = update_chart_svg(fig, chart_id)
    else:
        fig = update_chart_html(fig, chart_id)
    return fig, b64


def get_metadata(df, bigdata=False, timeseries_column=""):
    """
    Extract metadata from DataFrame including missing values, unique counts, and data types.

    Args:
        df (pd.DataFrame): Input data.
        bigdata (bool): Flag for large datasets.
        timeseries_column (str or list): Columns considered as timeseries.

    Returns:
        dict: Metadata summary including missing value percentage, unique counts, etc.
    """
    time_uniques, unique_map = {}, {}
    stages = len(df.index)
    fields = {
        col: dtype_conv[str(dtype).lower()] for col, dtype in zip(df.columns, df.dtypes)
    }
    unique_map = df.nunique().to_dict()
    missing_map = {}
    for column in df.columns:
        missing_map[column] = df[column].isna().sum()
    if (
        timeseries_column is not None
        and not bigdata
        and isinstance(timeseries_column, list)
    ):
        for col in timeseries_column:
            time_uniques[col] = df[col].replace(np.NaN, "").unique().flatten()
    mvs = sum(missing_map.values())

    return {
        "mvperc": round(mvs / df.size * 100, 2),
        "no_uniques": unique_map,
        "time_uniques": time_uniques,
        "stages": stages,
        "fields": fields,
        "missing_map": missing_map,
    }


def dtypes_changes(df):
    """
    Convert DataFrame columns to target data types based on predefined mapping.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with updated dtypes.
    """
    dtype_conv2 = {
        "object": "TEXT",
        "Int8": "int",
        "Int16": "int",
        "Int64": "int",
        "Float64": "float",
        "Float8": "float",
        "Float16": "float",
        "Float32": "float",
        "Int32": "int",
        "int8": "int",
        "int16": "int",
        "int64": "int",
        "float64": "float",
        "float8": "float",
        "float16": "float",
        "float32": "float",
        "int32": "int",
        "bool": "bool",
        "datetime64[ns]": "timestamp with time zone",
        "category": "TEXT",
    }
    fields = {col: dtype_conv2[str(dtype)] for col, dtype in zip(df.columns, df.dtypes)}
    for i in fields:
        if fields[i] in ["int", "float"]:
            df[i] = df[i].astype(fields[i])
    return df


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to serialize NumPy data types and related objects.

    Supports:
    - NumPy integer and unsigned integer types
    - NumPy floating types
    - NumPy complex numbers (serialized as dict with real and imag)
    - NumPy arrays (converted to lists)
    - NumPy booleans
    - Python sets (converted to lists)
    - pandas Categorical and IntegerArray (converted to lists)
    - NumPy void types (serialized as None)
    """

    def default(self, obj):
        """
        Serialize NumPy and related objects to JSON-compatible types.

        Args:
            obj: Object to serialize.

        Returns:
            JSON-serializable representation of obj.
        """
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        elif isinstance(obj, np.bool_):
            return bool(obj)

        elif isinstance(obj, set):
            return list(obj)

        elif isinstance(obj, IntegerArray) or isinstance(obj, pd.Categorical):
            return list(obj)

        elif isinstance(obj, np.void):
            return None


def stage_table(table_name, method, ratio):
    """
    Generate a staged table name based on method and ratio.

    Args:
        table_name (str): Base table name.
        method (str): "cluster" or other method identifier.
        ratio (int or float): Ratio or factor for staging.

    Returns:
        tuple: (staged table name, stage number, method code)
    """
    if method == "cluster":
        x = "clus"
    else:
        x = "sys"
    txt = table_name[::-1]
    for i in range(len(table_name)):
        if txt[i] == "_" or ord(txt[i]) < 48 or ord(txt[i]) > 57:
            break
    if txt[i] == "_" and i != 0:
        txt = txt[0:i]
        txt = txt[::-1]
        stage_numb = int(txt) + 1
        txt = (
            table_name[0 : len(table_name) - 1 - i]
            + "_"
            + str(ratio)
            + "_"
            + str(stage_numb)
        )
        return txt, stage_numb, x
    else:
        stage_numb = 1
        txt = table_name + "_" + x + "_" + str(ratio) + "_" + str(stage_numb)
        return txt, stage_numb, x


def stage_table_resampling(table_name):
    """
    Generate a staged table name for resampling, incrementing existing stage.

    Args:
        table_name (str): Base table name.

    Returns:
        tuple: (staged table name, stage number, stage suffix)
    """
    txt = table_name[::-1]
    for i in range(len(table_name)):
        if txt[i] == "_" or ord(txt[i]) < 48 or ord(txt[i]) > 57:
            break
    if txt[i] == "_" and i != 0:
        txt = txt[0:i]
        txt = txt[::-1]
        stage_numb = int(txt) + 1
        txt = table_name[0 : len(table_name) - 1 - i] + "_" + str(stage_numb)
        x = "_" + str(stage_numb)
        return txt, stage_numb, x
    else:
        stage_numb = 1
        txt = table_name + "_" + str(stage_numb)
        x = "_" + str(stage_numb)
        return txt, stage_numb, x


def get_schema(df):
    """
    Generate a dictionary representing the database schema for a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        dict: Mapping of column names to database schema types.
    """
    schema_ = {col: schema(dtype) for col, dtype in zip(df.columns, df.dtypes)}
    return schema_


def datatype_check(location: str, dataset: str, column_name: str, nrows: int = None):
    """
    Categorize the data type of a specified column in a dataset.

    Args:
        location (str): Data source location.
        dataset (str): Dataset identifier.
        column_name (str): Column to check.
        nrows (int, optional): Number of rows to load.

    Returns:
        int: Integer code representing the data type category.
             (1=int, 2=string, 3=date, 4=other)
    """
    df = fetch_data_as_df(location, dataset, nrows)
    dtype_ = str(df[column_name].dtype)
    res = dtype_conv[dtype_].lower()
    x = 4
    if res in int_:
        x = 1
    elif res in string_:
        x = 2
    elif res in date_:
        x = 3
    return x


def percentile(location: str, dataset: str, column_name: str, nrows: int = None):
    """
    Calculate specified percentiles for a column in a dataset.

    Args:
        location (str): Data source location.
        dataset (str): Dataset identifier.
        column_name (str): Column to analyze.
        nrows (int, optional): Number of rows to load.

    Returns:
        list of dict: List of percentiles with corresponding values.
    """
    df = fetch_data_as_df(location, dataset, nrows)
    percent = [
        "0%",
        "1%",
        "5%",
        "10%",
        "20%",
        "30%",
        "40%",
        "50%",
        "60%",
        "70%",
        "80%",
        "95%",
        "99.50%",
        "99.90%",
    ]
    percent_val = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 95, 99.50, 99.90]
    value = [np.percentile(df[column_name], i) for i in percent_val]
    value = ["{:.2f}".format(i) for i in value]
    res = {"Percentile": percent, "Value": value}
    res_df = pd.DataFrame(res)
    return res_df.to_dict(orient="records")


def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast float64 columns to reduce memory usage.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Optimized DataFrame.
    """
    floats = df.select_dtypes(include=["float64"]).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast="float")
    return df


def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast int64 columns to reduce memory usage.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Optimized DataFrame.
    """
    ints = df.select_dtypes(include=["int64"]).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast="integer")
    return df


def optimize_objects(df: pd.DataFrame, datetime_features: List[str]) -> pd.DataFrame:
    """
    Optimize memory by converting object columns to categories or datetime.

    Args:
        df (pd.DataFrame): Input DataFrame.
        datetime_features (List[str]): Columns to convert to datetime.

    Returns:
        pd.DataFrame: Optimized DataFrame.
    """
    for col in df.select_dtypes(include=["object"]):
        if col not in datetime_features:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if float(num_unique_values) / num_total_values < 0.5:
                df[col] = df[col].astype("category")
        else:
            df[col] = pd.to_datetime(df[col])
    return df


def optimize(df: pd.DataFrame, datetime_features=None) -> pd.DataFrame:
    """
    Optimize memory usage of a DataFrame including ints, floats, and objects.

    Args:
        df (pd.DataFrame): Input DataFrame.
        datetime_features (List[str], optional): Columns to convert to datetime.

    Returns:
        pd.DataFrame: Optimized DataFrame.
    """
    if datetime_features is None:
        datetime_features = []
    return optimize_floats(optimize_ints(optimize_objects(df, datetime_features)))


def csv_datatype(csv_file: pd.DataFrame) -> dict:
    """
    Map DataFrame columns to appropriate database data types.

    Args:
        csv_file (pd.DataFrame): Input DataFrame.

    Returns:
        dict: Mapping from column names to database types.
    """
    column_name = list(csv_file.columns)
    csv_file = pd.DataFrame(csv_file)
    for df in csv_file:
        if csv_file[df].dtype == "object":
            csv_file[df] = csv_file[df].astype("str")

    data_type = csv_file.dtypes
    data_type = pd.Series(data_type).tolist()
    dict_csv = {}
    for i in range(len(data_type)):
        if data_type[i] == "object":
            data_type[i] = "TEXT"
        elif data_type[i] == "int64":
            data_type[i] = "BIGINT"
        elif data_type[i] == "float64":
            data_type[i] = "float"
        elif data_type[i] == "bool":
            data_type[i] = "bool"
        elif data_type[i] == "datetime64[ns]":
            data_type[i] = "timestamp with time zone"
        else:
            try:
                json.loads(data_type[i])
            except Exception:
                data_type[i] = "TEXT"
        dict_csv[column_name[i]] = data_type[i]

    return dict_csv


def get_summary_stats(df: pd.DataFrame, column_name: str) -> list:
    """
    Compute summary statistics for a continuous column.

    Args:
        df (pd.DataFrame): DataFrame containing the column.
        column_name (str): Name of the column.

    Returns:
        list of dict: List of statistics with attribute names and values.
    """
    basic_stat = {}
    column = df
    basic_stat["Count"] = column.count()
    basic_stat["Min"] = column.min()
    basic_stat["5%"] = column.quantile(0.05)
    basic_stat["Mean"] = column.mean()
    basic_stat["Median"] = column.median()
    basic_stat["95%"] = column.quantile(0.95)
    basic_stat["Max"] = column.max()
    basic_stat["Range"] = column.max() - column.min()
    basic_stat["Variance"] = column.var()
    basic_stat["Standard Deviation"] = column.std()
    basic_stat["Coefficient of Variation"] = column.std() / column.mean()
    basic_stat["Skewness"] = column.skew()
    basic_stat["Kurtosis"] = column.kurtosis()
    basic_stat["Missing Value Count"] = column.isnull().sum()
    table_one = []

    for attr, val in basic_stat.items():
        if abs(float(val * 100)) < 1:
            table_one.append({"attribute": attr, column_name: float(val)})
        else:
            if np.isnan(val):
                continue
            table_one.append({"attribute": attr, column_name: float(val)})

    return table_one


def get_color(index: int) -> str:
    """
    Get a color from a predefined palette based on the index.

    Args:
        index (int): Index for color selection.

    Returns:
        str: Hex color code.
    """
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#1a55FF",
        "#AA3377",
        "#66CCEE",
        "#228833",
        "#CCBB44",
        "#EE6677",
    ]
    return palette[index % len(palette)]
