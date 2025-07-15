"""
This module provides functions for various data processing tasks including time zone handling,
progress bar updates, and interaction with cloud storage services.
The module imports necessary modules such as pytz, json, BytesIO, List, numpy, pandas, re, bs4,
and pyarrow.dataset.
"""
import re
import json
from typing import List
from datetime import datetime
import base64
from pandas.arrays import IntegerArray


import numpy as np
import pandas as pd
import bs4
import pytz
from .global_var import (
    dtype_conv,
    int_,
    string_,
    date_,
)

TIME_ZONE = "Asia/Kolkata"
TIME_STAMP_WITH_TIME_ZONE = "timestamp with time zone"
DATETIME64 = "datetime64[ns]"
config = {"displayModeBar": True}

def fig_to_b64(fig):
    """Convert a Plotly figure to a base64-encoded PNG image."""
    img_bytes = fig.to_image(format="png")
    b64 = base64.b64encode(img_bytes).decode('utf-8')
    return b64

INT = ["smallint",
       "integer",
       "bigint",
       "decimal",
       "numeric",
       "real",
       "double precision",
       "smallserial",
       "serial",
       "bigserial",
       "bit",
       "tinyint",
       "mediumint",
       "float",
       "int64",
       "float64",
       "int32",
       "float32",
       "int16",
       "float16",
       "int8",
       "float8",
       'Int8',
       'Int16',
       'Int64',
       'Float64',
       'Float8',
       'Float16',
       'Float32',
       'Int32',
       ]
string = [
    "character varying",
    "varchar",
    "character",
    "text",
    "blob",
    "enum",
    "binary",
    "varbinary",
    "object",
]
date = [
    "timestamp with timezone",
    "timestamp with time zone",
    "timestamp without time zone",
    "timestamp",
    "datetime",
    "datetime64",
]
features = []
schema = {
    "object": "text",
    "int64": "bigint",
    "float64": "float",
    "int32": "bigint",
    "float32": "float",
    "int16": "bigint",
    "float16": "float",
    "int8": "bigint",
    "float8": "float",
    "bool": "bool",
    "datetime64": date[1],
    "datetime64[ns]": date[1],
    "category": "text",
    'Int8': 'int',
    'Int16': 'int',
    'Int64': 'int',
    'Float64': 'float',
    'Float8': 'float',
    'Float16': 'float',
    'Float32': 'float',
    'Int32': 'int',
}

def get_column_type(df, column_name, CategoricalThreshold):
    dtype = str(df[column_name].dtype)
    if dtype in INT:
        if df[column_name].nunique() <= CategoricalThreshold:
            column_type = "catcont"
        else:
            column_type = "continuous"
    elif dtype in date:
        column_type = "timestamp"
    else:
        if df[column_name].nunique() <= CategoricalThreshold:
            column_type = "categorical"
        else:
            column_type = "string"
    return column_type


def get_summary_stat(df, column_name, col_type):
    def calculate_basic_stats(df, col_name):
        basic_stat = {}
        if col_type == "categorical" or col_type == "catcont"  or col_type == "string":
            basic_stat['Count'] = float(df[col_name].count())
            basic_stat['Most Frequent Observation (Mode)'] = str(df[col_name].mode().iloc[0])
            basic_stat['Unique'] = float(df[col_name].nunique())
        else:
            basic_stat['Count'] = float(df[col_name].count())
            basic_stat['Min'] = float(df[col_name].min())
            basic_stat['Mean'] = float(df[col_name].mean())
            basic_stat['Median'] = float(df[col_name].median())
            basic_stat['Max'] = float(df[col_name].max())
            basic_stat['Range'] = float(df[col_name].max() - df[col_name].min())
            basic_stat['Variance'] = float(df[col_name].var())
            basic_stat['Standard Deviation'] = float(df[col_name].std())
            basic_stat['Coefficient of Variation'] = float(df[col_name].std() / df[col_name].mean())
            basic_stat['Skewness'] = float(df[col_name].skew())
            basic_stat['Kurtosis'] = float(df[col_name].kurtosis())
            basic_stat['5%'] = float(df[col_name].quantile(0.05))
            basic_stat['95%'] = float(df[col_name].quantile(0.95))
        return basic_stat

    basic_stat_df = calculate_basic_stats(df, column_name)
    return basic_stat_df

def update_chart_modebar(fig, chart_title):
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
    # fig += f'<script>{code}</script>'
    return fig
    
def update_chart_html(txt, chart_id, task_id):
    """
    Update the HTML code of a chart for embedding images.
    Args:
        txt (str): The original HTML code containing the chart.
        chart_id (str): The identifier of the chart.
    Returns:
        str: The updated HTML code with image embedding, or the original text if
        no suitable match is found.
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


def update_chart_svg(txt, chartId, task_id):
    """
    Function to update the font of plotly chart in SVG format
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
    svg_html = '<html><head><meta charset="utf-8"/><style>body { font-family: "Poppins", sans-serif !important; }</style><link href="https://fonts.googleapis.com/css?family=Poppins:300,400,500,700&display=swap" rel="stylesheet"/><style> *::-webkit-scrollbar { width: 5px; height: 0.3rem; } *:hover::-webkit-scrollbar { width: 5px; height: 8px; } *::-webkit-scrollbar-track { // box-shadow: inset 0 0 5px $primaryColor; } *::-webkit-scrollbar-thumb { background: #aecdfc; border-radius: 10px; opacity: 0.5; } *::-webkit-scrollbar-track { // box-shadow: inset 0 0 5px $primaryColor; border-radius: 10px; } </style>    </head><body><div> <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: "local"};</script>'
    svg_html += str(svg) + "</div></body></html>"
    return svg_html


# def update_chart3(fig,static_chart,chart_title=None):
#     """This function takes figure as an input and return the customized
#     figure with changes in font and color"""
#     OPEN_SANS = "Open Sans"
#     fig.update_layout(
#         font_family=OPEN_SANS,
#         title={"x": 0.5, "font_size": 14},
#         legend=dict(
#             orientation="v",
#             font=dict(family=OPEN_SANS, size=12, color="#222222"),
#             yanchor="top",
#             y=1.5,
#             xanchor="right",
#             x=1,
#         ),
#     )
#     fig.update_layout(
#         {"plot_bgcolor": "#FFFFFF", "paper_bgcolor": "#FFFFFF"},
#         modebar=dict(bgcolor="rgba(34,34,34,0.6)"),
#     )
#     fig.update_xaxes(
#         showline=True,
#         linewidth=1,
#         linecolor="black",
#         title_font=dict(family=OPEN_SANS, size=14, color="#222222"),
#         tickfont=dict(family=OPEN_SANS, size=12, color="#222222"),
#         showgrid=False,
#     )
#     fig.update_yaxes(
#         showline=True,
#         linewidth=1,
#         linecolor="black",
#         title_font=dict(family=OPEN_SANS, size=14, color="#222222"),
#         tickfont=dict(family=OPEN_SANS, size=12, color="#222222"),
#         showgrid=False,
#     )
#     fig2 = fig.to_image('png')
#     if static_chart:
#         fig = fig.to_image("svg").decode("utf-8")
#     else:
#         fig = update_chart_modebar(fig, chart_title) 
#     return fig,fig2

def update_chart3(fig, static_chart=False, chart_title=None):
    """
    Updates a Plotly figure with custom styles. Optionally returns a static image (PNG).
    
    Args:
        fig (go.Figure): Plotly figure object.
        static_chart (bool): Whether to generate static image output.
        chart_title (str): Optional chart title.

    Returns:
        Tuple: (Styled Plotly Figure or HTML, PNG image as bytes or None)
    """
    OPEN_SANS = "Open Sans"

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
        except Exception as e:
            print("[⚠️ Warning] Static image generation failed:", str(e))
    else:
        try:
            fig = update_chart_modebar(fig, chart_title)
        except Exception as e:
            print("[⚠️ Warning] Modebar enhancement failed:", str(e))

    return fig, fig2


def time_update(task_id):
    """
    Generate a dictionary with task time information.
    Args:
        key (str): The key associated with the task.
    Returns:
        dict: A dictionary with task-related time information.
    """
    data = {}
    data["task_id"] = task_id
    time_zone = pytz.timezone(TIME_ZONE)
    data["execution_start_time"] = str(datetime.now(time_zone))
    return data


def get_charts(fig, chart_id, static_chart, task_id,chart_title):
    """This function get charts in either SVG or HTML format."""
    fig.update_layout(font_family="Poppins, sans-serif")
    fig,fig2 = update_chart3(fig,static_chart,chart_title)
    b64=fig_to_b64 (fig2)
    if static_chart:
        fig =update_chart_svg(fig, chart_id, task_id)
    else:
        fig = update_chart_html(fig, chart_id, task_id)
    return fig,b64

def get_metadata(df, bigdata=False, timeseries_column=""):
    """This function returns metadata"""
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
    """This function maps data type to the given format"""
    DATETIME64 = "datetime64[ns]"
    TIME_STAMP_WITH_TIME_ZONE = "timestamp with time zone"
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
        DATETIME64: TIME_STAMP_WITH_TIME_ZONE,
        "category": "TEXT",
    }
    fields = {col: dtype_conv2[str(dtype)]
              for col, dtype in zip(df.columns, df.dtypes)}
    for i in fields:
        if fields[i] in ["int", "float"]:
            df[i] = df[i].astype(fields[i])
    return df

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling NumPy objects during JSON serialization."""
    def default(self, obj):
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
    Generate a staged table name based on the input parameters.
    Args:
        table_name (str): The base name of the table.
        method (str): The method identifier, either "cluster" or "sys".
        ratio (int or float): The ratio or factor associated with the staging.
    Returns:
        tuple: A tuple containing the staged table name, the stage number, and
               the method identifier.
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
            table_name[0: len(table_name) - 1 - i]
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
    Generates a staged table name for resampling purposes.
    Args:
        table_name (str): The base name of the table.
    Returns:
        tuple: A tuple containing the staged table name, the stage number, and
               the stage identifier.
    """
    txt = table_name[::-1]
    for i in range(len(table_name)):
        if txt[i] == "_" or ord(txt[i]) < 48 or ord(txt[i]) > 57:
            break
    if txt[i] == "_" and i != 0:
        txt = txt[0:i]
        txt = txt[::-1]
        stage_numb = int(txt) + 1
        txt = table_name[0: len(table_name) - 1 - i] + "_" + str(stage_numb)
        x = "_" + str(stage_numb)
        return txt, stage_numb, x
    else:
        stage_numb = 1
        txt = table_name + "_" + str(stage_numb)
        x = "_" + str(stage_numb)
        return txt, stage_numb, x


def schema(data_type):
    """
    Converts a given data type into its corresponding database schema type.
    """
    if data_type == "object":
        return "TEXT"
    elif data_type == "int64":
        return "BIGINT"
    elif data_type == "float64":
        return "float"
    elif data_type == "bool":
        return "bool"
    elif data_type == DATETIME64:
        return TIME_STAMP_WITH_TIME_ZONE


def get_schema(df):
    """Generate a dictionary representing the database schema for a DataFrame."""
    schema_ = {col: schema(dtype) for col, dtype in zip(df.columns, df.dtypes)}
    return schema_



def datatype_check(location: str, dataset: str, column_name: str, nrows: int = None):
    """Perform basic data type categorization for a specific column in a dataset."""
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


# not for sampling


def percentile(location: str, dataset: str, column_name: str, nrows: int = None):
    """Calculate percentiles of a specific column in a dataset."""
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
    # To take values only upto 2 decimal places
    value = ["{:.2f}".format(i) for i in value]
    res = {"Percentile": percent, "Value": value}
    res_df = pd.DataFrame(res)
    return res_df.to_dict(orient="records")


def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    """Optimizes memory usage by downcasting float columns in a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to optimize.
    Returns:
        pd.DataFrame: The optimized DataFrame.
    """
    floats = df.select_dtypes(include=["float64"]).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast="float")
    return df


def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize memory usage by downcasting integer columns in a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to optimize.
    Returns:
        pd.DataFrame: The optimized DataFrame.
    """
    ints = df.select_dtypes(include=["int64"]).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast="integer")
    return df


def optimize_objects(df: pd.DataFrame, datetime_features: List[str]) -> pd.DataFrame:
    """
    Optimizes memory usage of object columns in a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to optimize.
        datetime_features (List[str]): List of column names considered as datetime features.
    Returns:
        pd.DataFrame: The optimized DataFrame.
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


def optimize(df: pd.DataFrame, datetime_features=None):
    """
    Optimizes memory usage of a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to optimize.
        datetime_features (List[str], optional): List of column names considered as datetime features.
    Returns:
        pd.DataFrame: The optimized DataFrame.
    """
    if datetime_features is None:
        datetime_features = []
    return optimize_floats(optimize_ints(optimize_objects(df, datetime_features)))


def csv_datatype(csv_file):
    """
    Determine the appropriate database data types for columns in a CSV file.
    Args:
        csv_file: The input CSV file represented as a DataFrame.
    Returns:
        dict: A dictionary mapping column names to their corresponding database data types.
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
        elif data_type[i] == DATETIME64:
            data_type[i] = TIME_STAMP_WITH_TIME_ZONE
        else:
            try:
                json.loads(data_type[i])
            except Exception:
                data_type[i] = "TEXT"
        dict_csv[column_name[i]] = data_type[i]

    return dict_csv

def get_summary_stats(df, column_name):
    
    """Fucntion to get continious stats
    Args:
        table_data (df): pandas dataframe
        column_name (str): Column name
    Returns:
        tables : pandas.dataframe
    """
    basic_stat = {}
    column = df
    basic_stat['Count'] = column.count()
    basic_stat['Min'] = column.min()
    basic_stat['5%'] = column.quantile(0.05)
    basic_stat['Mean'] = column.mean()
    basic_stat['Median'] = column.median()
    basic_stat['95%'] = column.quantile(0.95)
    basic_stat['Max'] = column.max()
    basic_stat['Range'] = column.max() - column.min()
    basic_stat['Variance'] = column.var()
    basic_stat['Standard Deviation'] = column.std()
    basic_stat['Coefficient of Variation'] = column.std()/column.mean()
    basic_stat['Skewness'] = column.skew()
    basic_stat['Kurtosis'] = column.kurtosis()
    basic_stat['Missing Value Count'] = column.isnull().sum()
    table_one = []

    for attr in basic_stat.keys():
        if abs(float(basic_stat[attr] * 100)) < 1:
            table_one.append(
                {"attribute": attr, column_name: float(basic_stat[attr])})
        else:
            if np.isnan(basic_stat[attr]):
                continue
            table_one.append(
                {"attribute": attr, column_name: float(basic_stat[attr])})
    return table_one  


def get_color(index):
    print(f"[DEBUG] Getting color for index: {index}")
    palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf', '#1a55FF', '#AA3377',
        '#66CCEE', '#228833', '#CCBB44', '#EE6677',
    ]
    return palette[index % len(palette)]
