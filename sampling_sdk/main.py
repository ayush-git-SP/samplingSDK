import re
import json
import traceback
from dateutil import parser
import pandas as pd

from .visualizeimbalance import plot_fun, shanon
from .HelperScripts import (
    get_column_type,
    get_schema,
    update_chart3,
    update_chart_html,
    stage_table_resampling,
)
from .sampling import sample
from .samplingSubmitTS import (
    SamplingTimeseriesSave,
    sampleTSsubmit,
    convert_ddmmyyyy_to_q,
    convert_from_yyyy_q,
)
from .splitting import sp
from .resampling import resample
from PIL import Image
from io import BytesIO

visualimbalance_chart_title = ["Count_Plot"]
samplingts_chart_title = [
    "Insample",
    "Outsample",
    "Insample Summary",
    "Outsample Summary",
]


class VisualizeImbalance:
    """
    Visualize Imbalance SDK Class:
    Provides static methods for:
    1. Count Plot (by_count)
    2. Shannon Entropy (shanonentropy)
    """

    @staticmethod
    def count_plot(parameters={}, df=pd.DataFrame()):
        """
        Generate a Count Plot for imbalance visualization.

        Args:
            parameters (dict): Must contain 'column_name', 'userpref', 'categorical_threshold'.
            df (pd.DataFrame): Input dataset.

        Returns:
            fig (Plotly/Matplotlib figure), Optional: base64 image if needed
        """
        try:
            print("Inside count_plot")
            col_name = parameters.get("column_name")
            categorical_threshold = parameters.get("categorical_threshold", 0)

            if df.dtypes[col_name] == object and isinstance(
                df[col_name].values[0], bytes
            ):
                df[col_name] = df[col_name].apply(lambda x: x.decode("utf-8"))

            col_type = get_column_type(df, col_name, categorical_threshold)
            fig = plot_fun(col_name, df, col_type)

            return fig

        except Exception:
            print("Exception in count_plot:", traceback.format_exc())
            return None

    @staticmethod
    def shannon_entropy(parameters={}, df=pd.DataFrame()):
        """
        Compute Shannon Entropy for imbalance visualization.

        Args:
            parameters (dict): Must contain 'column_name'.
            df (pd.DataFrame): Input dataset.

        Returns:
            list: List of dicts representing a table with entropy info
        """
        try:
            print("inside shannon entropy")
            col_name = parameters.get("column_name")

            if df.dtypes[col_name] == object and isinstance(
                df[col_name].values[0], bytes
            ):
                df[col_name] = df[col_name].apply(lambda x: x.decode("utf-8"))

            entropy = shanon(df[col_name])

            # Determine interpretation based on entropy value
            if entropy == "Not Applicable":
                interpretation = "Not Applicable"
            elif entropy >= 0.85:
                interpretation = "Highly Balanced"
            elif entropy >= 0.50:
                interpretation = "Moderately Balanced"
            else:
                interpretation = "Imbalanced"

            result_table = [
                {
                    "Variable": col_name,
                    "Shannon Entropy": entropy,
                    "Interpretation": interpretation,
                }
            ]

            return result_table

        except Exception:
            print("Exception in shannon_entropy:", traceback.format_exc())
            return None


class SamplingOperations:
    """
    Sampling SDK Class:
    Provides static methods to perform:
    1. Sampling and visualization ('submit' action)
    2. Sampling and save preparation ('save' action)
    """

    @staticmethod
    def execute_sampling(parameters={}, df=pd.DataFrame()):
        """
        Perform sampling operation based on provided parameters.

        Args:
            parameters (dict): Contains keys like 'column_name', 'method', 'action', etc.
            df (pd.DataFrame): Input dataset.

        Returns:
            dict: Contains figures, tables, meta info, sampled datasets
        """
        try:
            print("inside execute_sampling")
            column_name = parameters.get("column_name")
            method = parameters.get("method", "").lower()
            action = parameters.get("action", "").lower()
            ratio = parameters.get("ratio", 0.5)
            cluster = parameters.get("cluster", None)
            categorical_threshold = parameters.get("categorical_threshold", 0)
            print(parameters.get("method"))

            if df.dtypes[column_name] == object and isinstance(
                df[column_name].values[0], bytes
            ):
                df[column_name] = df[column_name].apply(lambda x: x.decode("utf-8"))

            col_type = get_column_type(df, column_name, categorical_threshold)

            if method == "random":
                figs, train, test, res_table = sample(
                    df, column_name, ratio, 1, None, action, col_type
                )
            elif method == "stratified":
                figs, train, test, res_table = sample(
                    df, column_name, ratio, 2, None, action, col_type
                )
            elif method == "cluster":
                figs, train, res_table = sample(
                    df, column_name, cluster, 3, None, action, col_type
                )
                test = None
            elif method == "systematic":
                figs, train, res_table = sample(
                    df, column_name, ratio, 4, None, action, col_type
                )
                test = None
            else:
                raise ValueError("Invalid sampling method provided.")

            output = {}

            if action == "submit":
                res_table.fillna(0, inplace=True)
                res_table_json = json.loads(res_table.to_json(orient="records"))

                # Determine chart titles based on method
                if method == "random" or method == "stratified":
                    chart_titles = [
                        f"Before Sampling: {column_name}",
                        f"After Sampling Train: {column_name}",
                        f"After Sampling Test: {column_name}",
                    ]
                elif method == "cluster" or method == "systematic":
                    chart_titles = [
                        f"Before Sampling: {column_name}",
                        f"After Sampling: {column_name}",
                    ]
                else:
                    chart_titles = [
                        f"Chart {i+1}: {column_name}" for i in range(len(figs))
                    ]

                charts = []
                for i, fig in enumerate(figs):
                    title = (
                        chart_titles[i]
                        if i < len(chart_titles)
                        else f"Chart {i+1}: {column_name}"
                    )
                    charts.append({"figure": fig, "title": title})

                output["charts"] = charts
                output["result_table"] = res_table_json
                output["parameters"] = parameters
                print(f"[DEBUG] Number of figures: {len(figs)}")
                print(f"[DEBUG] Number of chart titles: {len(chart_titles)}")
                print(f"[DEBUG] Table Preview: {res_table_json[:2]}")

            elif action == "save":
                displayname = parameters.get("new_table_name", "sampled_data")
                meta_info = {}

                if method in ["cluster", "systematic"]:
                    if method == "systematic":
                        ratio_int = int(ratio * 100)
                        if ratio_int == 50:
                            ratio_int = f"{ratio_int}_00"
                        prefix = f"_sys_{ratio_int}"
                    else:
                        prefix = "_clus"
                    meta_info = {
                        "prefix": [prefix],
                        "newdisplayname": [displayname + prefix],
                        "stages": [len(train)],
                        "dtypes": [train.dtypes.astype(str).to_dict()],
                        "new_table_names": [displayname + prefix],
                    }

                else:
                    splitt = ratio * 100
                    train_ = int(splitt)
                    test_ = int(100 - splitt)
                    prefix1 = f"_ran_{train_}"
                    prefix2 = f"_ran_{test_}"

                    meta_info = {
                        "prefix": [prefix1, prefix2],
                        "newdisplayname": [
                            displayname + prefix1,
                            displayname + prefix2,
                        ],
                        "stages": [len(train), len(test)],
                        "dtypes": [
                            train.dtypes.astype(str).to_dict(),
                            test.dtypes.astype(str).to_dict(),
                        ],
                        "new_table_names": [
                            displayname + prefix1,
                            displayname + prefix2,
                        ],
                    }

                output["metadata"] = meta_info
                output["dataframes"] = {"train": train, "test": test}
                output["parameters"] = parameters

            return output

        except Exception:
            print("Error:\n", traceback.format_exc())
            return None


class ResamplingOperations:
    """
    Resampling SDK Class:
    - Supports undersampling, oversampling, and combined methods.
    - Handles 'submit' and 'save' actions.
    """

    @staticmethod
    def execute_resampling(parameters={}, df=pd.DataFrame()):
        """
        Perform resampling operation based on provided parameters.

        Args:
            parameters (dict): Must contain keys like 'methodology', 'method', 'action', etc.
            df (pd.DataFrame): Input DataFrame.

        Returns:
            dict: Result with resampled data, tables, charts, or metadata depending on action.
        """
        try:
            print("inside execute_resampling")
            table_name = parameters.get("table_name")
            column_name = parameters.get("column_name")
            methodology = parameters.get("methodology", "").lower()
            method = parameters.get("method", "").lower()
            action = parameters.get("action", "").lower()
            categorical_threshold = parameters.get("categorical_threshold", 0)
            new_table_name = parameters.get("new_table_name", "resampled")

            if df.dtypes[column_name] == object and isinstance(
                df[column_name].values[0], bytes
            ):
                df[column_name] = df[column_name].apply(lambda x: x.decode("utf-8"))

            col_type = get_column_type(df, column_name, categorical_threshold)

            rm = None
            sampling_type = None

            if methodology == "undersampling":
                sampling_type = 1
                if method == "random undersampling":
                    rm = 1
                elif method == "neighbour cleaning rule":
                    rm = 6
                elif method == "near miss under":
                    rm = 2
                elif method == "tomek link":
                    rm = 3
                elif method == "edited nearest neighbour":
                    rm = 4
                elif method == "one sided select":
                    rm = 5
                else:
                    raise ValueError("Invalid undersampling method.")

            elif methodology == "oversampling":
                sampling_type = 2
                if method == "random oversampling":
                    rm = 1
                elif method == "smote":
                    rm = 2
                elif method == "borderline-smote":
                    rm = 3
                elif method == "adasyn":
                    rm = 4
                else:
                    raise ValueError("Invalid oversampling method.")

            elif methodology == "combined":
                sampling_type = 3
                if method == "smoteen":
                    rm = 1
                elif method == "smotetomek":
                    rm = 2
                else:
                    raise ValueError("Invalid combined method.")

            else:
                raise ValueError("Invalid methodology.")

            # Perform resampling
            plot_before, plot_after, before_table, after_table, df_resampled = resample(
                df, column_name, sampling_type, rm, None, action, col_type
            )

            result = {}

            if action == "submit":
                before_table_dict = before_table.to_dict("records")
                after_table_dict = after_table.to_dict("records")
                result["tables"] = {
                    "before_table": before_table_dict,
                    "after_table": after_table_dict,
                }

                chart_titles = ["Before Resampling", "After Resampling"]
                figs = [plot_before, plot_after]
                charts = []
                for i, fig in enumerate(figs):
                    charts.append(
                        {"figure": fig, "title": f"{chart_titles[i]}: {column_name}"}
                    )

                result["charts"] = charts
                result["parameters"] = parameters

            elif action == "save":
                stage_name = stage_table_resampling(table_name)
                dtypes = get_schema(df_resampled)
                stages = len(df_resampled)

                meta_info = {
                    "stage_name": stage_name,
                    "stages": stages,
                    "dtypes": dtypes,
                    "isFolder": False,
                    "new_table_name": f"{new_table_name}_resampled",
                }

                result["resampled_data"] = df_resampled
                result["metadata"] = meta_info
                result["parameters"] = parameters

            else:
                raise ValueError("Invalid action. Must be 'submit' or 'save'.")

            return result

        except Exception:
            print(traceback.format_exc())
            return None


class SplittingSDK:
    """
    Splitting SDK:
    Provides static methods for:
    1. Random Splitting
    2. Stratified Splitting
    """

    @staticmethod
    def random_split(parameters={}, df=pd.DataFrame()):
        """
        Perform Random Splitting.

        Args:
            parameters (dict): Must contain 'column_name', 'ratio_1', 'ratio_2', 'userpref', 'categorical_threshold', 'action'.
            df (pd.DataFrame): Input dataset.

        Returns:
            If action == 'submit': returns result_table, figures
            If action == 'save': returns modified DataFrame saved to storage (None here, assuming save_to_s3 handles it)
        """
        try:
            print("inside random_split")
            column_name = parameters.get("column_name")
            ratio1 = parameters.get("ratio_1")
            ratio2 = parameters.get("ratio_2") / (1 - ratio1)
            action = parameters.get("action", "submit")
            categorical_threshold = parameters.get("categorical_threshold", 0)

            col_type = get_column_type(df, column_name, categorical_threshold)

            (
                initial_graph,
                train_graph,
                test_graph,
                validate_graph,
                result_table,
                train_df,
                test_df,
                validate_df,
            ) = sp(df, column_name, ratio1, ratio2, 1, action, col_type)

            if action == "submit":
                result_table.fillna(0, inplace=True)
                tables = result_table.to_dict("records")

                figs = [initial_graph, train_graph, test_graph, validate_graph]
                charts = []
                for fig in figs:
                    # fig = update_chart(fig)
                    fig.show()
                    charts.append(fig)

                return tables, charts

            elif action == "save":
                stages = [len(train_df), len(test_df), len(validate_df)]
                dtypes = [
                    get_schema(train_df),
                    get_schema(test_df),
                    get_schema(validate_df),
                ]
                parameters["stages"] = stages
                parameters["dtypes"] = dtypes

                new_table_name = parameters.get("new_table_name", "split_table")
                prefixs = ["train", "test", "validate"]
                meta_info = {
                    "prefix": prefixs,
                    "newdisplayname": [f"{new_table_name}_{p}" for p in prefixs],
                    "stages": stages,
                    "dtypes": dtypes,
                    "new_table_names": [f"{new_table_name}_{p}" for p in prefixs],
                }

                return {
                    "metadata": meta_info,
                    "dataframes": {
                        "train": train_df,
                        "test": test_df,
                        "validate": validate_df,
                    },
                    "parameters": parameters,
                }

        except Exception as ex:
            raise ValueError(ex, traceback.format_exc())

    @staticmethod
    def stratified_split(parameters={}, df=pd.DataFrame()):
        """
        Perform Stratified Splitting.

        Args:
            parameters (dict): Must contain 'column_name', 'ratio_1', 'ratio_2', 'userpref', 'categorical_threshold', 'action'.
            df (pd.DataFrame): Input dataset.

        Returns:
            If action == 'submit': returns result_table, figures
            If action == 'save': returns modified DataFrame saved to storage (None here, assuming save_to_s3 handles it)
        """
        try:
            column_name = parameters.get("column_name")
            ratio1 = parameters.get("ratio_1")
            ratio2 = parameters.get("ratio_2") / (1 - ratio1)
            action = parameters.get("action", "submit")
            categorical_threshold = parameters.get("categorical_threshold", 0)

            col_type = get_column_type(df, column_name, categorical_threshold)

            (
                initial_graph,
                train_graph,
                test_graph,
                validate_graph,
                result_table,
                train_df,
                test_df,
                validate_df,
            ) = sp(df, column_name, ratio1, ratio2, 2, action, col_type)

            if action == "submit":
                result_table.fillna(0, inplace=True)
                tables = result_table.to_dict("records")

                figs = [initial_graph, train_graph, test_graph, validate_graph]
                charts = []
                for fig in figs:
                    fig = update_chart(fig)
                    fig.show()
                    charts.append(fig)

                return tables, charts

            elif action == "save":
                stages = [len(train_df), len(test_df), len(validate_df)]
                dtypes = [
                    get_schema(train_df),
                    get_schema(test_df),
                    get_schema(validate_df),
                ]
                parameters["stages"] = stages
                parameters["dtypes"] = dtypes

                new_table_name = parameters.get("new_table_name", "split_table")
                prefixs = ["train", "test", "validate"]
                dataframes = [train_df, test_df, validate_df]

                for i in range(3):
                    save_to_s3(
                        location=None,
                        df=dataframes[i],
                        dataset=None,
                        new=f"{new_table_name}_{prefixs[i]}",
                    )

                return df

        except Exception as ex:
            raise ValueError(ex, traceback.format_exc())


class SamplingTimeSeries:

    @staticmethod
    def time_series(
        parameters={}, df=pd.DataFrame(), color=["#FF5733", "#33FF57", "#3357FF"]
    ):
        """
        Apply Time Series Sampling on the provided DataFrame.

        Args:
            parameters (dict): Parameters for sampling including 'time_column', 'start_date', 'end_date', 'target', 'action', etc.
            df (pd.DataFrame): Input DataFrame.
            color (list): List of colors for plotting.

        Returns:
            tuple or dict: Returns tables and graphs if action is 'submit', otherwise returns saved DataFrames info.
        """
        try:
            action = parameters.get("action", "submit")
            time_column = parameters.get("time_column")
            start_date = parameters.get("start_date")
            end_date = parameters.get("end_date")
            target = parameters.get("target")

            # Parse time range
            pattern = r"\d{4} Q[1-4]"
            if re.match(pattern, str(df[time_column].iloc[0])):
                from_ts = convert_ddmmyyyy_to_q(start_date)
                to_ts = convert_ddmmyyyy_to_q(end_date)
            else:
                if re.match(pattern, start_date):
                    from_ts = convert_from_yyyy_q(start_date)
                    to_ts = convert_from_yyyy_q(end_date)
                else:
                    from_ts = parser.parse(start_date, dayfirst=True).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    to_ts = parser.parse(end_date, dayfirst=True).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

            # Decode byte strings
            if df.dtypes[time_column] == object and isinstance(
                df[time_column].iloc[0], bytes
            ):
                df[time_column] = df[time_column].apply(lambda x: x.decode("utf-8"))

            # === SUBMIT ACTION ===
            if action == "submit":
                chart_ids = parameters.get("chart_ids", [])
                graphs, res_table1, res_table2, summary_insample, summary_outsample = (
                    sampleTSsubmit(df, time_column, from_ts, to_ts, target)
                )

                tables_df = []
                graph_list = []

                for i, fig in enumerate(graphs):
                    chart_title = f"{samplingts_chart_title[i]}: {target}"
                    fig, fig2 = update_chart3(
                        fig, static_chart=False, chart_title=chart_title
                    )

                    # Show preview in notebooks
                    if fig2:
                        try:
                            from PIL import Image
                            from io import BytesIO

                            Image.open(BytesIO(fig2)).show()
                        except Exception as img_err:
                            print(f"[⚠️ Warning] Unable to preview image: {img_err}")

                    graph_list.append(fig)

                # Prepare tables
                tables_df = []
                for tbl in [
                    res_table1,
                    res_table2,
                    summary_insample,
                    summary_outsample,
                ]:
                    df_tbl = pd.DataFrame(tbl)
                    df_tbl.fillna(0, inplace=True)
                    tables_df.append(df_tbl)

                return tables_df, graph_list

            # === SAVE ACTION ===
            elif action == "save":
                insample_df, outsample_df = SamplingTimeseriesSave(
                    df, time_column, from_ts, to_ts
                )

                stages = [len(insample_df), len(outsample_df)]
                schema = [get_schema(insample_df), get_schema(outsample_df)]

                return {
                    "dataframes": {"insample": insample_df, "outsample": outsample_df},
                    "stages": stages,
                    "schema": schema,
                }

        except Exception as ex:
            raise ValueError(str(ex), traceback.format_exc())
