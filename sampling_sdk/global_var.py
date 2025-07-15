"""
Global variables and constants used across the sampling SDK.
"""

PLOT_BGCOLOR = "rgba(0,0,0,0)"
TIME_ZONE = "Asia/Kolkata"
OPEN_SANS = "Open Sans"

date_ = [
    "timestamp with time zone",
    "timestamp without time zone",
    "timestamp",
    "datetime",
]

string_ = [
    "character varying",
    "varchar",
    "character",
    "text",
    "blob",
    "enum",
    "binary",
    "varbinary",
]

int_ = [
    "smallint",
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
]

dtype_conv = {
    "object": "TEXT",
    "category": "TEXT",
    "int8": "BIGINT",
    "int32": "BIGINT",
    "int64": "BIGINT",
    "int16": "BIGINT",
    "float64": "float",
    "float32": "float",
    "float16": "float",
    "float8": "float",
    "bool": "bool",
    "datetime64[ns]": "timestamp with time zone",
}


INT = [
    "smallint",
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
    "Int8",
    "Int16",
    "Int64",
    "Float64",
    "Float8",
    "Float16",
    "Float32",
    "Int32",
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


schema_map = {
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
    "datetime64": "date",
    "datetime64[ns]": "date",
    "category": "text",
    "Int8": "int",
    "Int16": "int",
    "Int64": "int",
    "Float64": "float",
    "Float8": "float",
    "Float16": "float",
    "Float32": "float",
    "Int32": "int",
}


def schema(dtype):
    # Convert dtype to string first, then map to schema type, default to 'text'
    dtype_str = str(dtype)
    # Sometimes pandas dtypes include extra info, so just look for key substrings:
    for key in schema_map:
        if key.lower() in dtype_str.lower():
            return schema_map[key]
    return "text"


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
