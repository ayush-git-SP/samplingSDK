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
