from dateutil import parser
import pandas as pd
from pandas import DataFrame

FDATA_cols = ["cnumat", "schedsdt", "jid", "cnumr",
                  "nnumr","nnuma",
                  "mszl", "pri", "sdt",
                  "adt", "elpl", "usr", "avgpcon"]

def rename_columns(df: DataFrame):
    """Helper function to convert FDATA column's names to concat it with PM100."""

    df_renamed = df.rename(columns={"cnumat": "cpus_allocated",
                       "schedsdt": "eligible_time",
                       "jid": "job_id",
                       "cnumr": "num_cores_req",
                       "nnumr": "num_nodes_req",
                       "nnuma": "nodes_allocated",
                       "mszl": "mem_req",
                       "pri": "priority",
                       "sdt": "start_time",
                       "adt": "submit_time",
                       "elpl": "time_limit",
                       "usr": "user_id",
                       "avgpcon": "node_power_consumption"})
    return df_renamed

def adapt_fdata_features(df):
    """Helper function to convert non-numerical values of FDATA data."""

    df["jid"] = df["jid"].apply(lambda id_str: float(id_str.replace("jid_", "")))
    df["usr"] = df["usr"].apply(lambda usr_str: float(usr_str.replace("usr_", "")))

    time_cols = ["schedsdt", "sdt", "adt"]
    for col in time_cols:
        df[col] = df[col].apply(lambda x: parser.isoparse(x).timestamp())
    return rename_columns(df)

def read_fdata(file_path: str, columns_to_read: list[str]):
    """Reads jobs from FDATA parquet file for given columns."""

    data_frame = pd.read_parquet(file_path, columns=columns_to_read)
    return adapt_fdata_features(data_frame)