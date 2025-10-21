import pandas as pd
import numpy as np
from pandas import DataFrame

PM100_cols = ["cpu_power_consumption", "eligible_time", "job_id", "num_cores_req",
                  "num_nodes_req","node_power_consumption",
                  "mem_req", "priority", "start_time",
                  "submit_time", "time_limit", "user_id"]

def adapt_pm100_features(df: DataFrame):
    """Helper function to convert non-numerical values of PM100 data."""

    df["cpu_power_consumption"] = df["cpu_power_consumption"].apply(len)
    df.rename(columns={"cpu_power_consumption": "cpus_allocated"}, inplace=True)
    df["nodes_allocated"] = df["node_power_consumption"].apply(len)

    time_cols = ["eligible_time", "start_time", "submit_time"]
    for col in time_cols:
        df[col] = pd.to_datetime(df[col]).astype(int) / 3.6e12

    # Mean node power consumption for jobs that used multiple nodes
    df["node_power_consumption"] = df["node_power_consumption"].apply(lambda x: np.mean(x))
    return df

def read_pm100(file_path: str, columns_to_read: list[str]):
    """Reads jobs from PM100 parquet file for given columns."""

    data_frame = pd.read_parquet(file_path, columns=columns_to_read)
    return adapt_pm100_features(data_frame)