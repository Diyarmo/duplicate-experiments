import sys
sys.path.append('.')

from ceph_interface import CephConfig, CephInterface
import os
from glob import glob
import pandas as pd
import json
from datetime import timedelta, datetime

queue_name = os.environ["QUEUE_NAME"]

bucket_name = "automation-pipeline"
base_bucket = "deep_duplicates/"

base_raw_data_path = "../../storage/data/raw/"
base_prepared_data_path = "../../storage/data/prepared/"

DATASET_SHIFT_DAYS = 8


def get_datasets_s3_keys():
    execution_date = os.environ["execution_date"]
    execution_date = datetime.strptime(execution_date[:10], '%Y-%m-%d')
    train_dataset_days = int(os.environ['train_dataset_days'])
    val_dataset_days = int(os.environ['val_dataset_days'])

    train_shift_days = train_dataset_days + val_dataset_days + DATASET_SHIFT_DAYS
    train_start_date = execution_date - timedelta(days=train_shift_days)
    train_end_date = train_start_date + timedelta(days=train_dataset_days)

    val_start_date = train_end_date + timedelta(days=DATASET_SHIFT_DAYS)
    val_end_date = val_start_date + timedelta(days=val_dataset_days)

    train_start_date = str(train_start_date.date())
    train_end_date = str(train_end_date.date())
    val_start_date = str(val_start_date.date())
    val_end_date = str(val_end_date.date())

    train_dataset_key = f"train_{train_start_date}_{train_end_date}"
    val_dataset_key = f"val_{val_start_date}_{val_end_date}"
    print("train_dataset_key : ", train_dataset_key , "val_dataset_key : ", val_dataset_key)
    return train_dataset_key, val_dataset_key

def _get_key_from_json(row, key):
    res = row.copy()
    for sub_key in key.split("."):
        res = res[sub_key]
    return res

def _remove_same_tokens(train_dataset, val_dataset):
    train_tokens = set(train_dataset['post1_token'].unique())
    val_dataset = val_dataset[val_dataset['post1_token'].apply(lambda x: x not in train_tokens)]

    return train_dataset, val_dataset

def get_ceph_interface():

    ceph_config = CephConfig(
        access_key=os.environ['AWS_ACCESS_KEY_ID'],
        secret_key=os.environ['AWS_SECRET_ACCESS_KEY']
    )

    ceph = CephInterface(ceph_config)

    return ceph

def download_datasets(ceph):
    print("Start Downloading")
    train_dataset_key, val_dataset_key = get_datasets_s3_keys()

    ceph.download_bucket(
        bucket_name=bucket_name,
        dir_name=base_bucket+f"{queue_name}/{train_dataset_key}", 
        local_base_path=base_raw_data_path + "train/"
        )
    
    ceph.download_bucket(
        bucket_name=bucket_name,
        dir_name=base_bucket+f"{queue_name}/{val_dataset_key}", 
        local_base_path=base_raw_data_path + "val/"
        )
    
    ceph.download_to_local(
        bucket_name=bucket_name, 
        ceph_file_key="categories.csv",
        local_file_path=base_prepared_data_path + "categories.csv"
        )
    print("Downloding done")

def read_datasets():
    train_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in glob(base_raw_data_path + "train/*.parquet")
    ).reset_index(drop=True)

    val_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in glob(base_raw_data_path + "val/*.parquet")
    ).reset_index(drop=True)

    return train_df, val_df

if __name__ == "__main__":
    
    ceph = get_ceph_interface()
    download_datasets(ceph)
    train_df, val_df = read_datasets()
    print("Download Finished")
    # train_df, val_df = _remove_unseen_samples(train_df, val_df, ['category', "location.city"])
    print("Removed unseen sample")
    train_df, val_df = _remove_same_tokens(train_df, val_df)
    print("Removed same tokens")

    train_df.to_parquet(f"{base_prepared_data_path}{queue_name}_train.parquet")
    val_df.to_parquet(f"{base_prepared_data_path}{queue_name}_val.parquet")
