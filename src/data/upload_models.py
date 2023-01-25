import sys
sys.path.append('.')

from ceph_interface import CephConfig, CephInterface
import json
import os
from pathlib import Path
from datetime import datetime
model_name = sys.argv[1]
queue_name = sys.argv[2]

execution_date = os.environ["execution_date"]
training_datetime = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
models_s3_path = f"{queue_name}/{training_datetime}/"
bucket_name = "production-models"
base_bucket = "deep-duplicate/" + models_s3_path

base_models_path = "../../storage/models/"

ceph_config = CephConfig(
    access_key=os.environ['AWS_ACCESS_KEY_ID'],
    secret_key=os.environ['AWS_SECRET_ACCESS_KEY']
)

ceph = CephInterface(ceph_config)
if model_name == 'bilstm':
    files_to_upload = [
        f"siamese-bilstm-model-{queue_name}.ckpt",
        f"tokenizers/tokenizer_{queue_name}.json",
        f"categorical_encodings/{queue_name}_cities_encoding.npy",
        f"categorical_encodings/{queue_name}_categories_encoding.npy",
        f"categorical_encodings/{queue_name}_neighbors_encoding.npy"
    ]
    
for file_path in files_to_upload:
    print(f"Uploading {file_path} ...")

    ceph.upload_from_local(
        bucket_name=bucket_name,
        key_name=base_bucket + Path(file_path).name, 
        local_file_path=base_models_path + file_path
    )
    
with open("../../README.md", "a") as f:
    f.write(f"\n* Model Files Stored in: `{base_bucket}` \n\n")
