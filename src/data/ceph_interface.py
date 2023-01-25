import io
import time

import boto3
import pandas as pd
import pickle



class CephConfig(object):
    def __init__(self, access_key, secret_key):
        self.access_key = access_key
        self.secret_key = secret_key


class CephInterface(object):
    def __init__(self, config):
        self.config = config
        self.s3 = boto3.resource(service_name='s3',
                                 aws_access_key_id=self.config.access_key,
                                 aws_secret_access_key=self.config.secret_key,
                                 endpoint_url='https://s3.thr2.sotoon.ir')

    def delete_directory(self, bucket_name: str, dir_name: str):
        bucket = self.s3.Bucket(bucket_name)
        bucket.objects.filter(Prefix=dir_name).delete()

    def duplicate_dir(self, bucket_name, dir_name: str, new_dir: str):
        bucket = self.s3.Bucket(bucket_name)
        bucket_keys = [obj.key for obj in bucket.objects.filter(Prefix=dir_name)]
        for obj in bucket_keys:
            old_source = {'Bucket': bucket_name,
                          'Key': obj}
            new_key = obj.replace(dir_name, new_dir, 1)
            new_obj = bucket.Object(new_key)
            new_obj.copy(old_source)
        bucket_keys_copied = [obj.key for obj in bucket.objects.filter(Prefix=new_dir)]
        return bucket_keys_copied

    def read_data(self, bucket_name, key_name):
        try:
            key = self.s3.Object(bucket_name, key_name)
            key_content = io.BytesIO(key.get()['Body'].read())
        except Exception as e:
            print(f"failed reading from ceph: {key_name} Error: {str(e)}", flush=True)
            time.sleep(30)
            print(f"start second time reading from ceph: {key_name}", flush=True)
            key = self.s3.Object(bucket_name, key_name)
            key_content = io.BytesIO(key.get()['Body'].read())
        return key_content

    def read_parquet_frame(self, bucket_name, key_name, columns=None):
        key_content = self.read_data(bucket_name, key_name)
        data = pd.read_parquet(key_content, columns=columns)
        key_content.close()
        return data

    def upload_data_as_parquet(self, data_frame, bucket_name, key_name):
        buffer = io.BytesIO()
        data_frame.to_parquet(buffer, index=False, allow_truncated_timestamps=True)
        data = buffer.getvalue()
        self.s3.Object(bucket_name, key_name).put(Body=data)

    def download_to_local(self, bucket_name, local_file_path, ceph_file_key):
        bucket = self.s3.Bucket(bucket_name)
        bucket.download_file(ceph_file_key, local_file_path)

    def get_list_of_parquet_keys(self, bucket_name, dir_name: str):
        bucket = self.s3.Bucket(bucket_name)
        bucket_keys = [obj.key for obj in bucket.objects.filter(Prefix=dir_name) if obj.key.endswith("parquet")]
        return bucket_keys
    
    def download_bucket(self, bucket_name, dir_name, local_base_path):
        bucket_keys = self.get_list_of_files(bucket_name, dir_name)
        for key in bucket_keys:
            local_name = key.split("/")[-1]
            self.download_to_local(bucket_name, f"{local_base_path}/{local_name}", key)
        

    def get_list_of_files(self, bucket_name: str, dir_name: str):
        bucket = self.s3.Bucket(bucket_name)
        bucket_keys = [obj.key for obj in bucket.objects.filter(Prefix=dir_name)]
        return bucket_keys

    def read_parquet_dir(self, bucket_name, ceph_dir, columns=None):
        parquet_keys = self.get_list_of_parquet_keys(bucket_name=bucket_name, dir_name=ceph_dir)
        print(parquet_keys)
        if len(parquet_keys) != 1:
            raise ValueError(f"We except exactly one parquet file in {ceph_dir}")
        return self.read_parquet_frame(bucket_name=bucket_name, key_name=parquet_keys[0], columns=columns)

    def read_pickle(self, bucket_name, key_name):
        return pickle.load(self.read_data(bucket_name=bucket_name, key_name=key_name))

    def upload_data_as_pickle(self, data_object, bucket_name, key_name):
        self.s3.Object(bucket_name, key_name).put(Body=pickle.dumps(data_object))

    def upload_from_local(self, local_file_path, bucket_name, key_name):
        object = self.s3.Object(bucket_name, key_name)
        result = object.put(Body=open(local_file_path, 'rb'))

        if result.get('ResponseMetadata').get('HTTPStatusCode') == 200:
            print('File Uploaded Successfully')
        else:
            print('File Not Uploaded')