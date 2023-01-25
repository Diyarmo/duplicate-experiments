import sys
from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_categorical_feature(values: List, save_path: str):
    label_encoder = LabelEncoder()
    label_encoder.fit(values)
    np.save(save_path, label_encoder.classes_, allow_pickle=True)


def get_encoding_data(encoding_train_file, encoding_val_file, encode_columns):
    values_to_encode_train = pd.read_parquet(encoding_train_file, columns=encode_columns)
    values_to_encode_val = pd.read_parquet(encoding_val_file, columns=encode_columns)
    values_to_encode = pd.concat([values_to_encode_train, values_to_encode_val], axis=0)
    values_to_encode = pd.concat([values_to_encode[col] for col in encode_columns], axis=0).values
    return values_to_encode


save_path = "../../storage/models/categorical_encodings/{}_encoding.npy"
print(save_path.format(sys.argv[4]))
encode_categorical_feature(values=get_encoding_data(sys.argv[1], 
                                                    sys.argv[2], 
                                                    sys.argv[3].split(",")),
                                                     save_path=save_path.format(sys.argv[4]))
print("Done!")

# data_file_train = sys.argv[1]
# data_file_val = sys.argv[2]
# features_to_encode = sys.argv[3].split(",")
# print(features_to_encode)
# name = sys.argv[4]

# save_path = f"storage/models/categorical_encodings/{name}_encoding.npy"

# values_to_encode_train = pd.read_parquet(data_file_train, columns=features_to_encode)
# values_to_encode_test = pd.read_parquet(data_file_val, columns=features_to_encode)
# values_to_encode = pd.concat([values_to_encode_train, values_to_encode_test], axis=0)
# values_to_encode = pd.concat([values_to_encode[col] for col in features_to_encode], axis=0).values

# print(f"Encoding {features_to_encode}...")
# encode_categorical_feature(values=values_to_encode, save_path=save_path)
# print("Done!")




# def encode_categorical_feature(values: List, save_path: str):
#     label_encoder = LabelEncoder()
#     label_encoder.fit(values)
#     np.save(save_path, label_encoder.classes_, allow_pickle=True)


# data_file = sys.argv[1]
# feature_to_encode = sys.argv[2]
# name = sys.argv[3]

# save_path = f"storage/models/categorical_encodings/{name}_encoding.npy"

# encode_df = pd.read_csv(data_file)
# values_to_encode = encode_df[feature_to_encode].values

# print(f"Encoding {feature_to_encode}...")
# encode_categorical_feature(values=values_to_encode, save_path=save_path)
# print("Done!")
