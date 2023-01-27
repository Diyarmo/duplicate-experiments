from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sunburst_evaluation import create_and_save_sunburst_plot
from modeling.siamese_simple_bilstm.evaluate import get_simple_bilstm_model_and_dataloader, get_simple_bilstm_model_predictions
from modeling.siamese_transformer.evaluate import get_transformer_model_and_dataloader, get_transformer_model_predictions
from modeling.siamese_bilstm.evaluate import get_bilstm_model_and_dataloader, get_bilstm_model_predictions
from tqdm import tqdm
import numpy as np
import torch
import json
import yaml
import sys
import os
sys.path.append('.')

queue_name = os.environ["QUEUE_NAME"]


def calculate_roc(row):
    mean = row['is_duplicate'].mean()
    try:
        auc = roc_auc_score(
            y_true=row['is_duplicate'], y_score=row['prediction'])
    except:
        auc = np.nan
    return auc, mean


model_name = sys.argv[1]
test_path = sys.argv[2]
tokenizer_file = sys.argv[3]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if model_name == "siamese_simple_bilstm":

    model_path = sys.argv[4]

    model, data_loader = get_simple_bilstm_model_and_dataloader(
        model_file=model_path,
        tokenizer_file=tokenizer_file,
        test_path=test_path,
    )

    predictions = get_simple_bilstm_model_predictions(model, data_loader)
elif model_name == "siamese_bilstm":
    slug_tokenizer_file = sys.argv[4]
    city_tokenizer_file = sys.argv[5]
    neighbor_tokenizer_file = sys.argv[6]
    model_path = sys.argv[7]

    model, data_loader = get_bilstm_model_and_dataloader(
        model_file=model_path,
        tokenizer_file=tokenizer_file,
        slug_encoder_filename=slug_tokenizer_file,
        city_encoder_filename=city_tokenizer_file,
        neighbor_encoder_filename=neighbor_tokenizer_file,
        test_path=test_path,
        device=device
    )

    predictions, labels = get_bilstm_model_predictions(
        model, data_loader, device)
elif model_name == "siamese_transformer":
    slug_tokenizer_file = sys.argv[4]
    city_tokenizer_file = sys.argv[5]
    neighbor_tokenizer_file = sys.argv[6]
    model_path = sys.argv[7]
    model, data_loader = get_transformer_model_and_dataloader(
        model_file=model_path,
        tokenizer_file=tokenizer_file,
        slug_encoder_filename=slug_tokenizer_file,
        city_encoder_filename=city_tokenizer_file,
        neighbor_encoder_filename=neighbor_tokenizer_file,
        test_path=test_path,
        device=device
    )

    predictions, labels = get_transformer_model_predictions(
        model, data_loader, device)


test_data = data_loader.dataset.data
test_data['prediction'] = predictions
test_data.to_parquet(
    f"../../storage/data/prepared/{model_name}_{queue_name}_val_with_preds.parquet")

create_and_save_sunburst_plot(
    test_data, f"duplicate_{model_name}_{queue_name}_auc_per_cat")

auc = roc_auc_score(y_true=test_data['is_duplicate'], y_score=test_data['prediction'])

with open(f"../../logs/{model_name}_{queue_name}_scores_file.json", "w") as fd:
    json.dump({'auc': auc}, fd)
