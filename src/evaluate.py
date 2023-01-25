import yaml
import sys
import os
sys.path.append('.')
import json
import torch
import numpy as np
from tqdm import tqdm
from modeling.siamese_bilstm.evaluate import get_bilstm_model_and_dataloader, get_bilstm_model_predictions
from modeling.siamese_transformer.evaluate import get_transformer_model_and_dataloader, get_transformer_model_predictions
from sunburst_evaluation import create_and_save_sunburst_plot
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_auc_score

queue_name = os.environ["QUEUE_NAME"]

def presicion_recall(y_true, y_pred):
    y_pred = np.array(y_pred)
    threshold = 0.75
    print("duplicate count: ", np.unique(np.array(y_pred) >= threshold, return_counts=True))
    cm = confusion_matrix(y_true=y_true, y_pred=(y_pred >= threshold))
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    recall_neg = tn / (tn + fp)
    recall_pos = tp / (tp + fn)
    precision = tp / (tp + fp)
    print('tp: ', tp, 'fp: ', fp)
    auc = roc_auc_score(y_true, y_pred)
    return  precision, recall_neg, recall_pos, threshold, auc


def calculate_roc(row):
    mean = row['is_duplicate'].mean()
    try:
        auc = roc_auc_score(y_true=row['is_duplicate'], y_score=row['prediction'])
    except:
        auc = np.nan
    try:
        precision = presicion_recall(y_true=row['is_duplicate'], y_pred=row['prediction'])[0]
    except:
        precision = []
    return auc, mean, precision


model_name = sys.argv[1]
test_path = sys.argv[2]
tokenizer_file = sys.argv[3]
slug_tokenizer_file = sys.argv[4]
city_tokenizer_file = sys.argv[5]
neighbor_tokenizer_file = sys.argv[6]
params_file = sys.argv[7]
with open(params_file, 'r') as f:
    params = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


if model_name == "siamese_bilstm":
    
    model_path = sys.argv[8]

    model, data_loader = get_bilstm_model_and_dataloader(
        model_file=model_path,
        tokenizer_file=tokenizer_file,
        slug_encoder_filename=slug_tokenizer_file,
        city_encoder_filename=city_tokenizer_file,
        neighbor_encoder_filename=neighbor_tokenizer_file,
        test_path=test_path,
        device=device
            )

    predictions, labels = get_bilstm_model_predictions(model, data_loader, device)
elif model_name == "siamese_transformer":

    model_path = sys.argv[8]
    model, data_loader = get_transformer_model_and_dataloader(
        model_file=model_path,
        tokenizer_file=tokenizer_file,
        slug_encoder_filename=slug_tokenizer_file,
        city_encoder_filename=city_tokenizer_file,
        neighbor_encoder_filename=neighbor_tokenizer_file,
        test_path=test_path,
        device=device
            )

    predictions, labels = get_transformer_model_predictions(model, data_loader, device)


test_data = data_loader.val.data
test_data['prediction'] = predictions
test_data.to_parquet(f"../../storage/data/prepared/{model_name}_{queue_name}_val_with_preds.parquet")

create_and_save_sunburst_plot(test_data, f"duplicate_{model_name}_{queue_name}_auc_per_cat")

groups = test_data.groupby("post1_category")
sliced_df = groups.apply(lambda x: (calculate_roc(x), len(x))).sort_values()
precision, recall_neg, recall, thresholds, auc=presicion_recall(test_data["is_duplicate"], test_data['prediction'])

overall_metrics = {
    "precision": precision,
    "recall_neg": recall_neg,
    "recall": recall,
    "thresholds": thresholds,
    "auc": auc
}

with open(f"../../logs/{model_name}_{queue_name}_scores_file.json", "w") as fd:
    json.dump({'auc': auc}, fd)
with open(f"../../logs/{model_name}_{queue_name}_slug_auc.json", "w") as fd:
    json.dump(sliced_df.to_dict(), fd)
with open(f"../../logs/{model_name}_{queue_name}_plots_file.json", "w") as fd:
    json.dump({
        'precision': str(precision),
        'recall': str(recall),
        'threshold': str(thresholds)},fd)
