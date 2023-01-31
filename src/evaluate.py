from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sunburst_evaluation import create_and_save_sunburst_plot
from modeling.siamese_simple_bilstm.evaluate import get_simple_bilstm_model_and_dataloader, get_simple_bilstm_model_predictions
from modeling.siamese_simple_distil_bert.evaluate import get_simple_distil_bert_model_predictions, get_simple_distil_bert_model_and_dataloader
from modeling.siamese_simple_transformer.evaluate import get_simple_transformer_model_and_dataloader, get_simple_transformer_model_predictions
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
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


def add_hist_to_plot(df, label):
    plt.hist(
        df,
        alpha=0.5,
        bins=20,
        label=label,
        weights=np.ones(len(df)) / len(df))


def create_hist_plot(auc, edited_df, not_edited_df):
    plt.figure(figsize=(12, 8))
    plt.title(
        f"Comparison of Prediction Prob. Density by Label | ROC-AUC = {auc:.4f}")
    add_hist_to_plot(edited_df, "Dupliated Ad")
    add_hist_to_plot(not_edited_df, "Not Duplicated Ad")
    plt.legend()
    plt.xlabel("Prediction Prob.")
    plt.ylabel("Density")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))


def save_prob_density_comparison_plot(df, plot_name):
    auc = roc_auc_score(y_true=df['is_duplicate'],
                        y_score=df['prediction'])

    dup_df = df[df['is_duplicate'].astype(bool)]['prediction']
    not_dup_df = df[~df['is_duplicate'].astype(bool)]['prediction']
    create_hist_plot(auc, dup_df, not_dup_df)

    plt.savefig(f"../../logs/{plot_name}.jpg")


model_name = sys.argv[1]
test_path = sys.argv[2]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if model_name == "siamese_simple_bilstm":
    tokenizer_file = sys.argv[3]
    slug_tokenizer_file = sys.argv[4]
    city_tokenizer_file = sys.argv[5]
    model_path = sys.argv[6]

    model, data_loader = get_simple_bilstm_model_and_dataloader(
        model_file=model_path,
        tokenizer_file=tokenizer_file,
        test_path=test_path,
        slug_tokenizer_file=slug_tokenizer_file,
        city_tokenizer_file=city_tokenizer_file
    )

    predictions = get_simple_bilstm_model_predictions(model, data_loader)

elif model_name == "siamese_simple_distil_bert":
    model_path = sys.argv[3]

    model, data_loader = get_simple_distil_bert_model_and_dataloader(
        model_file=model_path,
        test_path=test_path,
    )
    predictions = get_simple_distil_bert_model_predictions(model, data_loader)

elif model_name == "siamese_simple_transformer":
    tokenizer_file = sys.argv[3]
    model_path = sys.argv[4]

    model, data_loader = get_simple_transformer_model_and_dataloader(
        model_file=model_path,
        tokenizer_file=tokenizer_file,
        test_path=test_path,
    )

    predictions = get_simple_transformer_model_predictions(model, data_loader)


test_data = data_loader.dataset.data
test_data['prediction'] = predictions
test_data.to_parquet(
    f"../../storage/data/prepared/{model_name}_{queue_name}_val_with_preds.parquet")

create_and_save_sunburst_plot(
    test_data, f"{model_name}_{queue_name}_auc_per_cat")

save_prob_density_comparison_plot(
    test_data, f"{model_name}_{queue_name}_prob_density_by_label")

with open(f"../../logs/{model_name}_{queue_name}_scores_file.json", "w") as fd:
    json.dump({
        'auc': roc_auc_score(y_true=test_data['is_duplicate'], y_score=test_data['prediction'])
    }, fd)
