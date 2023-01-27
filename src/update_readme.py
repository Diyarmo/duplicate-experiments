import json
import sys
from typing import Dict
import yaml
import os

train_params_path = sys.argv[1]
basic_params_path = sys.argv[2]
model_name = sys.argv[3]
overall_metrics_filename = sys.argv[4]
queue_name = os.environ["QUEUE_NAME"]


with open(basic_params_path, 'r') as f:
    basic_params = yaml.safe_load(f)
with open(train_params_path, 'r') as f:
    train_params = yaml.safe_load(f)
with open(overall_metrics_filename, "r") as fd:
    overall_metrics = json.load(fd)

auc = overall_metrics['auc']
model_params: Dict = train_params[model_name]
tokenizer_params = basic_params["tokenizer"]
experiment_specs = "\n\t\t".join(model_params["experiment_specs"])
del model_params["experiment_specs"]

detailed_evaluation_msg = f"* Open this [link](logs/{model_name}_{queue_name}_auc_per_cat.html) for a detailed evaluation on dataset."
sunburst_plot_msg = f"![Prob Density Comparison](logs/{model_name}_{queue_name}_prob_density_by_label.jpg)"

with open(f"../../README.md", "w") as f:
    f.write("# Duplicate Detection Experiments Project")
    f.write(f"\n ## Model Name: {model_name}\n")
    f.write(f"\n {detailed_evaluation_msg} \n")
    f.write(f"\n {sunburst_plot_msg} \n")
    f.write(f"\n Experiment Specs: \n\n\t\t{experiment_specs}\n")
    f.write(f"\n Model Params: `{json.dumps(model_params, indent=2)}`\n")
    f.write(f"\n Tokenizer Params: {tokenizer_params}\n")
