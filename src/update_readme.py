import json
import sys
from typing import Dict
import yaml

params_filename = sys.argv[1]
model_name = sys.argv[2]
overall_metrics_filename = sys.argv[3]


with open(params_filename, 'r') as f:
    params = yaml.safe_load(f)

with open(overall_metrics_filename, "r") as fd:
    overall_metrics = json.load(fd)

model_params: Dict = params[model_name]
model_params_str = ", ".join([f"{k} = {v}" for k,v in model_params.items()])
tokenizer_params = params["tokenizer"]

experiment_specs = " & ".join(model_params["experiment_specs"])
del model_params["experiment_specs"]

auc = overall_metrics['auc']
with open(f"./README.md", "a") as fd:
    fd.write(f"{model_name} | {experiment_specs} | {model_params} | {tokenizer_params} | {auc} \n")