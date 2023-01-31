import yaml
from modeling.siamese_transformer.trainer import train_model as train_transformer_model
from modeling.siamese_bilstm.trainer import train_model as train_bi_lstm_model
from modeling.siamese_simple_bilstm.trainer import train_model as train_simple_bilstm_model
from modeling.siamese_simple_transformer.trainer import train_model as train_simple_transformer_model
from modeling.siamese_simple_distil_bert.trainer import train_model as train_simple_distil_bert_model
import sys
sys.path.append('.')


params_file = sys.argv[1]
model_name = sys.argv[2]

with open(params_file, 'r') as f:
    params = yaml.safe_load(f)

if "experiment_name" in params:
    experiment_name = params["experiment_name"]
else:
    experiment_name = input("==== PLEASE ENTER A NAME FOR EXPERIMENT ====\n")


if model_name == "siamese_simple_bilstm":
    train_file = sys.argv[3]
    test_file = sys.argv[4]
    tokenizer_file = sys.argv[5]
    slug_tokenizer_file = sys.argv[6]
    city_tokenizer_file = sys.argv[7]

    print(f"Training {model_name} model...")
    models_params = params[model_name]

    model = train_simple_bilstm_model(
        train_file=train_file,
        test_file=test_file,
        tokenizer_model_filename=tokenizer_file,
        slug_tokenizer_file=slug_tokenizer_file,
        city_tokenizer_file=city_tokenizer_file,
        models_params=models_params,
        experiment_name=experiment_name
    )
elif model_name == "siamese_simple_transformer":
    train_file = sys.argv[3]
    test_file = sys.argv[4]
    tokenizer_file = sys.argv[5]
    slug_tokenizer_file = sys.argv[6]
    city_tokenizer_file = sys.argv[7]

    print(f"Training {model_name} model...")
    models_params = params[model_name]

    model = train_simple_transformer_model(
        train_file=train_file,
        test_file=test_file,
        tokenizer_model_filename=tokenizer_file,
        models_params=models_params,
        experiment_name=experiment_name
    )
elif model_name == "siamese_simple_distil_bert":
    train_file = sys.argv[3]
    test_file = sys.argv[4]
    
    print(f"Training {model_name} model...")
    models_params = params[model_name]

    model = train_simple_distil_bert_model(
        train_file=train_file,
        test_file=test_file,
        models_params=models_params,
        experiment_name=experiment_name
    )
elif model_name == "siamese_transformer":
    train_file = sys.argv[3]
    test_file = sys.argv[4]
    tokenizer_file = sys.argv[5]
    slug_tokenizer_file = sys.argv[6]
    city_tokenizer_file = sys.argv[7]
    neighbor_tokenizer_file = sys.argv[8]

    print(f"Training {model_name} model...")
    models_params = params[model_name]

    model = train_transformer_model(
        train_file=train_file,
        test_file=test_file,
        tokenizer_model_filename=tokenizer_file,
        slug_tokenizer_model_filename=slug_tokenizer_file,
        city_tokenizer_model_filename=city_tokenizer_file,
        neighbor_tokenizer_model_filename=neighbor_tokenizer_file,
        models_params=models_params
    )
