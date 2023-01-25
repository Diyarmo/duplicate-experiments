import sys
sys.path.append('.')
from modeling.siamese_bilstm.trainer import train_model as train_bi_lstm_model
from modeling.siamese_transformer.trainer import train_model as train_transformer_model
import yaml

print(sys.argv)

params_file = sys.argv[1]
model_name = sys.argv[2]

with open(params_file, 'r') as f:
    params = yaml.safe_load(f)

if model_name == "siamese_bilstm":
    train_file = sys.argv[3]
    test_file = sys.argv[4]
    tokenizer_file = sys.argv[5]
    slug_tokenizer_file = sys.argv[6]
    city_tokenizer_file = sys.argv[7]
    neighbor_tokenizer_file = sys.argv[8]

    print(f"Training {model_name} model...")
    models_params = params[model_name]

    model = train_bi_lstm_model(
        train_file=train_file,
        test_file=test_file,
        tokenizer_model_filename=tokenizer_file,
        slug_tokenizer_model_filename=slug_tokenizer_file,
        city_tokenizer_model_filename=city_tokenizer_file,
        neighbor_tokenizer_model_filename=neighbor_tokenizer_file,
        models_params=models_params
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
