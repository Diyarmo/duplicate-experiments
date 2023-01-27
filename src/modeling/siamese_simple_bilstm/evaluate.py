
from modeling.siamese_bilstm.model import DuplicateSiameseBiLSTM
from modeling.siamese_bilstm.data import DuplicateDataLoader
import torch
from tqdm import tqdm
from typing import Tuple
import numpy as np
import gc

def get_bilstm_model_predictions(model, data_loader, device):
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader.val_dataloader()):
            post1_x, post2_x, stats, y_duplicate = batch

            post1_x = [x.to(device) for x in post1_x]
            post2_x = [x.to(device) for x in post2_x]
            stats = stats.to(device)
            y_duplicate = y_duplicate.to(device)

            duplicate_score = model.forward(post1_x=post1_x, post2_x=post2_x, stats=stats)
            
            predictions += list(duplicate_score.cpu())
            labels += list(y_duplicate.cpu())    
            gc.collect()
            torch.cuda.empty_cache()

    predictions = np.array(predictions)
    labels =  np.array(labels)

    return predictions, labels
    


def get_bilstm_model_and_dataloader(
    model_file,
    tokenizer_file,
    slug_encoder_filename,
    city_encoder_filename,
    neighbor_encoder_filename,
    test_path,
    device
    ) -> Tuple[DuplicateSiameseBiLSTM, DuplicateDataLoader]:

    data_loader = DuplicateDataLoader(
        tokenizer_file = tokenizer_file,
        slug_tokenizer_file = slug_encoder_filename,
        city_tokenizer_file = city_encoder_filename,
        neighbor_tokenizer_file =  neighbor_encoder_filename,
        batch_size=256,
        text_max_length=None,
        train_file=None,
        test_file=test_path, 
        test_sample_size=1000000)

    model = DuplicateSiameseBiLSTM.load_from_checkpoint(
            checkpoint_path=model_file,
        ).to(device)
    model.eval()

    return model, data_loader
