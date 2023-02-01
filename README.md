# Duplicate Detection Experiments Project
 ## Model Name: siamese_simple_distil_bert

 * Open this [link](logs/siamese_simple_distil_bert_general_auc_per_cat.html) for a detailed evaluation on dataset. 

 ![Prob Density Comparison](logs/siamese_simple_distil_bert_general_prob_density_by_label.jpg) 

 Experiment Specs: 

		Distil Bert
		Using {title, desc, city, slug} features in encoder
		Using {distil bert pretrained} in encoder
		Using {attention} to aggregate embs in encoder
		Using {subtract, concat, fully-connected} for output

 Model Params: `{
  "text_max_length": 512,
  "output_feature_dim": 296,
  "max_epochs": 3,
  "dropout_rate": 0.2,
  "batch_size": 64,
  "initial_lr": 0.001,
  "accumulate_grad_batches": 4,
  "features_emb_dim": 20
}`

 Tokenizer Params: {'vocab_size': 40000}
