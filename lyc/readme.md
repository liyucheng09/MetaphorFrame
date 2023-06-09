# pipeline

- `BertWhitening`

# Model

- `SentenceEmbeddingModel`
- `SimCSE`

# utils

- `get_model`
- `get_rokenizer`
- `get_model_output`
- For `bert_whitening.py`
  - `compute_kernel_bias`
  - `transform_and_normalize`
  - `save_kernel_and_bias`
- `vector_l2_normlize`
- `get_optimizer_and_schedule`
- `BaiduTranslator`
- `get_vectors`
- class `lemmatizer`


# Visualize

- `plotDimensionReduction`
  - `PCA`
  - `tSNE`

# data

- `get_tokenized_ds`
- `get_dataloader`
- `wrap_sentences_to_ds`
- `SentencePairDataset`
- `SimCSEDataSet`
- `processor`
  - `get_true_length`
  - `combine`
  - `_tokenize_and_alingn_labels`
- `get_hf_ds_scripts_path`
- `get_hf_metrics_scripts_path`
- `SentenceDataset`
