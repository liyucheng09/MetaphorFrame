import sys
from transformers import RobertaForTokenClassification, DataCollatorForTokenClassification, Trainer, RobertaTokenizerFast
# from transformers.integrations import TensorBoardCallback
import os
import datasets
import torch
import numpy as np
from lyc.train import get_base_hf_args
from lyc.eval import tagging_eval_for_trainer, write_predict_to_file
from lyc.data import get_tokenized_ds
from model import FrameFinder

def novel_processor(x):
    return {'novel_metaphors': np.where(np.array(x['metaphor_classes']) == 1, x['novel_metaphors'], -100)}

if __name__ == '__main__':

    save_path, = sys.argv[1:]
    logging_dir = os.path.join(save_path, 'logs/')
    prediction_output_file = os.path.join(save_path, 'predictions.csv')
    do_train = True

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)

    data_collator = DataCollatorForTokenClassification(tokenizer, max_length=256, )

    ds = get_tokenized_ds('CreativeLang/ukp_novel_metaphor', tokenizer, tokenize_func='tagging', \
        tokenize_cols=['words'], tagging_cols={'novel_metaphors':-100, 'metaphor_classes': 0}, \
        batched=False, max_length=256, is_split_into_words=True)
    
    ds = ds.filter(lambda x: any(np.array(x['metaphor_classes']) == 1))
    ds = ds.map(novel_processor)

    ds = ds['train'].train_test_split()
    ds = ds.rename_column('metaphor_classes', 'token_type_ids')
    ds = ds.rename_column('novel_metaphors', 'labels')
    ds_train = ds['train']
    print(ds_train[0])
    # ds_train = datasets.concatenate_datasets([ds['train'], ds['validation']])
    ds_test = ds['test']
    # ds_test = ds['test'][:2]
    # ds_test = datasets.Dataset.from_dict(ds_test)

    # label_list = ds.features['labels'].feature.names
    label_list = ds['train'].features['labels'].feature.names
    print(label_list)
    model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels = len(label_list), type_vocab_size = 2, ignore_mismatched_sizes=True)
    model._init_weights(model.roberta.embeddings.token_type_embeddings)
    # model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=len(label_list),)
    # model = FrameFinder.from_pretrained('roberta-base', num_labels=len(label_list),)

    args = get_base_hf_args(
        output_dir=save_path,
        logging_steps=50,
        logging_dir = logging_dir,
        lr=5e-5,
        train_batch_size=64,
        eval_batch_size=32,
        save_strategy='no',
        epochs=7,
        # label_names=label_list,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=tagging_eval_for_trainer,
    )

    if do_train:
        trainer.train()
        trainer.save_model()

    result = trainer.evaluate()
    print(result)

    # ds_test = ds['test'].remove_columns(['labels'])

    # pred_out = trainer.predict(ds_test)
    # print(tagging_eval_for_trainer((pred_out.predictions[1], ds_test['labels'])))

    # write_predict_to_file(pred_out, ds['test']['tokens'], out_file=prediction_output_file)