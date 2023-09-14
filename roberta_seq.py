import sys
from transformers import RobertaForTokenClassification, DataCollatorForTokenClassification, Trainer, RobertaTokenizerFast
from transformers.integrations import TensorBoardCallback
import os
import datasets
import torch
import numpy as np
from lyc.train import get_base_hf_args
from lyc.eval import tagging_eval_for_trainer, write_predict_to_file
from lyc.data import get_tokenized_ds
from model import FrameFinder

if __name__ == '__main__':

    save_path, = sys.argv[1:]
    logging_dir = os.path.join(save_path, 'logs/')
    prediction_output_file = os.path.join(save_path, 'predictions.csv')
    do_train = True

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)

    data_collator = DataCollatorForTokenClassification(tokenizer, max_length=256, )

    ds = get_tokenized_ds('liyucheng/FrameNet_v17', tokenizer, tokenize_func='tagging', \
        tokenize_cols=['tokens'], tagging_cols={'frame_tags':-100}, \
        batched=False, name = 'frame_label', max_length=256)

    ds = ds.rename_column('frame_tags', 'labels')
    # ds_train = ds['train']
    ds_train = datasets.concatenate_datasets([ds['train'], ds['validation']])
    ds_test = ds['test']

    # label_list = ds.features['labels'].feature.names
    label_list = ds['train'].features['labels'].feature.names
    # model = RobertaForTokenClassification.from_pretrained(save_path, num_labels=len(label_list),)
    model = FrameFinder.from_pretrained('roberta-base', num_labels=len(label_list),)

    args = get_base_hf_args(
        output_dir=save_path,
        logging_steps=50,
        logging_dir = logging_dir,
        lr=5e-5,
        train_batch_size=64,
        eval_batch_size=32,
        save_strategy='no',
        epochs=15,
        label_names=label_list,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[TensorBoardCallback()],
        # compute_metrics=tagging_eval_for_trainer,
    )

    if do_train:
        trainer.train()
        trainer.save_model()

    # result = trainer.evaluate()
    # print(result)

    # ds_test = ds['test'].remove_columns(['labels'])

    pred_out = trainer.predict(ds_test)
    print(tagging_eval_for_trainer((pred_out.predictions[1], ds_test['labels'])))

    # write_predict_to_file(pred_out, ds['test']['tokens'], out_file=prediction_output_file)