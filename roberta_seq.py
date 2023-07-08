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

if __name__ == '__main__':

    save_path, = sys.argv[1:]
    logging_dir = os.path.join(save_path, 'logs/')
    prediction_output_file = os.path.join(save_path, 'predictions.csv')
    do_train = True

    args = get_base_hf_args(
        output_dir=save_path,
        logging_steps=50,
        logging_dir = logging_dir,
        lr=5e-5,
        train_batch_size=24,
        save_strategy='no',
    )

    model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=2,)
    # model = RobertaForTokenClassification.from_pretrained(save_path, num_labels=2, type_vocab_size=2)

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)

    data_collator = DataCollatorForTokenClassification(tokenizer, max_length=128, )

    ds = get_tokenized_ds('liyucheng/vua20', tokenizer, tokenize_func='tagging', \
        tokenize_cols=['tokens'], tagging_cols={'is_target':0, 'labels':-100}, \
        batched=False, name = 'combined')

    # ds = ds.rename_column('is_target', 'token_type_ids')

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[TensorBoardCallback()],
        compute_metrics=tagging_eval_for_trainer
    )

    if do_train:
        trainer.train()
        trainer.save_model()

    result = trainer.evaluate()
    print(result)

    # ds_test = ds['test'].remove_columns(['labels'])

    # pred_out = trainer.predict(ds_test)
    # write_predict_to_file(pred_out, ds['test']['tokens'], out_file=prediction_output_file)