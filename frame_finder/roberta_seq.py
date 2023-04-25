from lyc.utils import get_model, get_tokenizer
from lyc.data import get_hf_ds_scripts_path, get_tokenized_ds, get_dataloader
import sys
from lyc.train import get_base_hf_args, HfTrainer
from lyc.eval import tagging_eval_for_trainer, write_predict_to_file, \
    eval_with_weights, show_error_instances_id, get_true_label
from transformers import RobertaForTokenClassification, DataCollatorForTokenClassification, Trainer
from transformers.integrations import TensorBoardCallback
import os
import datasets
import torch
import numpy as np

def tokenize_alingn_labels_replace_with_mask_and_add_type_ids(ds, do_mask=False):
    results={}

    target_index = ds['word_index']
    tokens = ds['tokens']
    results['target_word'] = tokens[target_index]
    if do_mask:
        tokens[target_index] = '<mask>'
        ds['tokens'] = tokens

    for k,v in ds.items():
        if k != 'tokens':
            continue
        else:
            out_=tokenizer(v, is_split_into_words=True)
            results.update(out_)

    words_ids = out_.word_ids()
    label_sequence = [0 for i in range(len(words_ids))]
    target_mask = [0 for i in range(len(words_ids))]
    word_idx = words_ids.index(target_index)

    label_sequence[word_idx] = ds['label']
    target_mask[word_idx] = 1

    results['target_mask'] = target_mask
    results['labels'] = label_sequence
    results['tokenized_taregt_word_index'] = word_idx
    results['token_level_label'] = ds['label']
    return results

if __name__ == '__main__':

    model_name, data_dir, = sys.argv[1:]
    save_folder = '/vol/research/nlg/frame_finder/'
    # save_folder = './'
    output_dir = os.path.join(save_folder, 'checkpoints/labeling_no_token_type/')
    logging_dir = os.path.join(save_folder, 'logs/')
    prediction_output_file = os.path.join(output_dir, 'error_instances.csv')

    tokenizer = get_tokenizer(model_name, add_prefix_space=True)

    p = get_hf_ds_scripts_path('vua20')
    data_files={'train': os.path.join(data_dir, 'train.tsv'), 'test': os.path.join(data_dir, 'test.tsv')}

    # ds = datasets.load_dataset(p, data_files=data_files, split='test[:20]')
    # ds = ds.map(tokenize_alingn_labels_replace_with_mask_and_add_type_ids)
    ds = get_tokenized_ds(p, tokenizer, tokenize_func='tagging', \
        tokenize_cols=['tokens'], tagging_cols={'is_target':0, 'labels':-100}, \
        data_files=data_files, batched=False, name = 'combined')
    # ds.rename_column_('target_mask', 'token_type_ids')

    # ds = ds.rename_column('is_target', 'token_type_ids')

    # dl = get_dataloader(ds, cols=['attention_mask', 'input_ids', 'labels'])
    args = get_base_hf_args(
        output_dir=output_dir,
        logging_steps=50,
        logging_dir = logging_dir,
        lr=5e-5,
        train_batch_size=24,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer, max_length=128)
    model = get_model(RobertaForTokenClassification, model_name, num_labels = 2)
    # model = get_model(RobertaForTokenClassification, model_name, num_labels = 2, type_vocab_size=2)
    # model = get_model(RobertaForTokenClassification, model_name, num_labels = 2,)
    # model.roberta.embeddings.token_type_embeddings = torch.nn.Embedding(2, 768)
    # model._init_weights(model.roberta.embeddings.token_type_embeddings)

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

    trainer.train()
    trainer.save_model()

    result = trainer.evaluate()
    print(result)

    # pred_out = trainer.predict(ds['test'])
    # pred_out = trainer.predict(ds)

    # predictions = pred_out.predictions
    # labels = pred_out.label_ids
    # predictions = np.argmax(predictions, axis=-1)

    # true_p, true_l = get_true_label(predictions, labels)
    # show_error_instances_id(true_p, true_l, prediction_output_file, ds['sent_id'], ds['tokens'])
    # write_predict_to_file(pred_out, ds['tokens'], out_file=prediction_output_file)
    # result = eval_with_weights(pred_out, ds['test']['token_type_ids'])
    # print(result)
