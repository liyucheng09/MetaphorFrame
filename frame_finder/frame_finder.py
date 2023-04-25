from lyc.utils import get_tokenizer, get_model
from lyc.data import get_hf_ds_scripts_path, get_tokenized_ds, processor, get_dataloader
from lyc.train import get_base_hf_args, HfTrainer
from lyc.eval import frame_finder_eval_for_trainer, tagging_eval_for_trainer
import sys
import numpy as np
import datasets
from transformers import RobertaForTokenClassification, DataCollatorForTokenClassification, Trainer
from transformers.integrations import TensorBoardCallback
import torch
import pandas as pd
from model import FrameFinder, DataCollator

def combine_func(df):
    """combine a dataframe group to a one-line instance.

    Args:
        df ([dataframe]): a dataframe, represents a group

    Returns:
        a one-line dict
    """
    label_values = np.stack(df['frame_tags'].values)
    processed = np.zeros_like(label_values)

    for token_id in range(label_values.shape[1]):
        labels = label_values[:, token_id]
        for i in range(len(labels)):
            if labels[i] != 0:
                processed[i, token_id] = labels[i]
                break
    aggregated_tags = processed.sum(axis=0)
    result = df.iloc[0].to_dict()
    result['frame_tags'] = aggregated_tags

    return result

def get_sent_label(ds, combine_func, group_column):

    ds = ds.to_pandas()
    combined = []
    for sent_id, group in ds.groupby(group_column):
        sent_label=combine_func(group)['frame_tags'].tolist()
        sent_label = [i for i in sent_label if i]
        group['sent_labels']=[sent_label for i in range(len(group.index))]
        combined.append(group)
    return datasets.Dataset.from_pandas(pd.concat(combined))

def tokenize_alingn_labels_replace_with_mask_and_add_type_ids(ds, do_mask=True):
    results={}

    target_ids = []
    sent_labels = [0 for i in range(797)]
    for i in range(len(ds['frame_tags'])):
        l = ds['frame_tags'][i]
        if l:
            target_ids.append(i)
            sent_labels[l]=1
    
    if do_mask:
        tokens = ds['tokens']
        for target_idx in target_ids:
            tokens[target_idx] = '<mask>'
        ds['tokens'] = tokens

    for k,v in ds.items():
        if 'id' in k:
            results[k]=v
            continue
        if k == 'tokens':
            out_=tokenizer(v, is_split_into_words=True)
            results.update(out_)
    labels={'sent_labels': sent_labels}
    for i, column in enumerate([k for k in ds.keys() if 'tag' in k]):
        label = ds[column]
        words_ids = out_.word_ids()
        previous_word_idx = None
        label_ids = []
        is_target = []
        for word_idx in words_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx!=previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            if word_idx in target_ids:
                is_target.append(1)
            else:
                is_target.append(0)
            previous_word_idx = word_idx
        labels[column] = label_ids
        labels['is_target'] = is_target
    
    results.update(labels)
    return results

if __name__ == '__main__':
    model_name, data_dir, = sys.argv[1:]
    add_sent_labels = False
    do_mask = True
    output_path = '/vol/research/nlg/frame_finder/'
    # output_path = ''

    tokenizer = get_tokenizer(model_name, add_prefix_space=True)
    script = get_hf_ds_scripts_path('sesame')
    ds = datasets.load_dataset(script, data_dir=data_dir)

    label_list = ds['train'].features['frame_tags'].feature.names
    for k,v in ds.items():
        ds[k] = processor.combine(v, 'sent_id', combine_func)
    if add_sent_labels:
        model_class = FrameFinder
        # for k,v in ds.items():
        #     ds[k] = get_sent_label(v, combine_func, 'sent_id')
    else:
        model_class = RobertaForTokenClassification

    ds = ds.map(
        tokenize_alingn_labels_replace_with_mask_and_add_type_ids, fn_kwargs={'do_mask':do_mask}
    )

    train_ds = datasets.concatenate_datasets([ds['train'], ds['test']])
    train_ds = train_ds.rename_column('frame_tags', 'labels')
    train_ds = train_ds.rename_column('is_target', 'token_type_ids')

    eval_ds = ds['validation']
    eval_ds = eval_ds.rename_column('frame_tags', 'labels')
    eval_ds = eval_ds.rename_column('is_target', 'token_type_ids')

    args = get_base_hf_args(
        output_dir = output_path + 'checkpoints/mask_ff_combined/',
        train_batch_size=24,
        epochs=10,
        lr=5e-5,
        logging_steps = 50,
        # evaluation_strategy = 'epoch',
        evaluation_strategy = 'steps',
        eval_steps=50,
        save_strategy='no',
        label_names=['labels', 'sent_labels'] if add_sent_labels else ['labels'],
        logging_dir = output_path + 'logs/mask_ff_combined',
    )

    model = get_model(model_class, model_name, num_labels = len(label_list))
    model.roberta.embeddings.token_type_embeddings = torch.nn.Embedding(2, 768)
    model._init_weights(model.roberta.embeddings.token_type_embeddings)

    data_collator = DataCollatorForTokenClassification(tokenizer, max_length=128)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[TensorBoardCallback()],
        compute_metrics=frame_finder_eval_for_trainer if add_sent_labels else tagging_eval_for_trainer,
    )

    trainer.train()
    trainer.save_model()

    result = trainer.evaluate()
    print(result)