import numpy as np

import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from run_classifier_dataset_utils import (
    convert_examples_to_two_features,
    convert_examples_to_features,
    convert_two_examples_to_features,
)
import datasets
from lyc.data import get_hf_ds_scripts_path, get_tokenized_ds, processor, get_dataloader
from transformers import DataCollatorForTokenClassification

def tokenize_alingn_labels_replace_with_mask_and_add_type_ids(ds, tokenizer=None, do_mask=True):
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

def load_frame_data(tokenizer, args, combine = False, melbert_data_size=None, data_dir = 'data_all/open_sesame_v1_data/fn1.7', do_mask=False):
    data_collator = DataCollatorForTokenClassification(tokenizer, max_length=128)

    script = get_hf_ds_scripts_path('sesame')
    ds = datasets.load_dataset(script, data_dir=data_dir)
    if combine:
        for k,v in ds.items():
            ds[k] = processor.combine(v, 'sent_id', combine_func)
    ds = ds.map(
        tokenize_alingn_labels_replace_with_mask_and_add_type_ids, fn_kwargs={'tokenizer':tokenizer, 'do_mask':do_mask}
    )
    train_ds = datasets.concatenate_datasets([ds['train'], ds['test']])
    train_ds = train_ds.rename_column('frame_tags', 'labels')
    train_ds = train_ds.rename_column('is_target', 'token_type_ids')

    eval_ds = ds['validation']
    eval_ds = eval_ds.rename_column('frame_tags', 'labels')
    eval_ds = eval_ds.rename_column('is_target', 'token_type_ids')

    train_ds.set_format(columns=['input_ids', 'token_type_ids', 'labels', 'attention_mask'])
    eval_ds.set_format(columns=['input_ids', 'token_type_ids', 'labels', 'attention_mask'])

    train_dl = DataLoader(train_ds, args.train_batch_size if melbert_data_size is None else int((len(train_ds)/melbert_data_size)*args.train_batch_size), shuffle=True, collate_fn=data_collator)
    eval_dl = None

    # train_dl, eval_dl = get_dataloader(train_ds, cols=['input_ids', 'token_type_ids', 'labels', 'attention_mask'], batch_size=args.train_batch_size if melbert_data_size is None else int((len(train_ds)/melbert_data_size)*args.train_batch_size), collate_fn=data_collator), get_dataloader(eval_ds, cols=['input_ids', 'token_type_ids', 'labels', 'attention_mask'], batch_size=args.eval_batch_size)
    return train_dl, eval_dl

def load_train_data(args, logger, processor, task_name, label_list, tokenizer, output_mode, k=None):
    # Prepare data loader
    if task_name == "vua":
        train_examples = processor.get_train_examples(args.data_dir)
        # train_examples = train_examples[:20]
    elif task_name == "trofi":
        train_examples = processor.get_train_examples(args.data_dir, k)
    else:
        raise ("task_name should be 'vua' or 'trofi'!")

    # make features file
    if args.model_type == "BERT_BASE":
        train_features = convert_two_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode
        )
    if args.model_type in ["BERT_SEQ", "MELBERT_SPV"]:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )
    if args.model_type in ["MELBERT_MIP", "MELBERT", "FrameMelbert"]:
        train_features = convert_examples_to_two_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )

    # make features into tensor
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    # add additional features for MELBERT_MIP and MELBERT
    if args.model_type in ["MELBERT_MIP", "MELBERT", "FrameMelbert"]:
        all_input_ids_2 = torch.tensor([f.input_ids_2 for f in train_features], dtype=torch.long)
        all_input_mask_2 = torch.tensor([f.input_mask_2 for f in train_features], dtype=torch.long)
        all_segment_ids_2 = torch.tensor(
            [f.segment_ids_2 for f in train_features], dtype=torch.long
        )
        if args.spvmask or args.spvmaskcls:
            all_input_with_mask_ids = torch.tensor([f.input_with_mask_ids for f in train_features], dtype=torch.long)
            train_data = TensorDataset(
                all_input_ids,
                all_input_mask,
                all_segment_ids,
                all_label_ids,
                all_input_ids_2,
                all_input_mask_2,
                all_segment_ids_2,
                all_input_with_mask_ids
            )
        else:
            train_data = TensorDataset(
                all_input_ids,
                all_input_mask,
                all_segment_ids,
                all_label_ids,
                all_input_ids_2,
                all_input_mask_2,
                all_segment_ids_2,
            )
    else:
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size
    )

    return train_dataloader


def load_train_data_kf(
    args, logger, processor, task_name, label_list, tokenizer, output_mode, k=None
):
    # Prepare data loader
    if task_name == "vua":
        train_examples = processor.get_train_examples(args.data_dir)
    elif task_name == "trofi":
        train_examples = processor.get_train_examples(args.data_dir, k)
    else:
        raise ("task_name should be 'vua' or 'trofi'!")

    # make features file
    if args.model_type == "BERT_BASE":
        train_features = convert_two_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode
        )
    if args.model_type in ["BERT_SEQ", "MELBERT_SPV"]:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        train_features = convert_examples_to_two_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )

    # make features into tensor
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    # add additional features for MELBERT_MIP and MELBERT
    if args.model_type in ["MELBERT_MIP", "MELBERT"]:
        all_input_ids_2 = torch.tensor([f.input_ids_2 for f in train_features], dtype=torch.long)
        all_input_mask_2 = torch.tensor([f.input_mask_2 for f in train_features], dtype=torch.long)
        all_segment_ids_2 = torch.tensor(
            [f.segment_ids_2 for f in train_features], dtype=torch.long
        )
        train_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_input_ids_2,
            all_input_mask_2,
            all_segment_ids_2,
        )
    else:
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    gkf = StratifiedKFold(n_splits=args.num_bagging).split(X=all_input_ids, y=all_label_ids.numpy())
    return train_data, gkf


def load_test_data(args, logger, processor, task_name, label_list, tokenizer, output_mode, k=None):
    if task_name == "vua":
        eval_examples = processor.get_test_examples(args.data_dir)
        # eval_examples = eval_examples[:20]
    elif task_name == "trofi":
        eval_examples = processor.get_test_examples(args.data_dir, k)
    else:
        raise ("task_name should be 'vua' or 'trofi'!")

    if args.model_type == "BERT_BASE":
        eval_features = convert_two_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode
        )
    if args.model_type in ["BERT_SEQ", "MELBERT_SPV"]:
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )
    if args.model_type in ["MELBERT_MIP", "MELBERT", "FrameMelbert"]:
        eval_features = convert_examples_to_two_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
        )

    logger.info("***** Running evaluation *****")
    if args.model_type in ["MELBERT_MIP", "MELBERT", "FrameMelbert"]:
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_guids = [f.guid for f in eval_features]
        all_idx = torch.tensor([i for i in range(len(eval_features))], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_input_ids_2 = torch.tensor([f.input_ids_2 for f in eval_features], dtype=torch.long)
        all_input_mask_2 = torch.tensor([f.input_mask_2 for f in eval_features], dtype=torch.long)
        all_segment_ids_2 = torch.tensor([f.segment_ids_2 for f in eval_features], dtype=torch.long)
        if args.spvmask or args.spvmaskcls:
            all_input_with_mask_ids = torch.tensor([f.input_with_mask_ids for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(
                all_input_ids,
                all_input_mask,
                all_segment_ids,
                all_label_ids,
                all_idx,
                all_input_ids_2,
                all_input_mask_2,
                all_segment_ids_2,
                all_input_with_mask_ids
            )
        else:
            eval_data = TensorDataset(
                all_input_ids,
                all_input_mask,
                all_segment_ids,
                all_label_ids,
                all_idx,
                all_input_ids_2,
                all_input_mask_2,
                all_segment_ids_2,
            )
    else:
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_guids = [f.guid for f in eval_features]
        all_idx = torch.tensor([i for i in range(len(eval_features))], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_idx
        )

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    return all_guids, eval_dataloader
