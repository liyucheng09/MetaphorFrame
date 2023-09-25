from torch.utils.data import (DataLoader,
                              TensorDataset,
                              Dataset,
                              IterableDataset)
from transformers import (BertModel,
                          BertTokenizer,
                          AutoModel,
                          AutoTokenizer)
from datasets import load_dataset, concatenate_datasets, Dataset as hfds, DatasetDict
import datasets
from tqdm import tqdm
import numpy as np
import torch
import os
import pickle
import pandas as pd

def get_tokenized_ds(scripts, tokenizer, max_length=64,
            slice=None, num_proc=None, shuffle=False, tokenize_func=None,
            batched=True, tokenize_cols=['tokens'], 
            is_split_into_words=False, return_word_ids=None, tagging_cols={'labels':-100}, remove_cols=None,
            **kwargs):
    """
    Given huggingface dataset-loading scripts and datapath, return processed datasets.

    Args:
        scripts: .py file
        path: datapath corresponding to the scripts
        tokenizer: huggingface tokenizer object
        slice: if given, will return a slice of the process dataset for testing usage.
        num_proc:
        shuffle
        tokenize_func
        cache_file_names
        batched
        tokenize_cols
        is_split_into_words
        return_word_ids
        tagging_cols
    Returns:
        ds: python dict object, where features names are the key, values are pytorch tensor.
    """

    def _slice(ds, slice):
        for k,v in ds.items():
            ds[k]=hfds.from_dict(v[:slice])
        return ds

    def _tokenize1(ds):
        """
        This function return tokenized results nestedly. 返回了嵌套结构，不能使用多进程
        Return:
            {'texta':{'input_ids': ..., 'attention_mask': ...}, ...}
        """
        results={}
        for k,v in ds.items():
            if k not in tokenize_cols:
                results[k]=v
                continue
            out_=tokenizer(v, max_length=max_length, padding=True, truncation=True)
            results[k]=dict(out_)
        return results

    def _tokenize2(ds):
        """
        本方法返回非嵌套结构，可以使用多进程处理。返回的features中的token_ids前面会加上features_name
        """
        results={}
        for k,v in ds.items():
            if k not in tokenize_cols:
                results.update({k:v})
                continue
            out_=tokenizer(v, max_length=max_length, padding=True, truncation=True)
            out_={k+'-'+k_: v_ for k_, v_ in out_.items()}
            results.update(out_)
        return results
    
    def _tokenize3(ds):
        results={}
        for k,v in ds.items():
            if k not in tokenize_cols:
                results[k]=v
                continue
            out_=tokenizer(v, is_split_into_words=is_split_into_words, max_length=max_length, padding='max_length', truncation=True)
            results.update(out_)
            if return_word_ids is not None:
                words_ids = [out_.word_ids(i) for i in range(len(out_.encodings))]
                results['words_ids']=words_ids
        return results

    def _tokenize4(ds):
        results={}
        for k,v in ds.items():
            if k not in tokenize_cols:
                results[k]=v
                continue
            out_=tokenizer(v, is_split_into_words=is_split_into_words)
            results.update(out_)
            if return_word_ids is not None:
                words_ids = [out_.word_ids(i) for i in range(len(out_.encodings))]
                results['words_ids']=words_ids
        return results
    
    def _tokenize_and_alingn_labels(ds):
        results={}
        for k,v in ds.items():
            if k not in tokenize_cols:
                results[k]=v
                continue
            out_=tokenizer(v, is_split_into_words=True, max_length=max_length, padding='max_length', truncation=True)
            results.update(out_)
        labels={}
        for i, column in enumerate(tagging_cols.keys()):
            label = ds[column]
            fillin_value = tagging_cols[column]
            words_ids = out_.word_ids()
            previous_word_idx = None
            label_ids = []
            for word_idx in words_ids:
                if word_idx is None:
                    label_ids.append(fillin_value)
                elif word_idx!=previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(fillin_value)
                previous_word_idx = word_idx
            labels[column] = label_ids
        
        results.update(labels)
        return results

    # def _tokenize4(ds):
    #     results={}
    #     for k,v in ds.items():
    #         if k == 'label':
    #             results[k]=v
    #             continue
    #         out_=tokenizer(v, max_length=max_length, padding=True, truncation=True, return_length=True, return_offsets_mapping=True)
    #         out_['real_length'] = [len(i) - 2 for i in out_['offset_mapping']]
    #         out_['length'] = out_['length']
    #         out_.pop('offset_mapping')
    #         results.update(out_)
    #         results['length'] = results['length'] - 2

    #     return results

    tokenize_funcs={
        'nested': _tokenize1,
        'with_prefix': _tokenize2,
        'general': _tokenize3,
        'no_padding': _tokenize4,
        'tagging': _tokenize_and_alingn_labels
    }
    
    def _get_col_names(col_names):
        if isinstance(col_names, list):
            return col_names
        elif isinstance(col_names, dict):
            cols_needed_removed=set()
            for k,v in col_names.items():
                cols_needed_removed.update(v)
            return cols_needed_removed

    if isinstance(scripts, str):
        ds=load_dataset(scripts, **kwargs)
    elif isinstance(scripts, (hfds, DatasetDict)):
        ds=scripts

    # if ds_name in ds.column_names.keys():
    #     ds=ds[ds_name]
    # elif 'train' in ds.column_names.keys():
    #     train_ds = ds['train']
    # elif 'dev' in ds.column_names.keys():
    #     dev_ds = ds['dev']
    # elif 'test' in ds.column_names.keys():
    #     test_ds = ds['test']

    ds = _slice(ds, slice) if slice else ds

    if remove_cols is not None:
        remove_cols=_get_col_names(ds.column_names)
    
    print(ds)
    tokenize_cols = tokenize_cols or ['tokens']
    if tokenize_func is not None:
        ds = ds.map(
            tokenize_funcs[tokenize_func],
            remove_columns=remove_cols,
            batched=batched,
            num_proc=num_proc,
        )

    if shuffle:
        ds=ds.shuffle()
    
    return ds

def get_dataloader(ds: hfds, batch_size=32, cols=['input_ids', 'attention_mask', 'token_type_ids', 'label'], shuffle=False, **kwargs):
    ds.set_format(type='torch', columns=cols)
    dl=DataLoader(ds, batch_size, shuffle=shuffle, **kwargs)
    return dl

def wrap_sentences_to_ds(sents, tokenize_func, **kwargs):
    encoding = tokenize_func(sents, **kwargs)
    ds = datasets.Dataset.from_dict(encoding)
    return ds

class SentencePairDataset(Dataset):
    """
    句子对/单句数据集。用于训练/使用句向量模型。

    Train: 
        given sentence pair dataset: (sentence_a, sentence_b, label)
    
    predict:
        given single sentence: (sentence_a)
    """
    def __init__(self, tokenized_a, tokenized_b=None, label=None):
        self.tokenized_a=tokenized_a
        self.tokenized_b=tokenized_b
        self.label=label

    def __len__(self):
        return self.tokenized_a['input_ids'].shape[0]

    def __getitem__(self, index):
        input_a = {
            k:v[index] for k,v in self.tokenized_a.items()
        }
        output=(input_a, )
        if self.tokenized_b is not None:
            input_b={
                k:v[index] for k,v in self.tokenized_b.items()
            }
            output+=(input_b, )
        if self.label is not None:
            output+=(torch.LongTensor([self.label[index]]), )
        return output

class SentenceDataset(Dataset):
    """
    单句Sentences Dataset

    Args:
        tokenized_a: sentences的encoding
        label: list[int]
        idxs: 当要取某个位置的向量时给出
    """
    def __init__(self, tokenized_a, label=None, idxs=None):
        self.tokenized_a=tokenized_a
        self.idxs=idxs
        self.label=label

    def __len__(self):
        return self.tokenized_a['input_ids'].shape[0]

    def __getitem__(self, index):
        input_a = {
            k:v[index] for k,v in self.tokenized_a.items()
        }
        output=(input_a, )
        if self.idxs is not None:
            output+=(torch.LongTensor([self.idxs[index]]), )
        if self.label is not None:
            output+=(torch.LongTensor([self.label[index]]), )
        return output

class SimCSEDataSet(IterableDataset):
    """给定一个list的句子，打乱顺序后给出两两配对的句子对，并生成label(其形状为除了自己和自己配对为positive之外均为negative)。

    Args:
        tokenized_a: tokenized sentence encoding
    """
    def __init__(self, tokenized_a, batch_size=32):
        self.tokenized_a=tokenized_a
        self.batch_size=batch_size
        self.idxs=np.random.permutation(len(self.tokenized_a['input_ids']))

    def __iter__(self):
        count=0
        while count<len(self.idxs):
            selected_ids=self.idxs[count:count+self.batch_size]
            inputs={k: v[selected_ids].repeat(2,1) for k,v in self.tokenized_a.items() if k!='label'}
            idx1=torch.arange(self.batch_size*2)[None, :]
            idx2=(idx1.T+self.batch_size)%(self.batch_size*2)
            label = torch.LongTensor(idx2)
            yield {'inputs': inputs, 'label': label}
            count+=self.batch_size

class processor:

    block_size : int = 512
    tokenizer = None
    is_wordpiece = lambda x: x.startwith('##')
    
    @classmethod
    def lm_group_texts(cls, examples):
        """
        将离散句子合并为block_size长度的文本输入
        需要设定processor的block_size变量
        """

        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // cls.block_size) * cls.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + cls.block_size] for i in range(0, total_length, cls.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    @classmethod
    def get_true_length(cls, examples):
        assert cls.tokenizer is not None
        examples['n'] = [sum(i) for i in examples['attention_mask']]
        examples['n_real'] = [sum([0 if cls.tokenizer.convert_ids_to_tokens(i).startswith('##') 
                            else 1 for i in line])  for line in examples['input_ids']]
        return examples
    
    @classmethod
    def combine(cls, ds, group_column, combine_func):
        assert isinstance(ds, hfds), 'ds should be a datasets.Dataset object'
        ds = ds.to_pandas()
        combined = []
        for sent_id, group in ds.groupby(group_column) :
            combined.append(combine_func(group))
        return datasets.Dataset.from_pandas(pd.DataFrame(combined))
    
    @classmethod
    def _tokenize_and_alingn_labels(cls, ds):
        results={}
        for k,v in ds.items():
            if 'id' in k:
                results[k]=v
                continue
            if 'tag' not in k:
                out_=cls.tokenizer(v, is_split_into_words=True)
                results.update(out_)
        labels={}
        for i, column in enumerate([k for k in ds.keys() if 'tag' in k]):
            label = ds[column]
            words_ids = out_.word_ids()
            previous_word_idx = None
            label_ids = []
            for word_idx in words_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx!=previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels[column] = label_ids
        
        results.update(labels)
        return results

def get_hf_ds_scripts_path(ds_name):
    relative_path={
        'atec':'hfds_scripts/atec_dataset.py',
        'sesame':'hfds_scripts/sesame_dataset.py',
        'vua20': 'hfds_scripts/vua20_dataset.py',
        'moh': 'hfds_scripts/moh_dataset.py',
        'hard': 'hfds_scripts/hard_metaphor_dataset.py',
        'empathy': 'hfds_scripts/empathy_dataset.py',
        'cmc': 'hfds_scripts/cmc_dataset.py',
        'chinese_simile': 'hfds_scripts/chinese_simile_dataset.py'
    }

    return os.path.join(os.path.dirname(__file__), relative_path[ds_name])

def get_hf_metrics_scripts_path(metrics_name):
    relative_path={
        'bleu':'hfds_metrics/bleu.py'
    }

    return os.path.join(os.path.dirname(__file__), relative_path[metrics_name])


if __name__ == '__main__':
    # from utils import get_tokenizer
    # from copy import deepcopy

    # t=get_tokenizer('bert-base-chinese', is_zh=True)
    # ds = get_tokenized_ds('hfds_scripts/atec_dataset.py', '../sentence-embedding/data/ATEC/atec_nlp_sim_train.csv', t, tokenize_type='with_prefix')

    # ds = ds['atec']
    # ds2=deepcopy(ds)

    # for index, ds_ in enumerate([ds, ds2]):
    #     features=list(ds_.features)
    #     for feature in features:
    #         if index:
    #             if feature.startswith('textb') or feature == 'label':
    #                 ds_.remove_columns_(feature)
    #             else:
    #                 ds_.rename_column_(feature, feature[6:])
    #         else:
    #             if feature.startswith('texta') or feature == 'label':
    #                 ds_.remove_columns_(feature)
    #             else:
    #                 ds_.rename_column_(feature, feature[6:])
    
    # ds=concatenate_datasets([ds, ds2])
    # print(ds)

    script = get_hf_ds_scripts_path('vua20')
    data_dir = '/Users/yucheng/projects/Metaphor_Processing_analysis/VUA20/'
    vua = load_dataset(script, data_dir= data_dir)
    print(vua)