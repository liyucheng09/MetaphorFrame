import sys
from transformers import RobertaForTokenClassification, DataCollatorForTokenClassification, Trainer, RobertaTokenizerFast
from transformers.integrations import TensorBoardCallback
import os
import datasets
import torch
import numpy as np
# from lyc.eval import write_predict_to_file
from itertools import groupby

def get_true_label(predictions, pad_mask = None, labels = None, input_ids = None, ignore_index=-100):
    """去掉padding/BPE造成的填充label

    Args:
        pred ([type]): 预测到的label，非logits。可以是1-D，也可以是2-D array
        labels ([type]): 同pred的shape
        ignore_index: 要忽略的label id
    """
    if pad_mask is None:
        if labels is None:
            raise ValueError('pad_mask and labels cannot be both None')
        else:
            pad_mask = labels
    if len(predictions.shape)==2:
        print('&&& Assuming tagging predictions:')
        true_predictions = [
            [p for (p, l) in zip(prediction, pad) if l != ignore_index]
            for prediction, pad in zip(predictions, pad_mask)
        ]
        if labels is not None:
            true_labels = [
                [l for (p, l, d) in zip(prediction, label, pad) if d != ignore_index]
                for prediction, label, pad in zip(predictions, labels, pad_mask)
            ]
        if input_ids is not None:
            true_input_ids = [
                [i for (i, d) in zip(input_id, pad) if d != ignore_index]
                for input_id, pad in zip(input_ids, pad_mask)
            ]
    elif len(predictions.shape)==1:
        true_predictions = [p for p,d in zip(predictions, pad_mask) if d !=ignore_index]
        if labels is not None:
            true_labels = [l for p,l,d in zip(predictions, labels, pad_mask) if d !=ignore_index]
    else:
        raise ValueError('Do not support non 2-d, 1-d inputs')
    
    output = (true_predictions,)
    if labels is not None:
        output += (true_labels,)
    if input_ids is not None:
        output += (true_input_ids,)
    return output

def reconstruct_words_from_ids(encoding):
    """从encoding中重构出原始的words，主要是针对BPE的情况，因为BPE会把一个word分成多个token，这里把它们合并起来。
    """
    words = []
    for i in encoding.word_ids():
        if i is None:
            continue
        elif i == prev_i:
            words[-1] += encoding.tokens[i]


def write_predict_to_file(pred_out, pad_mask = None, tokens=None, out_file='predictions.csv', label_list=None):
    """将model的预测结果写入到文件中。目前支持2-D输入(tagging问题)和1-D输入(分类问题)。
    默认将去除label==-100的部分，因为大多数时候是padding/BPE带来的冗余部分。

    Args:
        pred_out ([type]): logits
        tokens ([type]): golden label
        out_file (str, optional): 输出地址. Defaults to 'predictions.csv'.
        label_list ([type], optional): id2label的mapping，支持dict. Defaults to None.
    """
    predictions = pred_out.predictions
    labels = pred_out.label_ids
    if labels is None:
        if pad_mask is None:
            raise ValueError('pad_mask and label_list cannot be both None')

    predictions = np.argmax(predictions, axis=-1)
    if labels is not None:
        true_predictions, true_labels, = get_true_label(predictions, labels = labels,)
    else:
        true_predictions, = get_true_label(predictions, pad_mask = pad_mask)

    if labels is None:
        if len(predictions.shape) == 2:
            with open(out_file, 'w', encoding='utf-8') as f:
                for p,token in zip(true_predictions, tokens):
                    for i,k in zip(p,token):
                        f.write(f'{k}\t{i}\n')
                    f.write('\n')
            print(f'Save to conll file {out_file}.')
            return
        elif len(predictions.shape) == 1:
            result = {'prediction': predictions, 'tokens': tokens}
            df = pd.DataFrame(result)
            df.to_csv(out_file, index=False)
            print(f'Save to csv file {out_file}.')
            return
    else:
        if len(labels.shape) == 2:
            with open(out_file, 'w', encoding='utf-8') as f:
                for p,l,token in zip(true_predictions, true_labels, tokens):
                    for i,j,k in zip(p,l,token):
                        f.write(f'{k}\t{j}\t{i}\n')
                    f.write('\n')
            print(f'Save to conll file {out_file}.')
            return
        elif len(labels.shape) == 1:
            result = {'prediction': predictions, 'labels': labels, 'tokens': tokens}
            df = pd.DataFrame(result)
            df.to_csv(out_file, index=False)
            print(f'Save to csv file {out_file}.')
            return

if __name__ == '__main__':

    model_path, = sys.argv[1:]
    prediction_output_file = os.path.join(model_path, 'predictions.csv')

    model = RobertaForTokenClassification.from_pretrained(model_path, num_labels=2)
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base',)

    data_collator = DataCollatorForTokenClassification(tokenizer, max_length=128, )

    sents = ['I like you', 'I hate you']
    ds = datasets.Dataset.from_dict({'tokens': sents})

    def tokenize(x):
        out = tokenizer(x['tokens'], truncation=True, padding='max_length')
        words_ids = out.word_ids()
        input_ids = out['input_ids']

        pad_mask = []
        for word_idx in words_ids:
            if word_idx is None:
                pad_mask.append(-100)
            elif word_idx!=previous_word_idx:
                pad_mask.append(0)
            else:
                pad_mask.append(-100)
            previous_word_idx = word_idx
        
        grouped_ids = [list(group) for word_id, group in groupby(input_ids, lambda id: words_ids[input_ids.index(id)]) if word_id is not None]
        words = [''.join(tokenizer.convert_ids_to_tokens(group)).replace('##', '') for group in grouped_ids]
        words = [token.replace('Ġ', '') for token in words]
        
        out['pad_mask'] = pad_mask
        out['words'] = words
        return out

    ds = ds.map(tokenize)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    pred_out = trainer.predict(ds)
    write_predict_to_file(pred_out, ds['pad_mask'], ds['words'], out_file=prediction_output_file)