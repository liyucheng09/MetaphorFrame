from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from transformers.trainer_callback import TrainerCallback
from lyc.utils import vector_l2_normlize
import numpy as np
import torch
import pandas as pd
from scipy.special import expit
from collections.abc import Iterable

metrics_computing={
    'acc': accuracy_score,
}

def tagging_eval_for_trainer(eval_prediction):
    """
    Trainer专用标注问题的通用compute_metrics函数
    This function can be sent to huggingface.Trainer as computing_metrics funcs.
    Args:
        eval_prediction ([type]): two atrributes
            - predictions
            - label_ids
    """

    predictions, labels = eval_prediction
    predictions = np.argmax(predictions, axis=-1)

    # true_predictions = [
    #     [p for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]
    # true_labels = [
    #     [l for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]

    true_predictions = [ p for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100]
    true_labels = [ l for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100]

    return {
        "accuracy_score": accuracy_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions, average='micro'),
        "recall": recall_score(true_labels, true_predictions, average='micro'),
        "micro_f1": f1_score(true_labels, true_predictions, average='micro'),
        "macro_f1": f1_score(true_labels, true_predictions, average='macro'),
    }

def frame_finder_eval_for_trainer(eval_prediction):
    """
    Trainer专用标注问题的通用compute_metrics函数
    This function can be sent to huggingface.Trainer as computing_metrics funcs.
    Args:
        eval_prediction ([type]): two atrributes
            - predictions
            - label_ids
    """

    predictions, labels = eval_prediction
    labels, sent_labels = labels
    sent_pred = predictions[:, 0]
    sent_pred = sent_pred>0

    predictions = np.argmax(predictions, axis=-1)

    # true_predictions = [
    #     [p for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]
    # true_labels = [
    #     [l for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]

    true_predictions = [ p for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100]
    true_labels = [ l for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100]

    return {
        "f1": f1_score(true_labels, true_predictions, average='micro'),
        "sent_f1": f1_score(sent_labels, sent_pred, average='micro'),
    }

def show_error_instances_id(preds, labels, output_file, *args):
    """直接输出错误预测样例的id，id需要和pred和label是同样的shape.

    Args:
        pred : 预测到的标签，应和label是同样的shape
        label: label
        output_file : 输出文件地址
    """

    columns = (preds, labels) + args
    out_file = open(output_file, 'w', encoding='utf-8')
    for ins in zip(*columns):
        pred, label = ins[:2]

        # if not all is 1-D array/list
        if any([isinstance(i, Iterable) for i in ins if not isinstance(i, str)]):
            # print('Assuming 2-D input preds and labels')
            iterrows = [i for i in ins if isinstance(i, Iterable) and not isinstance(i, str)]
            not_iterrows = [i for i in ins if isinstance(i, str) or not isinstance(i, Iterable)]
            to_write = '\t'.join([str(i) for i in not_iterrows])
            for ins2 in zip(*iterrows):
                p,l=ins2[:2]
                to_write_2 = '\t'.join([ str(i) for i in ins2])
                if p!=l:
                    out_file.write(to_write_2 + '\t' + to_write+'\n')
            out_file.write('\n')
        else:
            to_write = [ str(i) for i in ins]
            if pred != lable:
                out_file.write('\t'.join(to_write)+'\n')
    print('Done!')
    out_file.close()

def get_true_label_and_token(predictions, labels, ignore_index=-100, tokens = None, tokenizer = None):
    """去掉padding/BPE造成的填充label

    Args:
        pred ([type]): 预测到的label，非logits。可以是1-D，也可以是2-D array
        labels ([type]): 同pred的shape
        ignore_index: 要忽略的label id
    """
    if len(predictions.shape)==2:
        print('&&& Assuming tagging predictions:')
        true_predictions = [
            [p for (p, l) in zip(prediction, label) if l != ignore_index]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [l for (p, l) in zip(prediction, label) if l != ignore_index]
            for prediction, label in zip(predictions, labels)
        ]
        if tokens is not None:
            assert tokenizer is not None
            true_tokens = [
                [tokenizer.convert_ids_to_tokens(t) for (t, l) in zip(token, label) if l != ignore_index]
                for token, label in zip(tokens, labels)
            ]
    elif len(predictions.shape)==1:
        true_predictions = [p for p,l in zip(predictions, labels) if l !=ignore_index]
        true_labels = [l for p,l in zip(predictions, labels) if l !=ignore_index]
    else:
        raise ValueError('Do not support non 2-d, 1-d inputs')
    
    outputs = (true_predictions, true_labels,)
    if tokens is not None:
        outputs += (true_tokens,)
    return outputs

def write_predict_to_file(pred_out, tokens=None, out_file='predictions.csv', label_list=None, tokenizer = None):
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

    predictions = np.argmax(predictions, axis=-1)

    if len(labels.shape) == 2:
        outputs = get_true_label_and_token(predictions, labels, tokens = tokens, tokenizer = tokenizer)
        with open(out_file, 'w', encoding='utf-8') as f:
            for cols in zip(*outputs):
                for items in zip(*cols):
                    items = [str(i) for i in items]
                    f.write( "\t".join(items) + '\n')
                f.write('\n')
        print(f'Save to conll file {out_file}.')
        return
    elif len(labels.shape) == 1:
        result = {'prediction': predictions, 'labels': labels}
        if tokens is not None:
            result['tokens'] = tokens
        df = pd.DataFrame(result)
        df.to_csv(out_file, index=False, sep='\t')
        print(f'Save to csv file {out_file}.')
        return

def eval_with_weights(pred_out, weights):
    predictions = pred_out.predictions
    labels = pred_out.label_ids

    predictions = np.argmax(predictions, axis=-1)

    if len(labels.shape) == 2:
        print('&&& Assuming tagging predictions:')
        true_predictions = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [l for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_weights = [
            [w for (w, l) in zip(weight, label) if l != -100]
            for weight, label in zip(weights, labels)
        ]
        p,l,w = np.concatenate(true_predictions), np.concatenate(true_labels), np.concatenate(true_weights)

    elif len(labels.shape) == 1:
        p,l,w = predictions, labels, weights
    
    return {
        "accuracy_score": accuracy_score(p, l, sample_weight=w),
        "precision": precision_score(p, l, sample_weight=w, average='micro'),
        "recall": recall_score(p, l, sample_weight=w, average='micro'),
        "f1": f1_score(p, l, sample_weight=w, average='micro'),
    }


class Evaluator:

    preprosess_func = None

    @classmethod
    def pred_forward(cls, model, eval_dl):
        model.eval()
        all_preds = []
        all_true = []
        all_loss = []
        for batch in eval_dl:
            outputs=model(**batch)
            all_preds.append(outputs.pred)
            all_true.append(outputs.target)
            all_loss.append(outputs.loss)
        
        all_true = torch.cat(all_true)
        all_preds = torch.cat(all_preds)
        all_loss = torch.stack(all_loss)
        
        return all_true.detach().numpy(), all_preds.detach().numpy(), all_loss.detach().numpy()

    @classmethod
    def GeneralEval(cls, model, eval_dl, writer=None, metrics=None, global_step=None):

        all_true, all_preds, all_loss = cls.pred_forward(model, eval_dl)
        
        results = {}

        if metrics is not None:
            for metric in metrics:
                results[metric] = metrics_computing[metric](all_true, all_preds)
        
            for k,v in results.items():
                writer.add_scalar(k, v, global_step)
        
        mean_loss = all_loss.mean()
        writer.add_scalar('Eval_loss', mean_loss, global_step)
        results['eval_loss'] = mean_loss
        
        return results

def SimCSEEvalAccComputing(preds, threshold=0.4):
    prediction=preds.prediction
    labels=pred.label_ids

    prediction = vector_l2_normlize(prediction)
    embs_a, embs_b = np.split(prediction, 2)
    sims = np.dot(embs_a, embs_b.T)
    sims = np.diag(sims)
    acc=accuracy_score(labels, sims>threshold)
    print('ACC: ', acc)
    return {'ACC' : acc}
