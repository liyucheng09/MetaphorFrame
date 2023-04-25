from lyc.data import get_hf_ds_scripts_path, get_tokenized_ds, get_dataloader
from lyc.utils import get_model, get_tokenizer
from transformers import RobertaForTokenClassification, Trainer, DataCollatorForTokenClassification
import sys
from tqdm import tqdm
import torch
import json
import datasets
import os
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

def literal_processor(ds):
    
    results = {}
    out = tokenizer(ds['target_word'])
    words_ids = out.word_ids()
    target_mask = [1 if i is not None else 0 for i in words_ids]
    results['target_mask'] = target_mask
    results.update(out)

    return results

if __name__ == '__main__':

    model_name, data_dir, = sys.argv[1:]
    do_mask = False
    save_folder = '/vol/research/nlg/frame_finder/'
    # save_folder = ''
    output_dir = os.path.join(save_folder, 'checkpoints/no_mask_ff/')
    prediction_output_file = os.path.join(output_dir, 'good_examples.csv')

    tokenizer = get_tokenizer(model_name, add_prefix_space=True)
    script_path = get_hf_ds_scripts_path('vua20')

    data_files={'train': os.path.join(data_dir, 'train.tsv'), 'test': os.path.join(data_dir, 'test.tsv')}
    ds = datasets.load_dataset(script_path, data_files=data_files, split='test[:40%]')
    ds = ds.shuffle()
    ds = datasets.Dataset.from_dict(ds[:200])
    ds = ds.map(tokenize_alingn_labels_replace_with_mask_and_add_type_ids, fn_kwargs={'do_mask':False})
    ds.remove_columns_('label')
    ds.rename_column_('target_mask', 'token_type_ids')

    # mask_ds = ds.map(tokenize_alingn_labels_replace_with_mask_and_add_type_ids, fn_kwargs={'do_mask':True})
    # mask_ds.remove_columns_('label')
    # mask_ds.rename_column_('target_mask', 'token_type_ids')

    literal_ds = datasets.Dataset.from_dict({'target_word': ds['target_word']})
    literal_ds = literal_ds.map(literal_processor)
    literal_ds.rename_column_('target_mask', 'token_type_ids')

    # ds = get_tokenized_ds(script_path, tokenizer, max_length=128, tokenize_func='general', tokenize_cols=['tokens'],
    #     is_split_into_words=True, return_word_ids=True,
    #     data_files={
    #         'train':'/Users/liyucheng/projects/acl2021-metaphor-generation-conceptual-main/EM/data/VUA20/train.tsv',
    #         'test': '/Users/liyucheng/projects/acl2021-metaphor-generation-conceptual-main/EM/data/VUA20/test.tsv'})
    
    # dl = get_dataloader(ds['test'], batch_size=8, cols=['input_ids', 'attention_mask', 'word_index', 'words_ids', 'label'])
    
    frame_path = os.path.join('/user/HS502/yl02706/frame_finder/', 'frame_labels.json')
    with open(frame_path, encoding='utf-8') as f:
        label2id = json.load(f)
    id2label = {v:k for k,v in label2id.items()}

    model = get_model(RobertaForTokenClassification, model_name, type_vocab_size=2)
    # model.roberta.embeddings.token_type_embeddings = torch.nn.Embedding(2, 768)

    data_collator = DataCollatorForTokenClassification(tokenizer, max_length=128)
    trainer = Trainer(model=model, data_collator=data_collator, tokenizer=tokenizer)
    
    pred_out = trainer.predict(ds)
    pred = - pred_out.predictions

    literal_pred = trainer.predict(literal_ds)
    literal_pred = - literal_pred.predictions

    # mask_pred_out = trainer.predict(mask_ds)
    # mask_pred = - mask_pred_out.predictions

    df = ds.to_pandas()
    # df['frames'] = np.argmax(pred, axis=-1)[np.arange(len(df.index)), df['tokenized_taregt_word_index'].values]
    # df['literal_frame'] = np.argmax(literal_pred, axis=-1)[np.arange(len(df.index)), 1]

    df['frames'] = np.argsort(pred[np.arange(len(df.index)), df['tokenized_taregt_word_index'].values])[:, :3].tolist()
    df['frames'] = df['frames'].apply(lambda x:[id2label[i] for i in x])

    df['literal_frames'] = np.argsort(literal_pred[np.arange(len(df.index)), 1])[:, :3].tolist()
    df['literal_frames'] = df['literal_frames'].apply(lambda x:[id2label[i] for i in x])

    # df['mask_frames'] = np.argsort(mask_pred[np.arange(len(df.index)), df['tokenized_taregt_word_index'].values])[:, :3].tolist()
    # df['mask_frames'] = df['mask_frames'].apply(lambda x:[id2label[i] for i in x])

    file = open(prediction_output_file, 'w', encoding='utf-8')
    for sent_id, group in df.groupby('sent_id'):
        tokens = group.iloc[0]['tokens']
        target_ids = group['word_index'].values
        labels = group['token_level_label'].values
        frames = group['frames'].values
        # mask_frames = group['mask_frames'].values
        literal_frames = group['literal_frames'].values
        words = group['target_word'].values
        id2label_and_frame = {idx:[label, frame, word, literal_f] for idx, label, frame, word, literal_f in zip(target_ids, labels, frames, words, literal_frames)}
        for w_idx, word in enumerate(tokens):
            if w_idx in target_ids:
                is_target = 1
                label = id2label_and_frame[w_idx][0]
                frame = id2label_and_frame[w_idx][1]
                word = id2label_and_frame[w_idx][2]
                literal_f = id2label_and_frame[w_idx][3]
                # mask_f = id2label_and_frame[w_idx][4]

                frame = '\t'.join(frame)
                literal_f = '\t'.join(literal_f)
                # mask_f = '\t'.join(mask_f)
            else:
                is_target = '_'
                label = '_'
                frame = ''
                literal_f  = ''
                # mask_f = ''
            file.write(f"{sent_id}\t{word}\t{is_target}\t{label}\t{frame}\t{literal_f}\n")
        file.write('\n')
    file.close()

    # for index, batch in enumerate(tqdm(dl)):
    #     batch = {k: v.to(model.device) for k, v in batch.items()}
    #     word_index = batch.pop('word_index')
    #     word_ids = batch.pop('words_ids')
    #     mlabels = batch.pop('label')
    #     outputs = model(**batch)
    #     logits = outputs.logits
    #     labels = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)

    #     for words, label, mlabel, word_idx, word_id in zip(batch['input_ids'], labels, mlabels, word_index, word_ids):
    #         words = tokenizer.convert_ids_to_tokens(words)
    #         label = [id2label[i.item()] for i in label]
    #         words = [word.strip('Ġ') for word in words]
    #         word_id = [str(i.item()) if i is not None else str(i) for i in word_id]
    #         file.write('token\t' + '\t'.join(words)+'\n')
    #         file.write('frame\t' + '\t'.join(label)+'\n')
    #         file.write('word_id\t' + '\t'.join(word_id)+'\n')
    #         file.write('target index\t' + str(word_idx.item())+'\n')
    #         file.write('metaphor label\t' + str(mlabel.item())+'\n')

    #     if index>200:
    #         break
    
    # file.close()
        # print(outputs)
        # break