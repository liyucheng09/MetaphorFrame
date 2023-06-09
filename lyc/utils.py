from transformers import (BertModel,
                          BertTokenizer,
                          AutoModel,
                          AutoTokenizer,
                          PreTrainedModel,
                         GPT2Tokenizer,
                         GPT2TokenizerFast,
                         AutoConfig)
from datasets import load_dataset, Dataset as hfds
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score
import os
import pickle
from .data import get_dataloader, SentenceDataset
import random
from hashlib import md5
import requests
from nltk.stem import WordNetLemmatizer
try:
    import deepl
except ImportError:
    pass


def get_model(model_class, model_name, strict=None, **kwargs):
    """
    该方法载入模型。

    Args:
        model_class: 模型对象，支持lyc.model中定义的模型和huggingface支持的模型。
        model_name: base model 的 name 或 path
        cache_dir: （optional） from_pretrained 方法需要的参数
    
    Return：
        model: model object

    """

    if issubclass(model_class, PreTrainedModel):
        if strict is not None:
            model_config = AutoConfig.from_pretrained(model_name, **kwargs)
            model = model_class(model_config)
            model.load_state_dict(model_name)
        else:
            model=model_class.from_pretrained(model_name, **kwargs)
    else:
        raise ValueError()

    if torch.cuda.is_available():
        model.to('cuda')
        if torch.cuda.device_count()>1:
            model=torch.nn.DataParallel(model)
    return model

def get_tokenizer(tokenizer_name, cache_dir=None, is_zh=None, **kwargs):
    """
        Args:
            tokenizer_name: name or path
            cache_dir=None:
            is_zh=None:
    """

    if is_zh:
        tokenizer=BertTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir, **kwargs)
    else:
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir, **kwargs)
    return tokenizer

def get_model_output(model, tokenized_sents):
    """
    将一个list的句子，用dataset和dataloader包装，然后直接获得输出的list。

    Args:
        model:
        tokenized_sents: BatchedEncoding
    
    Return:
        Tensor([batch_size, embedding_dim])

    """
    ds=hfds.from_dict(tokenized_sents)
    dl=get_dataloader(ds, cols=['input_ids', 'attention_mask', 'token_type_ids'])
    results=[]
    for batch in tqdm(dl):
        if torch.cuda.is_available():
            batch=[to_gpu(i) for i in batch]
        output = model(**batch)
        results.append(output)
    return results


def to_gpu(inputs):
    """
    to_gpu

    Args:
        inputs: Tensor / dict([Tensor])
    """
    if isinstance(inputs, dict):
        return {
            k:v.to('cuda') for k,v in inputs.items()
        }
    else:
        return inputs.to('cuda')
        
def compute_kernel_bias(vecs):
    """
    BertWhitening 计算SVD需要的kernel和bias
    """

    vecs=np.concatenate(vecs)
    mean=vecs.mean(axis=0, keepdims=True)
    cov=np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W, -mean

def transform_and_normalize(vecs, kernel, bias):
    """
    BertWhitening 白化向量
    """
    vecs = (vecs + bias).dot(kernel)
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)

def get_optimizer_and_schedule(params, lr, 
            beta=(0.9, 0.999), eps=1e-8, weight_decay=1e-5,
            num_training_steps=None, num_warmup_steps=3000):
    """
    获取optimizer和schedule
    """

    # params=[{'params': [param for name, param in model.named_parameters() if 'sbert' not in name], 'lr': 5e-5},
    # {'params': [param for name, param in model.named_parameters() if 'sbert' in name], 'lr': 1e-3}]
    
    optimizer=AdamW(params, lr=lr, betas=beta, eps=eps, weight_decay=weight_decay)

    if num_training_steps is None:
        return optimizer

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    lr_schedule=LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    return optimizer, lr_schedule

def eval(model, tokenizer, ds='atec', n_components=768):
    model.eval()
    input_a, input_b, label = get_tokenized_ds(datasets_paths[ds]['scripts'], datasets_paths[ds]['data_path'], tokenizer, ds)

    with torch.no_grad():
        a_vecs, b_vecs = get_vectors(model, input_a, input_b)
    a_vecs=a_vecs.cpu().numpy()
    b_vecs=b_vecs.cpu().numpy()
    if n_components:
        kernel, bias = compute_kernel_bias([a_vecs, b_vecs])

        kernel=kernel[:, :n_components]
        a_vecs=transform_and_normalize(a_vecs, kernel, bias)
        b_vecs=transform_and_normalize(b_vecs, kernel, bias)
        sims=(a_vecs * b_vecs).sum(axis=1)
    else:
        sims=(a_vecs * b_vecs).sum(axis=1)

    return accuracy_score(sims>0.5, label)

def save_kernel_and_bias(kernel, bias, model_path):
    """
    BertWhitening 保存SVD需要的kernel和bias
    """
    np.save(os.path.join(model_path, 'kernel.npy'), kernel)
    np.save(os.path.join(model_path, 'bias.npy'), bias)
    
    print(f'Kernal and bias saved in {os.path.join(model_path, "kernel.npy")} and {os.path.join(model_path, "bias.npy")}')

def vector_l2_normlize(vecs):
    if isinstance(vecs, np.ndarray):
        norms = np.sqrt((vecs**2).sum(axis=1, keepdims=True))
        return vecs/np.clip(norms, 1e-8, np.inf) 
    elif isinstance(vecs, torch.Tensor):
        norms = torch.sqrt((vecs**2).sum(dim=1, keepdims=True))
        return vecs/ torch.clamp(norms, 1e-8, np.inf)
    else:
        raise NotImplementedError()

def get_pl_callbacks(args):
    """
    载入pytorch_lightning.callbacks

    Return:
        EarlyStoppingCallbacks
        LearningRateMonitor

    TODO
    """

    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

    return {
        'checkpoint': ModelCheckpoint(
            dirpath = args.save_path,
            filename = args.prefix + '-{epoch}-{train_loss:.2f}',
        ),
        'lr_monitor': LearningRateMonitor()
    }

def subtokenizer_of_gpt(tokenizer):
    return issubclass(tokenizer.__class__, GPT2Tokenizer) or issubclass(tokenizer.__class__, GPT2TokenizerFast)

class BaiduTranslator:
    """
    返回格式
        {
            "from": "zh",
            "to": "en",
            "trans_result": [
                {
                    "src": "读书是在我们的生命里，不断增长的我们的朋友！",
                    "dst": "Reading is our growing friend in our life!"
                },
                {
                    "src": "读书是一种受益，读过书的人将有所补偿；读不读书的人将陷于穷困。",
                    "dst": "Reading is a benefit, and those who have read books will be compensated; Those who can't read will fall into poverty."
                }
            ]
        }
    """
    def __init__(self):
        
        self.appid = '20210429000807616'
        self.appkey = 'ycRDNKgTtcNp8TiCMEin'

        endpoint = 'http://api.fanyi.baidu.com'
        path = '/api/trans/vip/translate'
        self.url = endpoint + path
    
    def make_md5(self, s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()

    def make_request(self, query, from_lang, to_lang):
        salt = random.randint(32768, 65536)
        sign = self.make_md5(self.appid + query + str(salt) + self.appkey)
        
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {'appid': self.appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}   

        r = requests.post(self.url, params=payload, headers=headers)
        print(r.status_code)
        result = r.json() 

        # return result['trans_result'][0]['dst']
        return result

class DeepLTranslator:
    """
    DeepL翻译的文档：https://www.deepl.com/docs-api
    本实现使用DeepL提供的Python官方翻译库访问API
    Python API文档：https://github.com/DeepLcom/deepl-python
    """
    def __init__(self):

        self.auth_key = 'ead2820d-1d46-e74c-da3c-8ed74471d93f:fx'
        self.translator = deepl.Translator(self.auth_key)

        endpoint = 'http://api.fanyi.baidu.com'
        path = '/api/trans/vip/translate'
        self.url = endpoint + path
    
    def translate(self, list_of_text: list, target_lang, source_lang, split_sentences='off'):
        return self.translator.translate_text(list_of_text, 
            target_lang=target_lang, source_lang=source_lang, split_sentences=split_sentences)

def get_vectors(model, tokenized_sentences, idxs = None, batch_size = 32):
    ds = SentenceDataset(tokenized_sentences, idxs = idxs)
    dl = DataLoader(ds, batch_size=batch_size)
    a_results = []

    for batch in tqdm(dl, desc='Vectorizing: '):
        if torch.cuda.is_available():
            batch=[to_gpu(i) for i in batch]
        input_a = batch[0]
        if idxs is not None:
            idxs = batch[1]
        output = model(idxs = idxs, **input_a)
        a_results.append(output)
    
    output=torch.cat(a_results)
    return output

class lemmatizer:
    """
        Arg:
            pos: `"n"` for nouns,
            `"v"` for verbs, `"a"` for adjectives, `"r"` for adverbs and `"s"`
            for satellite adjectives.
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    def __call__(self, word, pos = 'v'):
        return self.lemmatizer.lemmatize(word, pos=pos)
