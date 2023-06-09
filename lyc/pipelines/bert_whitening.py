from ..utils import get_model, get_tokenizer, get_vectors, transform_and_normalize
from ..model import SentenceEmbeddingModel
import torch
import os
import numpy as np

class BertWhitening:
    def __init__(self, model_type, model_path, max_length=64, \
        n_components=768, kernel_bias_path=None, \
        corpus_for_kernel_computing=None, \
        pool='first_last_avg_pooling', is_zh = False, add_prefix_space = False, **kwargs):
        """[summary]

        Args:
            model_path ([type]): [description]
            max_length (int, optional): [description]. Defaults to 64.
            n_components (int, optional): [description]. Defaults to 768.
            kernel_bias_path ([type], optional): [description]. Defaults to None.
            corpus_for_kernel_computing ([type], optional): 训练kernel和bias需要的语料，纯txt，一行一个句子. Defaults to None.
        """
        self.model = get_model(model_type, model_path, pooling_type = pool, **kwargs)
        self.tokenizer = get_tokenizer(model_path, is_zh=is_zh, add_prefix_space = add_prefix_space)
        self.max_length=max_length
        self.n_components=n_components
        self.pool=pool
        self._get_kernel_and_bias(kernel_bias_path, corpus_for_kernel_computing)
    
    def get_embeddings(self, sents, whitening=False):
        tokenized_sents=self.tokenizer(sents, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            vecs=get_vectors(self.model, tokenized_sents)
        vecs=vecs.cpu().numpy()

        if whitening:
            assert self.enable_whitening
            kernel, bias = self.kernel, self.bias
            kernel=kernel[:, :self.n_components]
            vecs=transform_and_normalize(vecs, kernel, bias)
            return vecs
        vecs=vector_l2_normlize(vecs)
        return vecs
    
    def _get_kernel_and_bias(self, kernel_bias_path, corpus_for_kernel_computing):
        if kernel_bias_path is None:
            self.enable_whitening = False
            return
        
        if corpus_for_kernel_computing is not None:
            self._computing_kernel_and_save(kernel_bias_path, corpus_for_kernel_computing)

        self.kernel=np.load(os.path.join(kernel_bias_path, 'kernel.npy'))
        self.bias=np.load(os.path.join(kernel_bias_path, 'bias.npy'))
        self.enable_whitening = True
    
    def _computing_kernel_and_save(self, kernel_bias_path, corpus_for_kernel_computing):
        with open(corpus_for_kernel_computing, encoding='utf-8') as f:
            sents=f.readlines()
        vecs=self.get_embeddings(sents, whitening=False)
        kernel, bias = compute_kernel_bias([vecs])
        save_kernel_and_bias(kernel, bias, kernel_bias_path)

if __name__=='__main__':
    demo=BertWhitening('bert-base-chinese', kernel_bias_path='kernel_path/')
    print(demo.get_embeddings(['我不知道', '我是猪']))