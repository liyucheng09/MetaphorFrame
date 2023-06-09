import torch.nn as nn
from transformers import (
    BertModel,
    AutoModel,
    PreTrainedModel,
    RobertaModel
)
import torch
import torch.nn.functional as F


class SentenceEmbeddingModel(RobertaModel):
    """
    从预训练语言模型里获取向量。可以经过pooling获取句向量，也可以直接提供idx得到某个token的向量。

    Args: 
        model_name
        pooling_type: pooling的方法
    """

    def __init__(self, *args, pooling_type=None, **kwargs):
        super(SentenceEmbeddingModel, self).__init__(*args, **kwargs)
        self.pooling_type=pooling_type
        # self.base=AutoModel.from_pretrained(model_name_or_path, output_hidden_states=True, **kwargs)

        self.pooling_funcs={
            'cls': self._cls_pooling,
            'last-average': self._average_pooling,
            'first-last-average': self._first_last_average_pooling,
            'cls-pooler': self._cls_pooler_pooling,
            'hidden': self._return_hidden,
            'idx-last': self._idx_last,
            'idx-first-last-average': self._idx_first_last_average,
            'idx-all-average': self._idx_all_average,
            'idx-first': self._idx_first,
            'idx-last-four-average': self._idx_last_four_average
        }

        self.pools_for_idx = ['idx-last', 'idx-first-last-average', 'idx-all-average', 'idx-first', 'idx-last-four-average']
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, idxs = None, **kwargs):

        outputs=super(SentenceEmbeddingModel, self).forward(input_ids, attention_mask, token_type_ids, **kwargs)
        # outputs=self.base(input_ids, attention_mask, token_type_ids, **kwargs)

        hidden=outputs.hidden_states
        pooler_outputs=outputs.pooler_output

        if idxs is not None: 
            assert self.pooling_type in self.pools_for_idx, f"when idxs are provides, you must choose pooling funcs designed for embedding idxing, {self.pooling_type} is chosen!"
            return self.pooling_funcs[self.pooling_type](hidden, attention_mask, pooler_outputs, idxs = idxs)
        
        return self.pooling_funcs[self.pooling_type](hidden, attention_mask, pooler_outputs)

    def _idx_last(self, hidden, attention_mask, pooled_outputs, idxs):
        return hidden[-1][torch.arange(idxs.size(0)).type_as(idxs), idxs.squeeze()]

    def _idx_first(self, hidden, attention_mask, pooled_outputs, idxs):
        return hidden[1][torch.arange(idxs.size(0)).type_as(idxs), idxs.squeeze()]
    
    def _idx_all_average(self, hidden, attention_mask, pooled_outputs, idxs):
        line_number = torch.arange(idxs.size(0)).type_as(idxs)
        idxs = idxs.squeeze()
        representation_each_layers = [layer[line_number, idxs] for layer in hidden[1:]]
        return torch.mean(
            torch.stack(representation_each_layers), dim=0
        )
    
    def _idx_last_four_average(self, hidden, attention_mask, pooled_outputs, idxs):
        line_number = torch.arange(idxs.size(0)).type_as(idxs)
        idxs = idxs.squeeze()
        representation_each_layers = [layer[line_number, idxs] for layer in hidden[-4:]]
        return torch.mean(
            torch.stack(representation_each_layers), dim=0
        )
    
    def _idx_first_last_average(self, hidden, attention_mask, pooled_outputs, idxs):
        line_number = torch.arange(idxs.size(0)).type_as(idxs)
        idxs = idxs.squeeze(1)
        last = hidden[-1][line_number, idxs]
        first = hidden[1][line_number, idxs]
        return (last+first)/2

    def _return_hidden(self, hidden, attention_mask, pooled_outputs):
        return hidden
    
    def _cls_pooling(self, hidden, attention_mask, pooled_outputs):
        last_hidden_state=hidden[-1]
        return last_hidden_state[:, 0]

    def _cls_pooler_pooling(self, hidden, attention_mask, pooled_outputs):
        return pooled_outputs
    
    def _average_pooling(self, hidden, attention_mask, pooled_outputs):
        last_hidden_states=hidden[-1]
        last_hidden_states = torch.sum(
            last_hidden_states * attention_mask.unsqueeze(-1), dim=1
        ) / attention_mask.sum(dim=-1, keepdim=True)

        return last_hidden_states
    
    def _first_last_average_pooling(self, hidden, attention_mask, pooled_outputs):
        first_hidden_states = hidden[1]
        last_hidden_states = hidden[-1]

        first_hidden_states = torch.sum(
            first_hidden_states * attention_mask.unsqueeze(-1), dim=1
        ) / attention_mask.sum(dim=-1, keepdim=True)

        last_hidden_states = torch.sum(
            last_hidden_states * attention_mask.unsqueeze(-1), dim=1
        ) / attention_mask.sum(dim=-1, keepdim=True)

        sentence_embedding=torch.mean(
            torch.stack([first_hidden_states, last_hidden_states]), dim=0
        )

        return sentence_embedding


class simcse(SentenceEmbeddingModel):

    def forward(self, *args, **kwargs):
        if kwargs['input_ids'].shape[0]==1 and len(kwargs['input_ids'].shape)==3:
            kwargs = { k:v.squeeze(0) for k,v in kwargs.items()}
        label = kwargs.pop('labels')
        return_sims = kwargs.pop('return_sims', None)
        embeddings=super(simcse, self).forward(*args, **kwargs)
        loss, sims = self.cce_losses(label, embeddings)

        results = {'loss': loss}
        if return_sims is not None:
            results['sims'] = sims
        return results
    
    def nll_losses(self, label, embeddings):

        label=F.one_hot(label)
        normalized_embedding = embeddings/torch.sqrt((embeddings**2).sum(-1))[:, None]
        sims=torch.matmul(normalized_embedding, normalized_embedding.T)
        masked_sims = sims - torch.eye(embeddings.shape[0]).type_as(embeddings)*100

        masked_sims.clip_(0,1)
        loss = F.binary_cross_entropy(masked_sims.view(-1), label.view(-1).float())
        
        return loss, sims

    def cce_losses(self, label, embeddings):

        normalized_embedding = embeddings/torch.sqrt((embeddings**2).sum(-1))[:, None]
        sims=torch.matmul(normalized_embedding, normalized_embedding.T)
        masked_sims=sims*20 - torch.eye(embeddings.shape[0]).type_as(embeddings)*1e12

        loss=F.cross_entropy(masked_sims, label)
        return loss, sims
