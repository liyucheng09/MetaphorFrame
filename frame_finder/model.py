from transformers import RobertaForTokenClassification, DataCollatorForTokenClassification, Trainer, BertForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
import torch

class FrameFinder(RobertaForTokenClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent_classifier=torch.nn.Linear(self.config.hidden_size, self.config.num_labels)
        # self.pooler=torch.nn.Linear(self.config.hidden_size, self.config.hidden_size)
        # self.pooler_activation = torch.nn.Tanh()
        self._init_weights(self.sent_classifier)
        # self._init_weights(self.pooler)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        sent_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        cls_embedding = sequence_output[:, 0]
        # pooled_output = self.pooler(cls_embedding)
        # pooled_output = self.pooler_activation(pooled_output)
        sent_logits = self.sent_classifier(cls_embedding)

        logits = self.classifier(sequence_output)
        logits[:, 0] = sent_logits

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if sent_labels is not None:
                sent_labels = sent_labels.float()
                loss_bcel = torch.nn.BCEWithLogitsLoss(pos_weight=(sent_labels*157)+1)
                loss_sent = loss_bcel(logits[:, 0], sent_labels)
                loss += 1 * loss_sent
                # loss = loss_sent

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class DataCollator(DataCollatorForTokenClassification):

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]

        batch = {k: torch.tensor(v, dtype=torch.int64) if k !='sent_labels' else v for k, v in batch.items()}
        return batch

# class FrameTrainer(Trainer):

