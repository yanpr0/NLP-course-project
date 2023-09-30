from transformers import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch
from typing import Optional


class BertGram(nn.Module):
    def __init__(self,
                 bert,
                 gram_model_path):
        super(BertGram, self).__init__()

        self.num_labels = 2

        self.bert = bert
        self.gram = BertModel.from_pretrained(gram_model_path)
        bert_dropout = (
            self.bert.config.classifier_dropout if self.bert.config.classifier_dropout is not None else self.bert.config.hidden_dropout_prob
        )
        self.bert_dropout = nn.Dropout(bert_dropout)

        bert_dropout = (
            self.gram.config.classifier_dropout if self.gram.config.classifier_dropout is not None else self.gram.config.hidden_dropout_prob
        )
        self.gram_dropout = nn.Dropout(bert_dropout)

        self.classifier = nn.Linear(self.bert.config.hidden_size + self.gram.config.hidden_size, self.num_labels)


    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                gram_ids: Optional[torch.Tensor] = None):
        return_dict = return_dict if return_dict is not None else self.bert.config.use_return_dict

        bert_outputs = self.bert(
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

        bert_pooled_output = bert_outputs[1]

        bert_pooled_output = self.bert_dropout(bert_pooled_output)

        return_dict = return_dict if return_dict is not None else self.cgram.onfig.use_return_dict

        gram_outputs = self.gram(
            gram_ids
        )

        gram_pooled_output = gram_outputs[1]

        gram_pooled_output = self.gram_dropout(gram_pooled_output)

        pooled_output = torch.cat((bert_pooled_output, gram_pooled_output), dim=-1)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
              loss_fct = nn.CrossEntropyLoss()
              loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + bert_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
        )

