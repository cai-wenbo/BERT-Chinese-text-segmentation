from transformers import BertPreTrainedModel
from torch import nn
from transformers import BertConfig 
from transformers import BertModel


class BertTagger(BertPreTrainedModel):
    def __init__(self, config):
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.fc = nn.Linear(config.hidden_size, config.num_labels)
        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()


    def forward(self, input_ids, attention_mask = None):
        bert_outputs = self.bert(input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)
     
        #  last_hidden_state shape = (batch_size, sequence_length, hidden_size):
        last_hidden_state = bert_outputs[0]

        #  batched_logits shape = (batch_size, sequence_length, num_labels)
        batched_logits = self.fc(last_hidden_state)


        #  batched_probs shape = (batch_size, sequence_length, num_labels)
        #  batched_probs = self.softmax(batched_logits)

        return batched_logits
