from torch import nn
from transformers import BertModel
import torch


class RelModel(nn.Module):
    def __init__(self, config):
        super(RelModel, self).__init__()
        self.config = config
        self.bert_dim = config.bert_dim

        # Pretrained BERT encoder
        self.bert_encoder = BertModel.from_pretrained("bert-base-cased", cache_dir='./pre_trained_bert')

        # Sentence-level weight learner: [bert_dim] -> [1]
        self.sentence_leaner = nn.Linear(self.bert_dim, 1)

        # Linear layer for triple representation
        self.triple_linear0 = nn.Linear(self.bert_dim * 3, self.bert_dim * 3)

        # Output layer projecting to [rel_num * tag_size]
        self.relation_matrix = nn.Linear(self.bert_dim * 3, self.config.rel_num * self.config.tag_size)

        # Dropout layers
        self.dropout = nn.Dropout(self.config.dropout_prob)
        self.dropout_2 = nn.Dropout(self.config.entity_pair_dropout)

        self.activation = nn.ReLU()

        # Relative position encoder
        self.PP_weight_linear = nn.Linear(self.bert_dim, self.bert_dim)
        self.trans_weight_probability = nn.Sigmoid()

    def get_encoded_text(self, token_ids, mask):
        """
        Encode input tokens using BERT.
        Args:
            token_ids: [batch_size, seq_len]
            mask: [batch_size, seq_len]
        Returns:
            encoded_text: [batch_size, seq_len, bert_dim]
        """
        encoded_text = self.bert_encoder(token_ids, attention_mask=mask)[0]
        return encoded_text

    def Coded_Self_Attention(self, bert_encoded_text):
        """
        Compute relative position-based attention weights.
        Args:
            bert_encoded_text: [batch_size, seq_len, bert_dim]
        Returns:
            position_weight: [batch_size, seq_len, bert_dim]
        """
        encoded_weight = self.PP_weight_linear(bert_encoded_text)                   # [B, L, D]
        trans_position_weight = encoded_weight.permute(0, 2, 1)                     # [B, D, L]
        position_weight = torch.matmul(encoded_weight, trans_position_weight)      # [B, L, L]
        position_weight = torch.matmul(position_weight, encoded_weight)            # [B, L, D]
        position_weight = self.trans_weight_probability(position_weight)           # [B, L, D]
        return position_weight

    def encoding(self, bert_encoded_text, position_weight):
        """
        Fuse BERT features with relative position encoding.
        Args:
            bert_encoded_text: [batch_size, seq_len, bert_dim]
            position_weight: [batch_size, seq_len, bert_dim]
        Returns:
            encoded_text: [batch_size, seq_len, bert_dim]
        """
        encoded_text = bert_encoded_text * position_weight
        return encoded_text

    def encoder(self, token_ids, mask):
        """
        Full encoding pipeline with BERT and relative positional encoding.
        Args:
            token_ids: [batch_size, seq_len]
            mask: [batch_size, seq_len]
        Returns:
            encoded_text: [batch_size, seq_len, bert_dim]
        """
        bert_encoded_text = self.get_encoded_text(token_ids, mask)
        position_weight = self.Coded_Self_Attention(bert_encoded_text)
        encoded_text = self.encoding(bert_encoded_text, position_weight)
        return encoded_text

    def decoder(self, encoded_text, train=True):
        """
        Decode pairwise relation representations from encoded text.
        Args:
            encoded_text: [batch_size, seq_len, bert_dim]
        Returns:
            if train:
                output: [batch_size, tag_size, rel_num, seq_len, seq_len]
            else:
                output: [batch_size, rel_num, seq_len, seq_len]
        """
        batch_size, seq_len, bert_dim = encoded_text.size()

        # Construct head-tail pairs
        head_matrix = encoded_text.unsqueeze(2).repeat(1, 1, seq_len, 1)    # [B, L, L, D]
        tail_matrix = encoded_text.unsqueeze(1).repeat(1, seq_len, 1, 1)    # [B, L, L, D]

        # Sentence-level context matrix (soft outer product)
        sentence = torch.matmul(encoded_text.permute(0, 2, 1), encoded_text)    # [B, D, L]
        sentence = self.sentence_leaner(sentence).permute(0, 2, 1)              # [B, L, 1]
        sentence = sentence.unsqueeze(1).repeat(1, seq_len, seq_len, 1)         # [B, L, L, 1]

        # Concatenate head, tail, and sentence vectors
        triple_matrix = torch.cat((head_matrix, tail_matrix, sentence), dim=-1) # [B, L, L, 3D]

        triple_matrix = self.triple_linear0(triple_matrix)                      # [B, L, L, 3D]
        triple_matrix = self.dropout_2(triple_matrix)
        triple_matrix = self.activation(triple_matrix)

        triple_matrix = self.relation_matrix(triple_matrix)                     # [B, L, L, rel_num * tag_size]

        triple_matrix = triple_matrix.view(batch_size, seq_len, seq_len, self.config.rel_num, self.config.tag_size)

        if train:
            return triple_matrix.permute(0, 4, 3, 1, 2)  # [B, tag_size, rel_num, L, L]
        else:
            return triple_matrix.argmax(dim=-1).permute(0, 3, 1, 2)  # [B, rel_num, L, L]

    def forward(self, data, train=True):
        """
        Forward function for relation extraction.
        Args:
            data: Dict with 'token_ids' and 'mask'
            train: whether in training mode
        Returns:
            output: Tensor with shape [B, tag_size, rel_num, L, L] or [B, rel_num, L, L]
        """
        token_ids = data['token_ids']  # [batch_size, seq_len]
        mask = data['mask']            # [batch_size, seq_len]

        encoded_text = self.encoder(token_ids, mask)  # [B, L, D]
        output = self.decoder(encoded_text, train)
        return output
