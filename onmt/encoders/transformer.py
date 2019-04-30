"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn

from torch.autograd import Variable

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward


class AttentionPooling(nn.Module):
    def __init__(self, enc_size, attn_size, dropout_rate=0.0):
        super().__init__()
        self.logits = nn.Sequential(
            nn.Linear(enc_size, attn_size),
            nn.Dropout(dropout_rate),
            nn.Tanh(),
            nn.Linear(attn_size, 1),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inp, mask=None):
        logits = self.logits(inp)[:, :, 0]
        if mask is not None:
            constant = torch.full_like(logits, -1e-9, dtype=torch.float32)
            logits = torch.where(mask, logits, constant)
        probs = self.softmax(logits)
        return torch.sum(inp * probs[:, :, None], dim=1)


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout,
                 max_relative_positions=0):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, src_len, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings,
                 max_relative_positions, arae_setting=False, noise_r=0):
        super(TransformerEncoder, self).__init__()

        self.arae_setting = arae_setting
        self.noise_r = noise_r # arae param

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        if arae_setting:
            self.attention_pooling = AttentionPooling(d_model, d_model)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout,
            embeddings,
            opt.max_relative_positions,
            opt.arae, opt.noise_r)

    def forward(self, src, lengths=None, noise=False):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out)

        if not self.arae_setting:
            return emb, out.transpose(0, 1).contiguous(), lengths

        device = out.device
        if noise and self.noise_r > 0:
            gauss_noise = torch.normal(mean=torch.zeros(out.size()), std=self.noise_r)
            out = out + gauss_noise.to(device)

        #out_new = torch.zeros_like(out)
        # comment for a while
        #out_new[:, 0, :] = self.attention_pooling(out, mask[:, 0, :])
        #out = out_new.transpose(0, 1)
        out = out.transpose(0, 1)

        token_mask = torch.zeros_like(out).to(device)
        token_mask[0] = 1.
        # size: (src_len, batch_size, model_dim)
        return emb, out.contiguous() * token_mask, lengths

