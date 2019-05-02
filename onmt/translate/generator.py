#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math
import time
from itertools import count

import torch

import onmt.model_builder
import onmt.translate.beam
import onmt.inputters as inputters
import onmt.decoders.ensemble
from onmt.translate.beam_search import BeamSearch
from onmt.translate.random_sampling import RandomSampling
from onmt.utils.misc import tile, set_random_seed
from onmt.modules.copy_generator import collapse_copy_scores

from onmt.gans.gan import MLP_D, MLP_G


def build_generator(opt):
    assert opt.arae
    out_file = codecs.open(opt.output, 'w+', 'utf-8')

    load_test_model = onmt.model_builder.load_test_model
    fields, model, model_opt = load_test_model(opt)
    model, gan_g, gan_d = model

    generator = TextGenerator.from_opt(model, gan_g, gan_d, fields, opt)
    return generator


def sample_next_idxs(log_probs):
    ARG_MAX = True

    if ARG_MAX:
        return torch.argmax(log_probs, dim=1, keepdim=True)

    sampling_temp = 1.5
    log_probs = torch.div(log_probs, sampling_temp)

    dist = torch.distributions.Multinomial(logits=log_probs, total_count=1)
    topk_ids = torch.argmax(dist.sample(), dim=1, keepdim=True)

    return topk_ids


class TextGenerator(object):
    def __init__(self, model, gan_g, gan_d, fields, max_length, gpu):
        self._gpu = gpu
        self._use_cuda = gpu > -1
        self.device = torch.device("cuda", self._gpu) if self._use_cuda else torch.device("cpu")

        self.model = model
        self.gan_g = gan_g
        self.gan_d = gan_d
        self.fields = fields
        self.max_length = max_length

        self.tgt_field = dict(self.fields)["tgt"].base_field
        self.tgt_vocab = self.tgt_field.vocab


    @classmethod
    def from_opt(cls, model, gan_g, gan_d, fields, opt):
        return cls(model, gan_g, gan_d, fields, opt.max_length, gpu=opt.gpu)

    def generate(self, n_sents):
        with torch.no_grad():
            return self._generate(n_sents)

    def _generate(self, n_sents):
        batch_size = n_sents
        # get Z
        z_hidden_size = self.gan_g.ninput
        noise = torch.Tensor(n_sents, z_hidden_size).normal_(0, 1).to(self.device)
        fake_hidden = self.gan_g(noise)

        memory_bank = torch.zeros(self.max_length, n_sents, 512).to(self.device)  # 512 - internal repr
        memory_bank[0] = fake_hidden

        self.model.decoder.init_state(None, memory_bank, None)

        BOS = 2  # hardcoded
        reconstruct_seq = torch.full([batch_size, 1], BOS, dtype=torch.long, device=self.device).to(self.device)

        for step in range(self.max_length):
            decoder_in = reconstruct_seq[:, -1].view(1, -1, 1)  # (1, B, 1)
            dec_out, attns = self.model.decoder(decoder_in, memory_bank, memory_lengths=None, step=step)

            # copy_attn is OFF
            attn = attns.get('std', None)

            log_probs = self.model.generator(dec_out.squeeze(0))
            next_ids = sample_next_idxs(log_probs)

            reconstruct_seq = torch.cat([reconstruct_seq, next_ids], -1)

        tokens = self._convert_idxs_to_tokens(reconstruct_seq)
        # filter <eos> after the first one
        eos_token = self.tgt_field.eos_token

        def get_slice_idx(tokens_line):
            if eos_token not in tokens_line:
                return len(tokens_line)
            return tokens_line.index(eos_token)+1

        tokens = [tokens_line[:get_slice_idx(tokens_line)] for tokens_line in tokens]
        # chop bos eos in addition
        sents = [' '.join(token_line[1:-1]) for token_line in tokens]
        return sents

    def _convert_idxs_to_tokens(self, idxs):
        idx2token = lambda idx: self.tgt_vocab.itos[idx]
        idxs_local = idxs.data.cpu().numpy()
        tokens = []
        for idxs_line in idxs_local:
            tokens.append(list(map(idx2token, idxs_line)))
        return tokens

