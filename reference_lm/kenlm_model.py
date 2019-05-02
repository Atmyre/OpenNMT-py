import os
import numpy as np
import math
import kenlm
import subprocess
import tempfile


# Wrapper for kenlm
from .ref_model import ReferenceLM


LMPLZ_PATH = '/usr/local/bin/lmplz'
MODEL_PATH = 'knlm.arpa'
COMPRESSING = 50


def dump_sents(sents):
        fp = tempfile.NamedTemporaryFile(mode='w')
        for sent in sents:
            fp.write(sent+'\n')
        return fp, fp.name


class KenlmModel(ReferenceLM):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = kenlm.Model(model_path)

    @classmethod
    def build(cls, sentences):
        '''
        Args:
            sentences: list of sents without <sos>, <eos> or a filepath
        '''
        if isinstance(sentences, list):
            tmp_f, fp = dump_sents(sentences)
        else:
            assert isinstance(sentences, str) and os.path.exists(sentences)
            fp = sentences
        params = {
            'lmplzp': LMPLZ_PATH,
            'N': 5,
            'trainp': fp,
            'modelp': os.path.join(os.getcwd(), MODEL_PATH)
        }
        if COMPRESSING:
            cmd = '{lmplzp} -o {N} < {trainp} > {modelp}'.format(**params)
        else:
            params['s'] = COMPRESSING
            cmd = '{lmplzp} -o {N} -S {s}% < {trainp} > {modelp}'.format(**params)
        subprocess.call(cmd, shell=True)
        print('Built kenlm model, N = {}, path: {}'.format(params['N'], params['modelp']))
        return cls(params['modelp'])

    def get_ppl(self, sentences):
        n_tokens, nll10 = 0, 0
        for sent in sentences:
            sent = sent.strip()
            nll10 += -self.model.score(sent)
            n_tokens += len(sent.split())
        ppl = math.pow(10., nll10 / n_tokens)
        return ppl

