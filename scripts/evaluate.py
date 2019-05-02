#from torch.autograd import Variable
#from utils import get_ppl, train_ngram_lm
#from utils import autoencode_sentences
#from models import Seq2Seq, MLP_D, MLP_G, generate

import os
import argparse
import logging
import numpy as np
import torch
from torch import nn
from onmt.utils.parse import ArgumentParser
from onmt.model_builder import build_model
from onmt.translate.generator import TextGenerator
from reference_lm.kenlm_model import KenlmModel
from reference_lm.utils import AttrDict
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


ReferenceLM = KenlmModel


def get_file_lines_count(filepath):
    return sum(1 for line in open(filepath))


def set_missing_args(args):
    if args.forward_ppl:
        # we need to generate sents for training ref lm
        if not args.gen_test_size:
            if not args.test_filepath:
                ValueError("gen_test_size should be set")
            args.gen_test_size = get_file_lines_count(args.test_filepath)
    if args.reverse_ppl:
        if not args.gen_train_size:
            if not args.train_filepath:
                ValueError("gen_train_size should be set")
            args.gen_train_size = get_file_lines_count(args.train_filepath)


def check_params(args):
    if not args.forward_ppl and not args.reverse_ppl and not args.bleu:
        ValueError("There is not metric to compute")
    if args.forward_ppl:
        if not args.reference_lm and not args.train_filepath:
            ValueError("reference_lm or train_filepath should be set")
        if not args.gen_test_size:
            ValueError("gen_test_size should be set for number of sents to generate")
    if args.reverse_ppl:
        if not args.gen_train_size:
            ValueError("gen_train_size should be set to generate train dataset for ref lm")
        if not args.test_filepath:
            ValueError("test_filepath should be set to evaluate reverse ppl")
    if args.bleu:
        if not args.test_filepath:
            ValueError("test_filepath should be set to evaluate bleu score")


def parse_args():
    parser = argparse.ArgumentParser()
    # Options for a language model to evaluate
    parser.add_argument('--autoencoder', type=str, required=True, help='path to .pt file with autoencoder model')
    parser.add_argument('--gan_model', type=str, required=True, help='path to .pt file with gan model')
    # Options for a reference language model
    parser.add_argument('--reference_lm', type=str, help='path to file with a reference model')
    # Options for datasets
    # would be used for training reference lang model. isn't required if reference_lm is passed
    parser.add_argument('--train_filepath', type=str, help='path to train dataset. should be the same as for training lm')
    parser.add_argument('--test_filepath', type=str, help='path to test dataset')
    parser.add_argument('--gen_test_size', type=int, help='number of generated sents for testing with ref lm')
    parser.add_argument('--gen_train_size', type=int, help='number of generated sents for training ref lm')
    # Options for metrics
    parser.add_argument('--forward_ppl', action='store_true', help='compute forward perplexity')
    parser.add_argument('--reverse_ppl', action='store_true', help='compute backward perplexity')
    parser.add_argument('--bleu', action='store_true', help='compute bleu')
    # Other options
    parser.add_argument('--gpu', type=int, default=-1, help='use gpu')
    parser.add_argument('--maxlen', type=int, default=15, help='max len of sents to generate')

    args = parser.parse_args()
    set_missing_args(args)
    check_params(args)

    return args


def load_model(autoencoder_fp, gan_model_fp, gpu):
    logger.info('Loading autoencoder checkpoint from {}'.format(autoencoder_fp))
    ae_checkpoint = torch.load(autoencoder_fp, map_location=lambda storage, loc: storage)
    model_opt = ArgumentParser.ckpt_model_opts(ae_checkpoint["opt"])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    vocab = ae_checkpoint['vocab']

    opt = AttrDict({
        'gpu': gpu,
        'arae': True,
        'model_arae': gan_model_fp
    })
    ae, gan_g, gan_d = build_model(model_opt, opt, vocab, ae_checkpoint)
    ae.eval()
    ae.generator.eval()
    gan_g.eval()
    gan_d.eval()

    return (ae, gan_g, gan_d), vocab


def create_reference_lm(reference_lm, train_filepath):
    if reference_lm:
        logger.debug('Loading reference lm')
        return ReferenceLM(reference_lm)
    logger.debug('Building reference lm')
    return ReferenceLM.build(train_filepath)


def read_sents(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [sent.strip() for sent in f.readlines()]


def dump_scores(scores):
    for key, score in scores.items():
        print("{}: {:.3f}".format(key, score))


### Metrics computation

def generate_sentences(model, vocab, count, maxlen, gpu, batch_size=None):
    opt = AttrDict({'max_length': maxlen, 'gpu': gpu})
    generator = TextGenerator.from_opt(*model, vocab, opt)
    logger.info('Generate {} sents with eval model...'.format(count))
    sents = []
    if batch_size is not None:
        for start_idx in tqdm(range(0, count, batch_size)):
            bsize = min(batch_size, count - start_idx)
            sents += generator.generate(n_sents=bsize)
    else:
        sents = generator.generate(n_sents=count)
    n_sents = len(sents)
    logger.info('{} sents were generated'.format(len(sents)))
    logger.info('Examples:\n    {}'.format('\n    '.join(sents[:3])))
    return sents


def compute_forward_ppl(model, vocab, reference_lm, count, maxlen, gpu):
    logger.info('Compute Forward PPL')
    generated_sents = generate_sentences(model, vocab, count, maxlen, gpu)
    ppl = reference_lm.get_ppl(generated_sents)
    return ppl


def compute_reverse_ppl(model, vocab, count, test_sents, maxlen, gpu):
    logger.info('Compute Reverse PPL')
    batch_size = count if count <= 3000 else 3000  # generation with batches

    generated_sents = generate_sentences(model, vocab, count, maxlen, gpu, batch_size=batch_size)
    # have to change <unk> into <oov> for building kenlm
    generated_sents = [sent.replace('<unk>', '<oov>') for sent in generated_sents]
    ref_gen_model = ReferenceLM.build(generated_sents)
    ppl = ref_gen_model.get_ppl(test_sents)
    return ppl


def compute_bleu(model, test_sents):
    logger.info('Compute BLEU')
    raise NotImplemented


def main(args):
    model, vocab = load_model(args.autoencoder, args.gan_model, args.gpu)
    scores = {}
    test_sents = None
    if args.forward_ppl:
        reference_lm = create_reference_lm(args.reference_lm, args.train_filepath)
        forward_ppl = compute_forward_ppl(model, vocab, reference_lm, args.gen_test_size, args.maxlen, args.gpu)
        scores['forward_ppl'] = forward_ppl
    if args.reverse_ppl:
        test_sents = read_sents(args.test_filepath)
        reverse_ppl = compute_reverse_ppl(model, vocab, args.gen_train_size, test_sents, args.maxlen, args.gpu)
        scores['reverse_ppl'] = reverse_ppl
    if args.bleu:
        test_sents = test_sents if test_sents else read_sents(args.test_filepath)
        bleu = compute_bleu(model, test_sents)
        scores['bleu'] = bleu
    dump_scores(scores)


if __name__ == '__main__':
    args = parse_args()
    main(args)

