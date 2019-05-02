import os
import subprocess
import argparse
import logging
import codecs
import numpy as np
import torch
from torch import nn
import onmt
from onmt.utils.parse import ArgumentParser
from onmt.model_builder import build_base_model
from onmt.translate.generator import TextGenerator
from onmt.translate.translator import Translator
from onmt.utils.misc import split_corpus
from reference_lm.kenlm_model import KenlmModel
from reference_lm.utils import AttrDict
from tqdm import tqdm
import tempfile

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


ReferenceLM = KenlmModel


def get_file_lines_count(filepath):
    return sum(1 for line in open(filepath))


def get_file_head(filepath):
    lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, start=1):
            lines.append(line.strip())
            if idx == 5:
                break
    return lines


def show_file_head(filepath):
    lines = get_file_head(filepath)
    logger.debug('\n  '+'\n  '.join(lines))


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
    parser.add_argument('--train_generated_sentences', type=str, help='path to train generated sentences')

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
    ae, gan_g, gan_d = build_base_model(model_opt, vocab, int(gpu > -1), checkpoint=ae_checkpoint,
                                        gpu_id=gpu, arae_setting=True, arae_model_path=gan_model_fp)
    ae.generator.eval()
    gan_g.eval()
    gan_d.eval()

    return (ae, gan_g, gan_d), vocab, model_opt


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


def compute_reverse_ppl(model, vocab, count, test_sents, maxlen, gpu, train_generated_sentences_path=None):
    logger.info('Compute Reverse PPL')
    batch_size = count if count <= 3000 else 3000  # generation with batches

    generated_sents = generate_sentences(model, vocab, count, maxlen, gpu, batch_size=batch_size)
    # have to change <unk> into <oov> for building kenlm
    generated_sents = [sent.replace('<unk>', '<oov>') for sent in generated_sents]

    if train_generated_sentences_path:
        with open(train_generated_sentences_path, 'w', encoding='utf-8') as f:
            for sent in generated_sents:
                f.write(sent+'\n')
        print('Dumped generated sentences: {}'.format(train_generated_sentences_path))

    ref_gen_model = ReferenceLM.build(generated_sents)
    ppl = ref_gen_model.get_ppl(test_sents)
    return ppl


def compute_bleu(model, vocab, model_opt, test_sents, gpu):
    logger.info('Compute BLEU')
    inp_fp = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8')
    for sent in test_sents:
        inp_fp.write(sent+'\n')
    inp_fp.flush()
    logger.debug(':::TEST:::')
    show_file_head(inp_fp.name)

    opt = AttrDict({
        'gpu': gpu,
        'n_best': 1,
        'min_length': 0, 'max_length': 100,
        'ratio': 0.,
        'beam_size': 1,
        'random_sampling_topk': 1, 'random_sampling_temp': 1,
        'stepwise_penalty': False, 'dump_beam': '',
        'block_ngram_repeat': 0, 'ignore_when_blocking': set(),
        'replace_unk': False, 'phrase_table': '',
        'data_type': 'text', 'verbose': False, 'report_bleu': False,
        'report_rouge': False, 'report_time': False, 'seed': 829,

        'alpha': 0.0, 'beta': 0.0, 'length_penalty': 'none',
        'coverage_penalty': 'none'
    })

    fp = tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8')
    scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)
    translator = Translator.from_opt(model[0], vocab,
        opt, model_opt,
        out_file=fp,  # should dump into a file cause of bleu script
        global_scorer=scorer,
        report_score=False
    )

    src_shards = split_corpus(inp_fp.name, shard_size=32)
    logger.info("Translating...")
    for src_shard in tqdm(src_shards):
        translator.translate(src=src_shard, batch_size=30)
    logger.debug(':::PRED:::')
    show_file_head(fp.name)

    cmd = 'perl ./multi-bleu.perl {} < {}'.format(inp_fp.name, fp.name)
    out = subprocess.check_output(cmd, shell=True).decode("utf-8")
    return float(out)


def main(args):
    model, vocab, model_opt = load_model(args.autoencoder, args.gan_model, args.gpu)
    scores = {}
    test_sents = None
    if args.forward_ppl:
        reference_lm = create_reference_lm(args.reference_lm, args.train_filepath)
        forward_ppl = compute_forward_ppl(model, vocab, reference_lm, args.gen_test_size, args.maxlen, args.gpu)
        scores['forward_ppl'] = forward_ppl
    if args.reverse_ppl:
        test_sents = read_sents(args.test_filepath)
        reverse_ppl = compute_reverse_ppl(model, vocab, args.gen_train_size, test_sents, args.maxlen, args.gpu,
                                          train_generated_sentences_path=args.train_generated_sentences)
        scores['reverse_ppl'] = reverse_ppl
    if args.bleu:
        test_sents = test_sents if test_sents else read_sents(args.test_filepath)
        bleu = compute_bleu(model, vocab, model_opt, test_sents, args.gpu)
        scores['bleu'] = bleu
    dump_scores(scores)


if __name__ == '__main__':
    args = parse_args()
    main(args)

