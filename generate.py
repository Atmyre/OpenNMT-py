#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.generator import build_generator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def dump_sent(sents):
    for sent in sents:
        print(sent)


def main(opt):
    ArgumentParser.validate_generate_opts(opt)

    generator = build_generator(opt)
    sents = generator.generate(n_sents=10)
    dump_sent(sents)


def _get_parser():
    parser = ArgumentParser(description='generate.py')

    opts.config_opts(parser)
    opts.generate_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)

