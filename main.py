import config
import framework
import argparse
import models
import os
import torch
import numpy as np
import random

from transformers import logging

logging.set_verbosity_error()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 2050
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()

parser.add_argument('--train', action='store_true')
parser.add_argument('--dataset', type=str, default='WebNLG_star', help="NYT_star or WebNLG_star")
parser.add_argument('--batch_size', type=int, default=8, help="NYT:batch_size=6"
                                                              "WebNLg:batch_size=8")
parser.add_argument('--experiment', type=int, default=0, help="Complex experiment.Optional content:"
                                                              "test_split_by_num:"
                                                              "test_triples_1.json:1"
                                                              "test_triples_2.json:2"
                                                              "test_triples_3.json:3"
                                                              "test_triples_4.json:4"
                                                              "test_triples_5.json:5"
                                                              ""
                                                              "test_split_by_type:"
                                                              "test_normal.json:   6"
                                                              "test_epo.json:      7"
                                                              "test_seo.json:      8")

parser.add_argument('--model_name', type=str, default='CS4TE', help='name of the model')
parser.add_argument('--detail', type=bool, default=False, help='Detail verification')

parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--entity_pair_dropout', type=float, default=0.3)
parser.add_argument('--multi_gpu', type=bool, default=False)

parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--test_epoch', type=int, default=1)
parser.add_argument('--train_prefix', type=str, default='train_triples')
parser.add_argument('--dev_prefix', type=str, default='dev_triples')
parser.add_argument('--test_prefix', type=str, default='test_triples')

parser.add_argument('--rel2id', type=str, default='rel2id')
parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('--bert_max_len', type=int, default=200)
parser.add_argument('--period', type=int, default=100)
parser.add_argument('--debug', type=bool, default=False)


args = parser.parse_args()
# tensorboard --logdir=./tensorboard --port 8123
if __name__ == '__main__':
    if args.train:
        con = config.Config(args)

        fw = framework.Framework(con)

        model = {
            'CS4TE': models.RelModel
        }
        print('dataset:', args.dataset)
        print('batch_size:', args.batch_size, 'entity_pair_dropout:', args.entity_pair_dropout)
        print('seed:', seed)

        fw.train(model[args.model_name])
    else:
        con = config.Config(args)

        fw = framework.Framework(con)

        model = {
            'CS4TE': models.RelModel
        }

        model_name = "CS4TE"
        print('dataset:', args.dataset, 'batch_size:', args.batch_size)
        fw.testall(model[args.model_name], model_name)
