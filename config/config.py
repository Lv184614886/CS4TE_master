import json
from main import seed


class Config(object):
    def __init__(self, args):
        self.args = args

        self.multi_gpu = args.multi_gpu
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.max_len = args.max_len
        self.bert_max_len = args.bert_max_len
        self.bert_dim = 768
        self.tag_size = 4
        self.entity_pair_dropout = args.entity_pair_dropout
        self.seed = str(seed)

        # dataset
        self.dataset = args.dataset
        self.experiment = args.experiment

        # path and name
        self.data_path = './data/' + self.dataset
        self.checkpoint_dir = './checkpoint/' + self.dataset
        self.log_dir = './log/' + self.dataset
        self.result_dir = './result/' + self.dataset
        self.train_prefix = args.train_prefix
        self.dev_prefix = args.dev_prefix

        if self.experiment != 0:
            # Conduct complex experiments
            # test_split_by_num
            if self.experiment ==1:
                self.test_prefix = "test_split_by_num" + '/test_triples_1'
            elif self.experiment == 2:
                self.test_prefix = "test_split_by_num" + '/test_triples_2'
            elif self.experiment == 3:
                self.test_prefix = "test_split_by_num" + '/test_triples_3'
            elif self.experiment == 4:
                self.test_prefix = "test_split_by_num" + '/test_triples_4'
            elif self.experiment == 5:
                self.test_prefix = "test_split_by_num" + '/test_triples_5'
            # test_split_by_type
            elif self.experiment == 6:
                self.test_prefix = "test_split_by_type" + '/test_triples_normal'
            elif self.experiment == 7:
                self.test_prefix = "test_split_by_type" + '/test_triples_epo'
            elif self.experiment == 8:
                self.test_prefix = "test_split_by_type" + '/test_triples_seo'

        elif self.experiment == 0:
            # No complicated experiments
            self.test_prefix = args.test_prefix

        self.detail = args.detail

        self.rel2id = args.rel2id

        self.model_save_name = args.model_name + '_DATASET_' + self.dataset + "_LR_" + str(
            self.learning_rate) + "_BS_" + str(self.batch_size) + "_seed_" + str(seed) + "_EDP_" + str(self.entity_pair_dropout)
        self.log_save_name = 'LOG_' + args.model_name + '_DATASET_' + self.dataset + "_LR_" + str(
            self.learning_rate) + "_BS_" + str(self.batch_size) + "_seed_" + str(seed) + "_EDP_" + str(self.entity_pair_dropout)
        self.result_save_name = 'RESULT_' + args.model_name + '_DATASET_' + self.dataset + "_LR_" + str(
            self.learning_rate) + "_BS_" + str(self.batch_size) + "_seed_" + str(seed) + "_EDP_" + str(self.entity_pair_dropout) + ".json"

        # log setting
        self.period = args.period
        self.test_epoch = args.test_epoch

        # debug
        self.debug = args.debug
        if self.debug:
            self.dev_prefix = self.train_prefix
            self.test_prefix = self.train_prefix

        def find_nums():
            rel_num = 0
            with open(self.data_path + '/' + self.rel2id + '.json', 'r') as json_file:
                data = json.load(json_file)
                rel_num = len(data[0])
            return rel_num

        self.rel_num = find_nums()
