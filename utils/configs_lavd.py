import os


class Configurations:
    def __init__(self, cfg):
        self.use_gpu, self.gpu_idx, self.random_seed = cfg.use_gpu, cfg.gpu_idx, cfg.random_seed
        self.log_level, self.iobes, self.train, self.restore = cfg.log_level, cfg.iobes, cfg.train, cfg.restore
        self.word_lowercase, self.char_lowercase = cfg.word_lowercase, cfg.char_lowercase
        self.word_threshold, self.char_threshold = cfg.word_threshold, cfg.char_threshold
        self.word_dim, self.char_dim, self.tune_emb = cfg.word_dim, cfg.char_dim, cfg.tune_emb
        self.word_project = cfg.word_project
        self.char_kernels, self.char_kernel_features = cfg.char_kernels, cfg.char_kernel_features
        self.highway_layers, self.num_units, self.concat_rnn = cfg.highway_layers, cfg.num_units, cfg.concat_rnn
        self.emb_drop_rate, self.rnn_drop_rate = cfg.emb_drop_rate, cfg.rnn_drop_rate
        self.lr, self.use_lr_decay, self.lr_decay = cfg.lr, cfg.use_lr_decay, cfg.lr_decay
        self.decay_step, self.minimal_lr, self.optimizer = cfg.decay_step, cfg.minimal_lr, cfg.optimizer
        self.grad_clip, self.epochs, self.batch_size = cfg.grad_clip, cfg.epochs, cfg.batch_size
        self.max_to_keep, self.no_imprv_tolerance = cfg.max_to_keep, cfg.no_imprv_tolerance
        if cfg.mode == 0:
            self._base_configs(cfg)
        elif cfg.mode == 1:
            self._dcrf_configs(cfg)
        else:
            raise ValueError("Unknown mode, only support [0: base model, 1: dCRF model]!!!")

    def _base_configs(self, cfg):
        self.model_name, r_path = "base_model_{}".format(cfg.task), cfg.raw_path
        self.train_file, self.dev_file, self.test_file = r_path + "train.txt", r_path + "valid.txt", r_path + "test.txt"
        self.save_path = cfg.save_path.format(self.model_name)
        self.train_set, self.dev_set = self.save_path + "train.json", self.save_path + "dev.json"
        self.test_set, self.vocab = self.save_path + "test.json", self.save_path + "vocab.json"
        self.wordvec_path = os.path.join(os.path.expanduser('~'), "utilities", "embeddings", "glove",
                                         cfg.wordvec_path.format(self.word_dim))
        self.wordvec = self.save_path + "wordvec.npz"
        self.checkpoint_path = "ckpt/lavd/{}/".format(self.model_name)
        self.summary_path = self.checkpoint_path + "summary/"

    def _dcrf_configs(self, cfg):
        self.folds, self.partial_rate, self.loss_method = cfg.folds, cfg.partial_rate, cfg.loss_method
        self.labeled_range, self.partial_range = cfg.labeled_range, cfg.partial_range
        self.unlabeled_range = cfg.unlabeled_range
        self.model_name, r_path = "dCRF_model_{}".format(cfg.task), cfg.raw_path
        self.train_file, self.dev_file, self.test_file = r_path + "train.txt", r_path + "valid.txt", r_path + "test.txt"
        self.save_path = cfg.save_path.format(self.model_name)
        self.train_set, self.dev_set = self.save_path + "train.json", self.save_path + "dev.json"
        self.test_set, self.vocab = self.save_path + "test.json", self.save_path + "vocab.json"
        self.train_set_p, self.train_set_u = self.save_path + "train_p.json", self.save_path + "train_u.json"
        self.wordvec_path = os.path.join(os.path.expanduser('~'), "utilities", "embeddings", "glove",
                                         cfg.wordvec_path.format(self.word_dim))
        self.wordvec = self.save_path + "wordvec.npz"
        self.checkpoint_path = "ckpt/lavd/{}/".format(self.model_name)
        self.summary_path = self.checkpoint_path + "summary/"
