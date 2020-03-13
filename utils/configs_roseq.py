import os
from utils.data_utils import RAW_DICT, EMB_PATH, EMB_NAME


class Configurations:
    def __init__(self, parser):
        self.use_gpu, self.gpu_idx, self.random_seed = parser.use_gpu, parser.gpu_idx, parser.random_seed
        self.log_level, self.train, self.dev_for_train = parser.log_level, parser.train, parser.dev_for_train
        self.language, self.iobes, self.label_weight = parser.language, parser.iobes, parser.label_weight
        self.word_lowercase, self.char_lowercase = parser.word_lowercase, parser.char_lowercase
        self.word_threshold, self.char_threshold = parser.word_threshold, parser.char_threshold
        self.use_orthographic, self.ortho_word_threshold = parser.use_orthographic, parser.ortho_word_threshold
        self.at, self.epsilon = parser.at, parser.epsilon
        self.word_dim, self.ortho_word_dim = parser.word_dim, parser.ortho_word_dim
        self.char_dim, self.ortho_char_dim = parser.char_dim, parser.ortho_char_dim
        self.word_project, self.tune_emb = parser.word_project, parser.tune_emb
        self.char_kernels, self.char_kernel_features = parser.char_kernels, parser.char_kernel_features
        self.highway_layers, self.focal_loss = parser.highway_layers, parser.focal_loss
        self.num_units, self.concat_rnn = parser.num_units, parser.concat_rnn
        self.lr, self.use_lr_decay, self.lr_decay = parser.lr, parser.use_lr_decay, parser.lr_decay
        self.decay_step, self.minimal_lr, self.optimizer = parser.decay_step, parser.minimal_lr, parser.optimizer
        self.grad_clip, self.epochs, self.batch_size = parser.grad_clip, parser.epochs, parser.batch_size
        self.emb_drop_rate, self.rnn_drop_rate = parser.emb_drop_rate, parser.rnn_drop_rate
        self.max_to_keep, self.no_imprv_tolerance = parser.max_to_keep, parser.no_imprv_tolerance
        # restore best model to test
        self._dataset_config()

    def _dataset_config(self):
        # model name
        self.model_name = "adv_model" if self.at else "base_model"
        self.model_name = self.model_name + "_ortho" if self.use_orthographic else self.model_name
        self.model_name = self.model_name + "_focal" if self.focal_loss else self.model_name
        self.model_name = self.model_name + "_{}".format(self.language)
        # raw dataset
        r_path = "datasets/raw/{}/".format(RAW_DICT[self.language])
        self.train_file, self.dev_file, self.test_file = r_path + "train.txt", r_path + "valid.txt", r_path + "test.txt"
        self.save_path = "datasets/data/roseq/{}/".format(self.model_name)
        self.train_set, self.dev_set = self.save_path + "train.json", self.save_path + "dev.json"
        self.test_set, self.vocab = self.save_path + "test.json", self.save_path + "vocab.json"
        self.word_weight, self.char_weight = self.save_path + "word_weight.npz", self.save_path + "char_weight.npz"
        if self.language.lower() == "english":
            self.word_vec_path = os.path.join(os.path.expanduser("~"), "utilities", "embeddings", "glove",
                                              "glove.6B.{}d.txt".format(self.word_dim))
        else:
            self.word_vec_path = os.path.join(EMB_PATH, EMB_NAME[self.language].format(self.word_dim))
        self.word_vec = self.save_path + "word_vec.npz"
        self.checkpoint_path = "ckpt/roseq/{}/".format(self.model_name)
        self.summary_path = self.checkpoint_path + "summary/"
