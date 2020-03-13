import tensorflow as tf
import numpy as np
from lavd.shares import BaseModel, BiRNN, CharCNNHW, CRF, Embedding
from utils.logger import Progbar


class BiLSTMCRFModel(BaseModel):
    def __init__(self, config):
        super(BiLSTMCRFModel, self).__init__(config)
        self._init_configs()
        with tf.Graph().as_default():
            self._add_placeholders()
            self._build_model()
            self.logger.info("total params: {}".format(self.count_params()))
            self._initialize_session()

    def _init_configs(self):
        vocab = self.load_dataset(self.cfg.vocab)
        self.word_dict, self.char_dict, self.label_dict = vocab["word_dict"], vocab["char_dict"], vocab["label_dict"]
        del vocab
        self.word_size, self.char_size, self.label_size = len(self.word_dict), len(self.char_dict), len(self.label_dict)
        self.rev_word_dict = dict([(idx, word) for word, idx in self.word_dict.items()])
        self.rev_char_dict = dict([(idx, char) for char, idx in self.char_dict.items()])
        self.rev_label_dict = dict([(idx, tag) for tag, idx in self.label_dict.items()])

    def _get_feed_dict(self, data, is_train=False, lr=None):
        feed_dict = {self.words: data["words"], self.seq_len: data["seq_len"], self.chars: data["chars"],
                     self.char_seq_len: data["char_seq_len"]}
        if "labels" in data:
            feed_dict[self.labels] = data["labels"]
        feed_dict[self.is_train] = is_train
        if lr is not None:
            feed_dict[self.lr] = lr
        return feed_dict

    def _add_placeholders(self):
        self.words = tf.placeholder(tf.int32, shape=[None, None], name="words")
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len")
        self.chars = tf.placeholder(tf.int32, shape=[None, None, None], name="chars")
        self.char_seq_len = tf.placeholder(tf.int32, shape=[None, None], name="char_seq_len")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        # hyper-parameters
        self.is_train = tf.placeholder(tf.bool, shape=[], name="is_train")
        self.lr = tf.placeholder(tf.float32, name="learning_rate")

    def _build_model(self):
        with tf.variable_scope("embeddings_op"):
            # word table
            word_table = Embedding(self.word_size, self.cfg.word_dim, self.cfg.wordvec, self.cfg.tune_emb,
                                   self.cfg.word_project, scope="word_table")
            word_emb = word_table(self.words)
            # char table
            char_table = Embedding(self.char_size, self.cfg.char_dim, None, True, False, scope="char_table")
            char_emb = char_table(self.chars)

        with tf.variable_scope("computation_graph"):
            # create module
            emb_dropout = tf.layers.Dropout(rate=self.cfg.emb_drop_rate)
            rnn_dropout = tf.layers.Dropout(rate=self.cfg.rnn_drop_rate)
            char_tdnn_hw = CharCNNHW(self.cfg.char_kernels, self.cfg.char_kernel_features, self.cfg.char_dim,
                                     self.cfg.highway_layers, padding="VALID", activation=tf.nn.tanh, use_bias=True,
                                     hw_activation=tf.nn.tanh, reuse=False, scope="char_tdnn_hw")
            bi_rnn = BiRNN(self.cfg.num_units, concat=self.cfg.concat_rnn, reuse=tf.AUTO_REUSE, scope="bi_rnn")
            crf_layer = CRF(self.label_size, reuse=False, scope="crf")

            # compute logits
            char_cnn = char_tdnn_hw(char_emb)
            emb = emb_dropout(tf.concat([word_emb, char_cnn], axis=-1), training=self.is_train)
            rnn_outputs, _ = bi_rnn(emb, self.seq_len)
            rnn_outputs = rnn_dropout(rnn_outputs, training=self.is_train)
            self.logits, self.transition, self.loss = crf_layer(rnn_outputs, self.labels, self.seq_len)

        optimizer = self._build_optimizer()
        if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
            grads, vs = zip(*optimizer.compute_gradients(self.loss))
            grads, _ = tf.clip_by_global_norm(grads, self.cfg.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(grads, vs))
        else:
            self.train_op = optimizer.minimize(self.loss)

    def _predict_op(self, data):
        feed_dict = self._get_feed_dict(data)
        logits, transition, seq_len = self.sess.run([self.logits, self.transition, self.seq_len], feed_dict=feed_dict)
        return self.viterbi_decode(logits, transition, seq_len)

    def train(self, dataset):
        self.logger.info("Start training...")
        best_f1, no_imprv_epoch, init_lr, lr, cur_step = -np.inf, 0, self.cfg.lr, self.cfg.lr, 0
        for epoch in range(1, self.cfg.epochs + 1):
            self.logger.info("Epoch {}/{}:".format(epoch, self.cfg.epochs))
            prog = Progbar(target=dataset.get_num_batches())
            for i, data in enumerate(dataset.get_data_batches()):
                cur_step += 1
                feed_dict = self._get_feed_dict(data, is_train=True, lr=lr)
                _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                prog.update(i + 1, [("Global Step", int(cur_step)), ("Train Loss", train_loss)])
            # learning rate decay
            if self.cfg.use_lr_decay:
                if self.cfg.decay_step:
                    lr = max(init_lr / (1.0 + self.cfg.lr_decay * epoch / self.cfg.decay_step), self.cfg.minimal_lr)
            # evaluate
            score = self.evaluate(dataset.get_data_batches("dev"), name="dev")
            self.evaluate(dataset.get_data_batches("test"), name="test")
            if score["FB1"] > best_f1:
                best_f1, no_imprv_epoch = score["FB1"], 0
                self.save_session(epoch)
                self.logger.info(" -- new BEST score on dev dataset: {:04.2f}".format(best_f1))
            else:
                no_imprv_epoch += 1
                if self.cfg.no_imprv_tolerance is not None and no_imprv_epoch >= self.cfg.no_imprv_tolerance:
                    self.logger.info("early stop at {}th epoch without improvement".format(epoch))
                    self.logger.info("best score on dev set: {}".format(best_f1))
                    break

    def evaluate(self, dataset, name):
        all_data = list()
        for data in dataset:
            predicts = self._predict_op(data)
            all_data.append((data["labels"], predicts, data["words"], data["seq_len"]))
        return self.evaluate_f1(all_data, self.rev_word_dict, self.rev_label_dict, name)
