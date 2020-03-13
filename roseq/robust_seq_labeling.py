import tensorflow as tf
import numpy as np
from roseq.shares import Base, BiRNN, CharTDNNHW, focal_loss
from roseq.lb_crf import LBCRF
from utils.logger import Progbar
from utils.data_utils import WEIGHT


class RobustSeqLabelingModel(Base):
    def __init__(self, config):
        super(RobustSeqLabelingModel, self).__init__(config)
        self._init_configs()
        with tf.Graph().as_default():
            self._add_placeholders()
            self._build_model()
            self.logger.info("total params: {}".format(self.count_params()))
            self._initialize_session()

    def _init_configs(self):
        vocab = self.load_dataset(self.cfg.vocab)
        self.word_dict, self.char_dict, self.label_dict = vocab["word_dict"], vocab["char_dict"], vocab["label_dict"]
        self.ortho_word_dict, self.ortho_char_dict = vocab["ortho_word_dict"], vocab["ortho_char_dict"]
        del vocab
        self.word_size, self.char_size, self.label_size = len(self.word_dict), len(self.char_dict), len(self.label_dict)
        self.ortho_word_size, self.ortho_char_size = len(self.ortho_word_dict), len(self.ortho_char_dict)
        self.rev_word_dict = dict([(idx, word) for word, idx in self.word_dict.items()])
        self.rev_char_dict = dict([(idx, char) for char, idx in self.char_dict.items()])
        self.rev_label_dict = dict([(idx, tag) for tag, idx in self.label_dict.items()])

    def _get_feed_dict(self, data, is_train=False, lr=None):
        feed_dict = {self.words: data["words"], self.seq_len: data["seq_len"], self.chars: data["chars"],
                     self.char_seq_len: data["char_seq_len"]}
        if self.cfg.use_orthographic:
            feed_dict[self.ortho_words] = data["ortho_words"]
            feed_dict[self.ortho_chars] = data["ortho_chars"]
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
        if self.cfg.use_orthographic:
            self.ortho_words = tf.placeholder(tf.int32, shape=[None, None], name="ortho_words")
            self.ortho_chars = tf.placeholder(tf.int32, shape=[None, None, None], name="ortho_chars")
        # hyper-parameters
        self.is_train = tf.placeholder(tf.bool, shape=[], name="is_train")
        self.lr = tf.placeholder(tf.float32, name="learning_rate")

    def viterbi_decode(self, logits, trans_params, seq_len):
        viterbi_sequences = []
        for logit, lens in zip(logits, seq_len):
            logit = logit[:lens]  # keep only the valid steps
            viterbi_seq, viterbi_score = self.crf_layer.viterbi_decode(logit, trans_params)
            viterbi_sequences += [viterbi_seq]
        return viterbi_sequences

    def _build_model(self):
        with tf.variable_scope("embeddings_op"):
            # word table
            if self.cfg.word_vec is not None:
                word_table = tf.Variable(initial_value=np.load(self.cfg.word_vec)["embeddings"],
                                         name="word_table", dtype=tf.float32, trainable=self.cfg.tune_emb)
                unk = tf.get_variable(name="unk", shape=[1, self.cfg.word_dim], trainable=True, dtype=tf.float32)
                word_table = tf.concat([unk, word_table], axis=0)
            else:
                word_table = tf.get_variable(name="word_table", shape=[self.word_size - 1, self.cfg.word_dim],
                                             dtype=tf.float32, trainable=True)
            '''if self.cfg.at:
                word_weights = tf.constant(np.load(self.cfg.word_weight)["embeddings"], dtype=tf.float32,
                                           name="w_weight", shape=[self.word_size - 1, 1])
                word_table = self.emb_normalize(word_table, word_weights)'''
            word_table = tf.concat([tf.zeros([1, self.cfg.word_dim], dtype=tf.float32), word_table], axis=0)
            word_emb = tf.nn.embedding_lookup(word_table, self.words)
            # char table
            char_table = tf.get_variable(name="char_table", shape=[self.char_size - 1, self.cfg.char_dim],
                                         trainable=True, dtype=tf.float32)
            '''if self.cfg.at:
                char_weights = tf.constant(np.load(self.cfg.char_weight)["embeddings"], dtype=tf.float32,
                                           name="c_weight", shape=[self.char_size - 1, 1])
                char_table = self.emb_normalize(char_table, char_weights)'''
            char_table = tf.concat([tf.zeros([1, self.cfg.char_dim], dtype=tf.float32), char_table], axis=0)
            char_emb = tf.nn.embedding_lookup(char_table, self.chars)

            if self.cfg.use_orthographic:
                # orthographic word table
                ortho_word_table = tf.get_variable(name="ortho_word_table", dtype=tf.float32, trainable=True,
                                                   shape=[self.ortho_word_size - 1, self.cfg.ortho_word_dim])
                ortho_word_table = tf.concat([tf.zeros([1, self.cfg.ortho_word_dim], dtype=tf.float32),
                                              ortho_word_table], axis=0)
                ortho_word_emb = tf.nn.embedding_lookup(ortho_word_table, self.ortho_words)
                # orthographic char table
                ortho_char_table = tf.get_variable(name="ortho_char_table", dtype=tf.float32, trainable=True,
                                                   shape=[self.ortho_char_size - 1, self.cfg.ortho_char_dim])
                ortho_char_table = tf.concat([tf.zeros([1, self.cfg.ortho_char_dim], dtype=tf.float32),
                                              ortho_char_table], axis=0)
                ortho_char_emb = tf.nn.embedding_lookup(ortho_char_table, self.ortho_chars)

        with tf.variable_scope("computation_graph"):
            # create module
            if self.cfg.word_project:
                word_dense = tf.layers.Dense(units=self.cfg.word_dim, use_bias=True, _reuse=tf.AUTO_REUSE,
                                             name="word_project")
            emb_dropout = tf.layers.Dropout(rate=self.cfg.emb_drop_rate)
            rnn_dropout = tf.layers.Dropout(rate=self.cfg.rnn_drop_rate)
            char_tdnn_hw = CharTDNNHW(self.cfg.char_kernels, self.cfg.char_kernel_features, self.cfg.char_dim,
                                      self.cfg.highway_layers, padding="VALID", activation=tf.nn.tanh, use_bias=True,
                                      hw_activation=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope="char_tdnn_hw")
            if self.cfg.use_orthographic:
                ortho_char_tdnn_hw = CharTDNNHW(self.cfg.char_kernels, self.cfg.char_kernel_features,
                                                self.cfg.ortho_char_dim, self.cfg.highway_layers, padding="VALID",
                                                activation=tf.nn.tanh, use_bias=True, hw_activation=tf.nn.tanh,
                                                reuse=tf.AUTO_REUSE, scope="ortho_char_tdnn_hw")
            bi_rnn = BiRNN(self.cfg.num_units, reuse=tf.AUTO_REUSE, scope="bi_rnn")
            if not self.cfg.concat_rnn:
                fw_dense = tf.layers.Dense(units=self.cfg.num_units, use_bias=False, _reuse=tf.AUTO_REUSE, name="fw_d")
                bw_dense = tf.layers.Dense(units=self.cfg.num_units, use_bias=False, _reuse=tf.AUTO_REUSE, name="bw_d")
                rnn_bias = tf.get_variable(name="r_bias", shape=[self.cfg.num_units], dtype=tf.float32, trainable=True)
            dense = tf.layers.Dense(units=self.label_size, use_bias=True, _reuse=tf.AUTO_REUSE, name="project")
            if self.cfg.label_weight:
                label_weight = tf.constant(WEIGHT[self.cfg.language], dtype=tf.float32)
            self.crf_layer = LBCRF(self.label_size)

            # compute outputs
            def compute_logits(w_emb, c_emb, o_w_emb=None, o_c_emb=None):
                if self.cfg.word_project:
                    w_emb = word_dense(w_emb)
                char_cnn = char_tdnn_hw(c_emb)
                emb = tf.concat([w_emb, char_cnn], axis=-1)
                if o_w_emb is not None:
                    ortho_char_cnn = ortho_char_tdnn_hw(o_c_emb)
                    emb = tf.concat([emb, o_w_emb, ortho_char_cnn], axis=-1)
                emb = emb_dropout(emb, training=self.is_train)
                rnn_outputs = bi_rnn(emb, self.seq_len, concat=self.cfg.concat_rnn)
                if self.cfg.concat_rnn:
                    rnn_outputs = rnn_dropout(rnn_outputs, training=self.is_train)
                else:
                    rnn_outputs_1 = rnn_dropout(rnn_outputs[0], training=self.is_train)
                    rnn_outputs_2 = rnn_dropout(rnn_outputs[1], training=self.is_train)
                    rnn_outputs = tf.tanh(tf.nn.bias_add(fw_dense(rnn_outputs_1) + bw_dense(rnn_outputs_2),
                                                         bias=rnn_bias))
                logits = dense(rnn_outputs)
                if self.cfg.label_weight:
                    logits = logits * label_weight
                crf_loss, transition = self.crf_layer.crf_log_likelihood(logits, self.labels, self.seq_len)
                if self.cfg.focal_loss:
                    lb_loss = focal_loss(logits, self.labels, self.seq_len)
                    return logits, transition, crf_loss, lb_loss, crf_loss + lb_loss
                else:
                    return logits, transition, crf_loss
            
            if self.cfg.focal_loss:
                if self.cfg.use_orthographic:
                    self.logits, self.transition, self.crf_loss, self.lb_loss, self.loss = compute_logits(
                        word_emb, char_emb, ortho_word_emb, ortho_char_emb)
                else:
                    self.logits, self.transition, self.crf_loss, self.lb_loss, self.loss = compute_logits(word_emb,
                                                                                                          char_emb)
            else:
                if self.cfg.use_orthographic:
                    self.logits, self.transition, self.loss = compute_logits(word_emb, char_emb, ortho_word_emb,
                                                                             ortho_char_emb)
                else:
                    self.logits, self.transition, self.loss = compute_logits(word_emb, char_emb)
            
            if self.cfg.at:
                perturb_word_emb = self.add_perturbation(word_emb, self.loss, epsilon=self.cfg.epsilon)
                perturb_char_emb = self.add_perturbation(char_emb, self.loss, epsilon=self.cfg.epsilon)
                if self.cfg.use_orthographic:
                    perturb_o_word_emb = self.add_perturbation(ortho_word_emb, self.loss, epsilon=self.cfg.epsilon)
                    perturb_o_char_emb = self.add_perturbation(ortho_char_emb, self.loss, epsilon=self.cfg.epsilon)
                    if self.cfg.focal_loss:
                        *_, adv_crf_loss, adv_lb_loss, adv_loss = compute_logits(
                            perturb_word_emb, perturb_char_emb, perturb_o_word_emb, perturb_o_char_emb)
                        self.crf_loss += adv_crf_loss
                        self.lb_loss += adv_lb_loss
                    else:
                        *_, adv_loss = compute_logits(perturb_word_emb, perturb_char_emb, perturb_o_word_emb,
                                                      perturb_o_char_emb)
                else:
                    if self.cfg.focal_loss:
                        *_, adv_crf_loss, adv_lb_loss, adv_loss = compute_logits(perturb_word_emb, perturb_char_emb)
                        self.crf_loss += adv_crf_loss
                        self.lb_loss += adv_lb_loss
                    else:
                        *_, adv_loss = compute_logits(perturb_word_emb, perturb_char_emb)
                self.loss += adv_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
            grads, vs = zip(*optimizer.compute_gradients(self.loss))
            grads, _ = tf.clip_by_global_norm(grads, self.cfg.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(grads, vs))
        else:
            self.train_op = optimizer.minimize(self.loss)

    def _predict_op(self, data):
        feed_dict = self._get_feed_dict(data)
        logits, trans_params, seq_len = self.sess.run([self.logits, self.transition, self.seq_len], feed_dict=feed_dict)
        return self.viterbi_decode(logits, trans_params, seq_len)

    def train(self, dataset):
        self.logger.info("Start training...")
        best_f1, no_imprv_epoch, init_lr, lr, cur_step = -np.inf, 0, self.cfg.lr, self.cfg.lr, 0
        for epoch in range(1, self.cfg.epochs + 1):
            self.logger.info("Epoch {}/{}:".format(epoch, self.cfg.epochs))
            prog = Progbar(target=dataset.get_num_batches())
            for i, data in enumerate(dataset.get_data_batches()):
                cur_step += 1
                feed_dict = self._get_feed_dict(data, is_train=True, lr=lr)
                if self.cfg.focal_loss:
                    _, crf_loss, lb_loss, train_loss = self.sess.run([self.train_op, self.crf_loss, self.lb_loss,
                                                                      self.loss], feed_dict=feed_dict)
                    prog.update(i + 1, [("Global Step", int(cur_step)), ("CRF Loss", crf_loss), ("LB Loss", lb_loss),
                                        ("Total Loss", train_loss)])
                else:
                    _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                    prog.update(i + 1, [("Global Step", int(cur_step)), ("Total Loss", train_loss)])
            # learning rate decay
            if self.cfg.use_lr_decay and self.cfg.decay_step:
                lr = max(init_lr / (1.0 + self.cfg.lr_decay * epoch / self.cfg.decay_step), self.cfg.minimal_lr)
            # evaluate
            if not self.cfg.dev_for_train:
                self.evaluate(dataset.get_data_batches("dev"), name="dev")
            score = self.evaluate(dataset.get_data_batches("test"), name="test")
            if score["FB1"] > best_f1:
                best_f1, no_imprv_epoch = score["FB1"], 0
                self.save_session(epoch)
                self.logger.info(" -- new BEST score on test dataset: {:04.2f}".format(best_f1))
            else:
                no_imprv_epoch += 1
                if self.cfg.no_imprv_tolerance is not None and no_imprv_epoch >= self.cfg.no_imprv_tolerance:
                    self.logger.info("early stop at {}th epoch without improvement".format(epoch))
                    self.logger.info("best score on test set: {}".format(best_f1))
                    break

    def evaluate(self, dataset, name):
        all_data = list()
        for data in dataset:
            predicts = self._predict_op(data)
            all_data.append((data["labels"], predicts, data["words"], data["seq_len"]))
        return self.evaluate_f1(all_data, self.rev_word_dict, self.rev_label_dict, name)
