import tensorflow as tf
import math
import random
import numpy as np
from lavd.shares import BaseModel, CharCNNHW, CRF, BiRNN, Embedding, rnn_autoencoder
from utils.logger import Progbar


class DeepCRFModel(BaseModel):
    def __init__(self, config):
        super(DeepCRFModel, self).__init__(config)
        self._init_configs()
        with tf.Graph().as_default():
            self._add_placeholders()
            self._build_model()
            self.logger.info("total params: {}".format(self.count_params()))
            self._initialize_session()

    def _init_configs(self):
        vocab = self.load_dataset(self.cfg.vocab)
        self.word_dict, self.char_dict, self.label_dict = vocab["word_dict"], vocab["char_dict"], vocab["label_dict"]
        self.word_size, self.char_size, self.label_size = len(self.word_dict), len(self.char_dict), len(self.label_dict)
        self.rev_word_dict = dict([(idx, word) for word, idx in self.word_dict.items()])
        self.rev_char_dict = dict([(idx, char) for char, idx in self.char_dict.items()])
        self.rev_label_dict = dict([(idx, tag) for tag, idx in self.label_dict.items()])

    def _get_feed_dict(self, data, loss_weight=1.0, is_train=False, lr=None):
        feed_dict = {self.words: data["words"], self.seq_len: data["seq_len"], self.chars: data["chars"],
                     self.char_seq_len: data["char_seq_len"]}
        if "labels" in data:
            feed_dict[self.labels] = data["labels"]
        feed_dict[self.is_train] = is_train
        feed_dict[self.loss_weight] = loss_weight
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
        self.loss_weight = tf.placeholder(tf.float32, name="loss_weight")
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
            # general computation components
            emb_dropout = tf.layers.Dropout(rate=self.cfg.emb_drop_rate, name="embedding_dropout_layer")
            rnn_dropout = tf.layers.Dropout(rate=self.cfg.rnn_drop_rate, name="rnn_dropout_layer")
            char_tdnn_hw = CharCNNHW(self.cfg.char_kernels, self.cfg.char_kernel_features, self.cfg.char_dim,
                                     self.cfg.highway_layers, padding="VALID", activation=tf.nn.tanh, use_bias=True,
                                     hw_activation=tf.nn.tanh, reuse=tf.AUTO_REUSE, scope="char_tdnn_hw")
            bi_rnn = BiRNN(self.cfg.num_units, concat=self.cfg.concat_rnn, reuse=tf.AUTO_REUSE, scope="bi_rnn")
            crf_layer = CRF(self.label_size, reuse=tf.AUTO_REUSE, scope="crf")

            # compute logits
            char_cnn = char_tdnn_hw(char_emb)
            emb = emb_dropout(tf.concat([word_emb, char_cnn], axis=-1), training=self.is_train)
            rnn_outputs, state = bi_rnn(emb, self.seq_len)
            rnn_outputs = rnn_dropout(rnn_outputs, training=self.is_train)
            self.logits, self.transition, self.loss = crf_layer(rnn_outputs, self.labels, self.seq_len)
            self.loss = self.loss * self.loss_weight
            self.ae_loss, self.ae_acc = rnn_autoencoder(rnn_outputs, state, self.words, self.seq_len,
                                                        2 * self.cfg.num_units, self.word_size, self.loss_weight,
                                                        reuse=tf.AUTO_REUSE, name="rnn_autoencoder")

        # build optimizer
        labeled_optimizer = self._build_optimizer()
        unlabeled_optimizer = self._build_optimizer()
        if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
            grads, vs = zip(*labeled_optimizer.compute_gradients(self.loss))
            grads, _ = tf.clip_by_global_norm(grads, self.cfg.grad_clip)
            self.train_op = labeled_optimizer.apply_gradients(zip(grads, vs))
        else:
            self.train_op = labeled_optimizer.minimize(self.loss)
        if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
            grads, vs = zip(*unlabeled_optimizer.compute_gradients(self.ae_loss))
            grads, _ = tf.clip_by_global_norm(grads, self.cfg.grad_clip)
            self.ae_train_op = unlabeled_optimizer.apply_gradients(zip(grads, vs))
        else:
            self.ae_train_op = unlabeled_optimizer.minimize(self.ae_loss)

    def _predict_op(self, data):
        feed_dict = self._get_feed_dict(data)
        logits, transition, seq_len = self.sess.run([self.logits, self.transition, self.seq_len], feed_dict=feed_dict)
        return self.viterbi_decode(logits, transition, seq_len)

    @staticmethod
    def _compute_loss_weights(label_dataset_size, partial_dataset_size, unlabeled_dataset_size, method=0):
        if method == 0:
            return 1.0, 1.0, 1.0
        elif method == 1:  # sqrt ratio weight
            partial_weight = float(label_dataset_size) / float(partial_dataset_size)
            unlabeled_weight = float(label_dataset_size) / float(unlabeled_dataset_size)
            return 1.0, partial_weight, unlabeled_weight
        elif method == 2:  # ratio weight
            partial_weight = math.sqrt(float(label_dataset_size) / float(partial_dataset_size))
            unlabeled_weight = math.sqrt(float(label_dataset_size) / float(unlabeled_dataset_size))
            return 1.0, partial_weight, unlabeled_weight
        elif method == 3:  # ratio 1.5
            partial_weight = math.pow(float(label_dataset_size) / float(partial_dataset_size), 1.5)
            unlabeled_weight = math.pow(float(label_dataset_size) / float(unlabeled_dataset_size), 1.5)
            return 1.0, partial_weight, unlabeled_weight
        elif method == 4:  # ratio 2.0
            partial_weight = math.pow(float(label_dataset_size) / float(partial_dataset_size), 2.0)
            unlabeled_weight = math.pow(float(label_dataset_size) / float(unlabeled_dataset_size), 2.0)
            return 1.0, partial_weight, unlabeled_weight
        else:
            raise ValueError("Unknown method...")

    @staticmethod
    def _create_batches(num_label, num_partial, num_unlabeled):
        tmp_num_label = math.ceil(num_label * 0.7)
        batches = ["partial"] * num_partial + ["unlabeled"] * num_unlabeled + ["label"] * tmp_num_label
        random.shuffle(batches)
        batches = batches + ["label"] * (num_label - tmp_num_label)
        return batches

    def train(self, label_dataset, partial_dataset, unlabeled_dataset):
        best_f1, no_imprv_epoch, lr, cur_step = -np.inf, 0, self.cfg.lr, 0
        loss_weight = self._compute_loss_weights(label_dataset.get_dataset_size(), partial_dataset.get_dataset_size(),
                                                 unlabeled_dataset.get_dataset_size(), method=self.cfg.loss_method)
        self.logger.info("Start training...")
        for epoch in range(1, self.cfg.epochs + 1):
            self.logger.info("Epoch {}/{}:".format(epoch, self.cfg.epochs))
            batches = self._create_batches(label_dataset.get_num_batches(), partial_dataset.get_num_batches(),
                                           unlabeled_dataset.get_num_batches())
            prog = Progbar(target=len(batches))
            prog.update(0, [("Global Step", int(cur_step)), ("Label Loss", 0.0), ("Partial Loss", 0.0),
                            ("AE Loss", 0.0), ("AE Acc", 0.0)])
            for i, batch_name in enumerate(batches):
                cur_step += 1
                if batch_name == "label":
                    data = label_dataset.get_next_train_batch()
                    feed_dict = self._get_feed_dict(data, loss_weight=loss_weight[0], is_train=True, lr=lr)
                    _, cost = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                    prog.update(i + 1, [("Global Step", int(cur_step)), ("Label Loss", cost)])
                if batch_name == "partial":
                    data = partial_dataset.get_next_train_batch()
                    feed_dict = self._get_feed_dict(data, loss_weight=loss_weight[1], is_train=True, lr=lr)
                    _, cost = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                    prog.update(i + 1, [("Global Step", int(cur_step)), ("Partial Loss", cost)])
                if batch_name == "unlabeled":
                    data = unlabeled_dataset.get_next_train_batch()
                    feed_dict = self._get_feed_dict(data, loss_weight=loss_weight[2], is_train=True, lr=lr)
                    _, cost, acc = self.sess.run([self.ae_train_op, self.ae_loss, self.ae_acc], feed_dict=feed_dict)
                    prog.update(i + 1, [("Global Step", int(cur_step)), ("AE Loss", cost), ("AE Acc", acc)])
            # learning rate decay
            if self.cfg.use_lr_decay:
                if self.cfg.decay_step:
                    lr = max(self.cfg.lr / (1.0 + self.cfg.lr_decay * epoch / self.cfg.decay_step), self.cfg.minimal_lr)
            self.evaluate(label_dataset.get_data_batches("dev"), name="dev")
            score = self.evaluate(label_dataset.get_data_batches("test"), name="test")
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
