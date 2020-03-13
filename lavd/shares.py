import os
import codecs
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell, RNNCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn
from utils.logger import get_logger
from utils.CoNLLeval import CoNLLeval


class Embedding:
    def __init__(self, token_size, token_dim, token2vec=None, tune_emb=True, word_project=False, scope="word_table"):
        self.scope, self.word_project = scope, word_project
        with tf.variable_scope(self.scope):
            if token2vec is not None:
                table = tf.Variable(initial_value=np.load(token2vec)["embeddings"], name="table", dtype=tf.float32,
                                    trainable=tune_emb)
                unk = tf.get_variable(name="unk", shape=[1, token_dim], trainable=True, dtype=tf.float32)
                table = tf.concat([unk, table], axis=0)
            else:
                table = tf.get_variable(name="table", shape=[token_size - 1, token_dim], dtype=tf.float32,
                                        trainable=True)
            self.table = tf.concat([tf.zeros([1, token_dim], dtype=tf.float32), table], axis=0)
            if self.word_project:
                self.dense = tf.layers.Dense(units=token_dim, use_bias=True, _reuse=tf.AUTO_REUSE, name="word_project")

    def __call__(self, tokens):
        with tf.variable_scope(self.scope):
            token_emb = tf.nn.embedding_lookup(self.table, tokens)
            if self.word_project:
                token_emb = self.dense(token_emb)
            return token_emb


class CharCNNHW:
    def __init__(self, kernels, kernel_features, dim, hw_layers, padding="VALID", activation=tf.nn.relu, use_bias=True,
                 hw_activation=tf.nn.tanh, reuse=None, scope="char_tdnn_hw"):
        assert len(kernels) == len(kernel_features), "kernel and features must have the same size"
        self.padding = padding
        self.activation = activation
        self.reuse = reuse
        self.scope = scope
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self.weights = []
            for i, (kernel_size, feature_size) in enumerate(zip(kernels, kernel_features)):
                weight = tf.get_variable("filter_%d" % i, shape=[1, kernel_size, dim, feature_size], dtype=tf.float32)
                bias = tf.get_variable("bias_%d" % i, shape=[feature_size], dtype=tf.float32)
                self.weights.append((weight, bias))
            self.dense_layers = []
            for i in range(hw_layers):
                trans = tf.layers.Dense(units=sum(kernel_features), use_bias=use_bias, activation=hw_activation,
                                        name="trans_%d" % i)
                gate = tf.layers.Dense(units=sum(kernel_features), use_bias=use_bias, activation=tf.nn.sigmoid,
                                       name="gate_%d" % i)
                self.dense_layers.append((trans, gate))

    def __call__(self, inputs):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            # cnn
            strides = [1, 1, 1, 1]
            outputs = []
            for i, (weight, bias) in enumerate(self.weights):
                conv = tf.nn.conv2d(inputs, weight, strides=strides, padding=self.padding, name="conv_%d" % i) + bias
                output = tf.reduce_max(self.activation(conv), axis=2)
                outputs.append(output)
            outputs = tf.concat(values=outputs, axis=-1)
            # highway
            for trans, gate in self.dense_layers:
                g = gate(outputs)
                outputs = g * trans(outputs) + (1.0 - g) * outputs
            return outputs


class BiRNN:
    def __init__(self, num_units, activation=tf.nn.tanh, concat=True, reuse=None, scope="bi_rnn"):
        self.reuse, self.scope, self.concat = reuse, scope, concat
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self.cell_fw = tf.nn.rnn_cell.LSTMCell(num_units)
            self.cell_bw = tf.nn.rnn_cell.LSTMCell(num_units)
            if not self.concat:
                self.dense_fw = tf.layers.Dense(units=num_units, use_bias=False, _reuse=self.reuse, name="dense_fw")
                self.dense_bw = tf.layers.Dense(units=num_units, use_bias=False, _reuse=self.reuse, name="dense_bw")
                self.bias = tf.get_variable(name="bias", shape=[num_units], dtype=tf.float32, trainable=True)
                self.activation = activation

    def __call__(self, inputs, seq_len):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            outputs, state = bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, inputs, seq_len, dtype=tf.float32)
            if self.concat:
                outputs = tf.concat(outputs, axis=-1)
            else:
                outputs = self.dense_fw(outputs[0]) + self.dense_bw(outputs[1])
                outputs = self.activation(tf.nn.bias_add(outputs, bias=self.bias))
            return outputs, state


class HighwayNets:
    def __init__(self, layers, num_units, activation=tf.nn.tanh, use_bias=True, reuse=None, scope="highway"):
        self.reuse, self.scope = reuse, scope
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self.dense_layers = []
            for i in range(layers):
                trans = tf.layers.Dense(units=num_units, use_bias=use_bias, activation=activation, name="trans_%d" % i)
                gate = tf.layers.Dense(units=num_units, use_bias=use_bias, activation=tf.nn.sigmoid, name="gate_%d" % i)
                self.dense_layers.append((trans, gate))

    def __call__(self, inputs):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            for trans, gate in self.dense_layers:
                g = gate(inputs)
                inputs = g * trans(inputs) + (1.0 - g) * inputs
            return inputs


class CRF:
    def __init__(self, num_units, reuse=None, scope="crf"):
        self.reuse, self.scope = reuse, scope
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self.dense = tf.layers.Dense(units=num_units, use_bias=True, _reuse=False, name="project")
            self.transition = tf.get_variable(name="transition", shape=[num_units, num_units], dtype=tf.float32)

    def __call__(self, inputs, labels, seq_len):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            logits = self.dense(inputs)
            crf_loss, transition = tf.contrib.crf.crf_log_likelihood(logits, labels, seq_len, self.transition)
            return logits, transition, tf.reduce_mean(-crf_loss)


class DecodeRNNCell(RNNCell):
    def __init__(self, num_units, decoder_inputs):
        super(DecodeRNNCell, self).__init__()
        self.num_units = num_units
        self.cell = LSTMCell(num_units)
        self.encoder = decoder_inputs
        self.decoder_inputs = None

    @property
    def state_size(self):
        return self.cell.state_size

    @property
    def output_size(self):
        return self.cell.output_size

    def compute_output_shape(self, input_shape):
        pass

    def initialize_decoder_inputs(self):
        self.decoder_inputs = tf.concat([tf.zeros_like(self.encoder), self.encoder], axis=1)

    def __call__(self, inputs, state, scope=None):
        output, state = self.cell(self.decoder_inputs, state)
        weight = tf.get_variable(name="weight", shape=[self.num_units, self.num_units], dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable(name="bias", shape=[self.num_units], initializer=tf.constant_initializer(0.1),
                               dtype=tf.float32)
        output1 = tf.matmul(output, weight) + bias
        self.decoder_inputs = tf.concat([output1, self.encoder], axis=1)
        return output, state


def rnn_autoencoder(inputs, state, words, seq_len, feature_size, word_size, loss_weight=None, reuse=None,
                    name="rnn_autoencoder"):
    with tf.variable_scope(name, reuse=reuse):
        # compute RNN outputs
        initial_cstate = tf.concat([state[0][0], state[1][0]], 1)
        initial_hstate = tf.concat([state[0][1], state[1][1]], 1)
        initial_state = tf.nn.rnn_cell.LSTMStateTuple(initial_cstate, initial_hstate)
        cell = DecodeRNNCell(feature_size, initial_hstate)
        cell.initialize_decoder_inputs()
        outputs, _ = dynamic_rnn(cell, inputs, seq_len, dtype=tf.float32, initial_state=initial_state)
        outputs = tf.layers.dense(outputs, units=word_size, use_bias=True, activation=None, reuse=tf.AUTO_REUSE,
                                  name="dense_layer")
        # compute labels and logits
        mask = tf.sequence_mask(seq_len, maxlen=tf.reduce_max(seq_len), dtype=tf.float32)
        labels = tf.one_hot(words, word_size)
        logits = tf.nn.softmax(outputs, axis=-1)
        # compute autoencoder loss
        ae_loss = tf.reduce_sum(-labels * tf.log(logits) * mask[:, :, None]) / tf.reduce_sum(mask)
        if loss_weight is not None:
            ae_loss = ae_loss * loss_weight
        # compute autoencoder accuracy
        predictions = tf.to_int32(tf.argmax(logits, axis=-1))
        groundtruth = tf.to_int32(tf.argmax(labels, axis=-1))
        ae_acc = tf.reduce_sum(tf.to_float(tf.equal(predictions, groundtruth)) * mask) / tf.reduce_sum(mask)
        return ae_loss, ae_acc


def self_attention(inputs, return_alphas=False, project=True, reuse=None, name="self_attention"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        hidden_size = inputs.shape[-1].value
        if project:
            x = tf.layers.dense(inputs, units=hidden_size, use_bias=False, activation=tf.nn.tanh)
        else:
            x = inputs
        weight = tf.get_variable(name="weight", shape=[hidden_size, 1], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=0.01, seed=1227))
        x = tf.tensordot(x, weight, axes=1)
        alphas = tf.nn.softmax(x, axis=-1)
        output = tf.matmul(tf.transpose(inputs, perm=[0, 2, 1]), alphas)
        output = tf.squeeze(output, axis=-1)
        if return_alphas:
            return output, alphas
        else:
            return output


class BaseModel:
    def __init__(self, config):
        tf.set_random_seed(config.random_seed)
        self.cfg = config
        # create folders and logger
        if not os.path.exists(self.cfg.checkpoint_path):
            os.makedirs(self.cfg.checkpoint_path)
        self.logger = get_logger(os.path.join(self.cfg.checkpoint_path, "log.txt"))

    def _initialize_session(self):
        if self.cfg.use_gpu:
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=sess_config)
        else:
            self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=self.cfg.max_to_keep)
        self.sess.run(tf.global_variables_initializer())

    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.cfg.checkpoint_path)  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_session(self, epoch):
        self.saver.save(self.sess, self.cfg.checkpoint_path + self.cfg.model_name, global_step=epoch)

    def close_session(self):
        self.sess.close()

    @staticmethod
    def viterbi_decode(logits, trans_params, seq_len):
        viterbi_sequences = []
        for logit, lens in zip(logits, seq_len):
            logit = logit[:lens]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            viterbi_sequences += [viterbi_seq]
        return viterbi_sequences

    @staticmethod
    def count_params(scope=None):
        if scope is None:
            return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        else:
            return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables(scope)])

    @staticmethod
    def load_dataset(filename):
        with codecs.open(filename, mode='r', encoding='utf-8') as f:
            dataset = json.load(f)
        return dataset

    def _add_summary(self, summary_path):
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        self.summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(summary_path + "train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter(summary_path + "test")

    def _build_optimizer(self):
        if self.cfg.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
        elif self.cfg.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        elif self.cfg.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        elif self.cfg.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
        else:  # default adam optimizer
            if self.cfg.optimizer != 'adam':
                print('Unsupported optimizing method {}. Using default adam optimizer.'.format(self.cfg.optimizer))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        return optimizer

    def evaluate_f1(self, dataset, rev_word_dict, rev_label_dict, name):
        save_path = os.path.join(self.cfg.checkpoint_path, name + "_result.txt")
        if os.path.exists(save_path):
            os.remove(save_path)
        predictions, groundtruth, words_list = list(), list(), list()
        for b_labels, b_predicts, b_words, b_seq_len in dataset:
            for labels, predicts, words, seq_len in zip(b_labels, b_predicts, b_words, b_seq_len):
                predictions.append([rev_label_dict[x] for x in predicts[:seq_len]])
                groundtruth.append([rev_label_dict[x] for x in labels[:seq_len]])
                words_list.append([rev_word_dict[x] for x in words[:seq_len]])
        conll_eval = CoNLLeval()
        score = conll_eval.conlleval(predictions, groundtruth, words_list, save_path)
        self.logger.info("{} dataset -- acc: {:04.2f}, pre: {:04.2f}, rec: {:04.2f}, FB1: {:04.2f}"
                         .format(name, score["accuracy"], score["precision"], score["recall"], score["FB1"]))
        return score
