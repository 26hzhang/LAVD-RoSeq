import os
import codecs
import json
import numpy as np
import tensorflow as tf
from functools import reduce
from operator import mul
from utils.logger import get_logger
from utils.CoNLLeval import CoNLLeval


class CharTDNNHW:
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


class CharBiRNN:
    def __init__(self, num_units, reuse=None, scope="char_bi_rnn"):
        self.cell_fw = tf.nn.rnn_cell.LSTMCell(num_units)
        self.cell_bw = tf.nn.rnn_cell.LSTMCell(num_units)
        self.reuse = reuse
        self.scope = scope

    def __call__(self, inputs, seq_len, use_last_state=False, time_major=False):
        assert not time_major, "BiRNN class cannot support time_major currently"
        with tf.variable_scope(self.scope):
            flat_inputs = flatten(inputs, keep=2)  # reshape to [-1, max_time, dim]
            seq_len = flatten(seq_len, keep=0)  # reshape to [x] (one dimension sequence)
            outputs, ((_, h_fw), (_, h_bw)) = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, flat_inputs,
                                                                              sequence_length=seq_len, dtype=tf.float32)
            if use_last_state:  # return last states
                output = tf.concat([h_fw, h_bw], axis=-1)  # shape = [-1, 2 * num_units]
                output = reconstruct(output, ref=inputs, keep=2, remove_shape=1)  # remove the max_time shape
            else:
                output = tf.concat(outputs, axis=-1)  # shape = [-1, max_time, 2 * num_units]
                output = reconstruct(output, ref=inputs, keep=2)  # reshape to same as inputs, except the last two dim
            return output


class BiRNN:
    def __init__(self, num_units, reuse=None, scope="bi_rnn"):
        self.num_units = num_units
        self.cell_fw = tf.nn.rnn_cell.LSTMCell(num_units)
        self.cell_bw = tf.nn.rnn_cell.LSTMCell(num_units)
        self.reuse = reuse
        self.scope = scope

    def __call__(self, inputs, seq_len, concat=True):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, inputs, seq_len, dtype=tf.float32)
            if concat:
                outputs = tf.concat(outputs, axis=-1)
            return outputs


class RNN:
    def __init__(self, num_units, reuse=None, scope="rnn"):
        self.num_units = num_units
        self.cell = tf.nn.rnn_cell.LSTMCell(num_units)
        self.reuse = reuse
        self.scope = scope

    def __call__(self, inputs, seq_len):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            outputs, _ = tf.nn.dynamic_rnn(self.cell, inputs, seq_len, dtype=tf.float32)
            return outputs


class CRF:
    def __init__(self, num_units, reuse=None, scope="crf"):
        self.reuse = reuse
        self.scope = scope
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self.transition = tf.get_variable(name="transition", shape=[num_units, num_units], dtype=tf.float32)

    def __call__(self, inputs, labels, seq_len):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            crf_loss, transition = tf.contrib.crf.crf_log_likelihood(inputs, labels, seq_len, self.transition)
            return transition, tf.reduce_mean(-crf_loss)


class Base:
    def __init__(self, config):
        tf.set_random_seed(config.random_seed)
        self.cfg = config
        # create folders and logger
        if not os.path.exists(self.cfg.checkpoint_path):
            os.makedirs(self.cfg.checkpoint_path)
        self.logger = get_logger(os.path.join(self.cfg.checkpoint_path, "log.txt"))

    def _initialize_session(self):
        if not self.cfg.use_gpu:
            self.sess = tf.Session()
        else:
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=sess_config)
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
    def emb_normalize(emb, weights):
        mean = tf.reduce_sum(weights * emb, axis=0, keepdims=True)
        var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.0), axis=0, keepdims=True)
        stddev = tf.sqrt(1e-6 + var)
        return (emb - mean) / stddev

    @staticmethod
    def add_perturbation(emb, loss, epsilon=5.0):
        """Adds gradient to embedding and recomputes classification loss."""
        grad, = tf.gradients(loss, emb, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        grad = tf.stop_gradient(grad)
        alpha = tf.reduce_max(tf.abs(grad), axis=(1, 2), keepdims=True) + 1e-12  # l2 scale
        l2_norm = alpha * tf.sqrt(tf.reduce_sum(tf.pow(grad / alpha, 2), axis=(1, 2), keepdims=True) + 1e-6)
        norm_grad = grad / l2_norm
        perturb = epsilon * norm_grad
        return emb + perturb

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


def focal_loss(logits, labels, seq_len, alpha=0.5, gamma=2):
    logits = tf.nn.softmax(logits, axis=-1)
    if labels.get_shape().ndims < logits.get_shape().ndims:
        labels = tf.one_hot(labels, depth=logits.shape[-1].value, axis=-1)
    labels = tf.cast(labels, dtype=tf.float32)
    zeros = tf.zeros_like(logits, dtype=logits.dtype)
    pos_logits_prob = tf.where(labels > zeros, labels - logits, zeros)
    neg_logits_prob = tf.where(labels > zeros, zeros, logits)
    '''cross_entropy = - alpha * (pos_logits_prob ** gamma) * tf.log(tf.clip_by_value(logits, 1e-8, 1.0)) \
                    - (1 - alpha) * (neg_logits_prob ** gamma) * tf.log(tf.clip_by_value(1.0 - logits, 1e-8, 1.0))'''
    cross_entropy = - (pos_logits_prob ** gamma) * tf.log(tf.clip_by_value(logits, 1e-8, 1.0)) \
                    - (neg_logits_prob ** gamma) * tf.log(tf.clip_by_value(1.0 - logits, 1e-8, 1.0))
    mask = tf.sequence_mask(seq_len, maxlen=tf.reduce_max(seq_len), dtype=tf.float32)
    cross_entropy = tf.reduce_sum(cross_entropy, axis=-1)
    cross_entropy = tf.reduce_sum(cross_entropy * mask) / tf.reduce_sum(mask)
    return cross_entropy


def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep, remove_shape=None):
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    if remove_shape is not None:
        tensor_start = tensor_start + remove_shape
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out
