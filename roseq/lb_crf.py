import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs


class LBCRF:
    def __init__(self, num_tags):
        self.num_tags = num_tags
        self.transition_params = vs.get_variable(name="transitions", shape=[num_tags, num_tags])

    def crf_log_likelihood(self, inputs, tag_indices, seq_lens):
        sequence_scores = self._crf_sequence_score(inputs, tag_indices, seq_lens)
        log_norm = self._crf_log_norm(inputs, seq_lens)
        log_likelihood = sequence_scores - log_norm
        return tf.reduce_mean(-log_likelihood), self.transition_params

    def _crf_log_norm(self, inputs, seq_lens):
        first_input = array_ops.slice(inputs, [0, 0, 0], [-1, 1, -1])
        first_input = array_ops.squeeze(first_input, [1])
        rest_of_input = array_ops.slice(inputs, [0, 1, 0], [-1, -1, -1])
        forward_cell = CrfForwardRnnCell(self.transition_params)
        seq_lens_less_one = math_ops.maximum(constant_op.constant(0, dtype=seq_lens.dtype), seq_lens - 1)
        _, alphas = rnn.dynamic_rnn(cell=forward_cell, inputs=rest_of_input, sequence_length=seq_lens_less_one,
                                    initial_state=first_input, dtype=dtypes.float32)
        log_norm = math_ops.reduce_logsumexp(alphas, [1])
        log_norm = array_ops.where(math_ops.less_equal(seq_lens, 0), array_ops.zeros_like(log_norm), log_norm)
        return log_norm

    def _crf_sequence_score(self, inputs, tag_indices, seq_lens):
        unary_scores = self._crf_unary_score(tag_indices, seq_lens, inputs)
        binary_scores = self._crf_binary_score(tag_indices, seq_lens)
        sequence_scores = unary_scores + binary_scores
        return sequence_scores

    def _crf_unary_score(self, tag_indices, seq_lens, inputs):
        batch_size = array_ops.shape(inputs)[0]
        max_seq_len = array_ops.shape(inputs)[1]
        flattened_inputs = array_ops.reshape(inputs, [-1])
        offsets = array_ops.expand_dims(math_ops.range(batch_size) * max_seq_len * self.num_tags, 1)
        offsets += array_ops.expand_dims(math_ops.range(max_seq_len) * self.num_tags, 0)
        if tag_indices.dtype == dtypes.int64:
            offsets = math_ops.to_int64(offsets)
        flattened_tag_indices = array_ops.reshape(offsets + tag_indices, [-1])
        unary_scores = array_ops.reshape(array_ops.gather(flattened_inputs, flattened_tag_indices), [batch_size,
                                                                                                     max_seq_len])
        masks = array_ops.sequence_mask(seq_lens, maxlen=array_ops.shape(tag_indices)[1], dtype=dtypes.float32)
        unary_scores = unary_scores * masks
        unary_scores = math_ops.reduce_sum(unary_scores, axis=1)
        return unary_scores

    def _crf_binary_score(self, tag_indices, seq_lens):
        num_transitions = array_ops.shape(tag_indices)[1] - 1
        start_tag_indices = array_ops.slice(tag_indices, [0, 0], [-1, num_transitions])
        end_tag_indices = array_ops.slice(tag_indices, [0, 1], [-1, num_transitions])
        flattened_transition_indices = start_tag_indices * self.num_tags + end_tag_indices
        flattened_transition_params = array_ops.reshape(self.transition_params, [-1])
        binary_scores = array_ops.gather(flattened_transition_params, flattened_transition_indices)
        masks = array_ops.sequence_mask(seq_lens, maxlen=array_ops.shape(tag_indices)[1], dtype=dtypes.float32)
        truncated_masks = array_ops.slice(masks, [0, 1], [-1, -1])
        binary_scores = binary_scores * truncated_masks
        binary_scores = math_ops.reduce_sum(binary_scores, axis=1)
        return binary_scores

    @staticmethod
    def viterbi_decode(score, transition_params):
        trellis = np.zeros_like(score)
        backpointers = np.zeros_like(score, dtype=np.int32)
        trellis[0] = score[0]
        for t in range(1, score.shape[0]):
            v = np.expand_dims(trellis[t - 1], 1) + transition_params
            trellis[t] = score[t] + np.max(v, 0)
            backpointers[t] = np.argmax(v, 0)
        viterbi = [np.argmax(trellis[-1])]
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()
        viterbi_score = np.max(trellis[-1])
        return viterbi, viterbi_score

    def crf_decode(self, potentials, seq_lens):
        crf_fwd_cell = CrfDecodeForwardRnnCell(self.transition_params)
        initial_state = array_ops.slice(potentials, [0, 0, 0], [-1, 1, -1])
        initial_state = array_ops.squeeze(initial_state, axis=[1])
        inputs = array_ops.slice(potentials, [0, 1, 0], [-1, -1, -1])
        seq_len_less_one = math_ops.maximum(constant_op.constant(0, dtype=seq_lens.dtype), seq_lens - 1)
        backpointers, last_score = rnn.dynamic_rnn(crf_fwd_cell, inputs=inputs, initial_state=initial_state,
                                                   sequence_length=seq_len_less_one, time_major=False,
                                                   dtype=dtypes.int32)
        backpointers = gen_array_ops.reverse_sequence(backpointers, seq_len_less_one, seq_dim=1)
        crf_bwd_cell = CrfDecodeBackwardRnnCell(self.num_tags)
        initial_state = math_ops.cast(math_ops.argmax(last_score, axis=1), dtype=dtypes.int32)
        initial_state = array_ops.expand_dims(initial_state, axis=-1)
        decode_tags, _ = rnn.dynamic_rnn(crf_bwd_cell, inputs=backpointers, sequence_length=seq_len_less_one,
                                         initial_state=initial_state, time_major=False, dtype=dtypes.int32)
        decode_tags = array_ops.squeeze(decode_tags, axis=[2])
        decode_tags = array_ops.concat([initial_state, decode_tags], axis=1)
        decode_tags = gen_array_ops.reverse_sequence(decode_tags, seq_lens, seq_dim=1)
        best_score = math_ops.reduce_max(last_score, axis=1)
        return decode_tags, best_score


class CrfForwardRnnCell(rnn_cell.RNNCell):
    def __init__(self, transition_params, **kwargs):
        super().__init__(**kwargs)
        self._transition_params = array_ops.expand_dims(transition_params, 0)
        self._num_tags = transition_params.get_shape()[0].value

    def compute_output_shape(self, input_shape):
        return self._num_tags, self._num_tags

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def __call__(self, inputs, state, scope=None):
        state = array_ops.expand_dims(state, 2)
        transition_scores = state + self._transition_params
        new_alphas = inputs + math_ops.reduce_logsumexp(transition_scores, [1])
        return new_alphas, new_alphas


class CrfDecodeForwardRnnCell(rnn_cell.RNNCell):
    def __init__(self, transition_params, **kwargs):
        super().__init__(**kwargs)
        self._transition_params = array_ops.expand_dims(transition_params, 0)
        self._num_tags = transition_params.get_shape()[0].value

    def compute_output_shape(self, input_shape):
        return self._num_tags, self._num_tags

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def __call__(self, inputs, state, scope=None):
        state = array_ops.expand_dims(state, 2)
        transition_scores = state + self._transition_params
        new_state = inputs + math_ops.reduce_max(transition_scores, [1])
        backpointers = math_ops.argmax(transition_scores, 1)
        backpointers = math_ops.cast(backpointers, dtype=dtypes.int32)
        return backpointers, new_state


class CrfDecodeBackwardRnnCell(rnn_cell.RNNCell):
    def __init__(self, num_tags, **kwargs):
        super().__init__(**kwargs)
        self._num_tags = num_tags

    def compute_output_shape(self, input_shape):
        return 1, 1

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 1

    def __call__(self, inputs, state, scope=None):
        state = array_ops.squeeze(state, axis=[1])
        batch_size = array_ops.shape(inputs)[0]
        b_indices = math_ops.range(batch_size)
        indices = array_ops.stack([b_indices, state], axis=1)
        new_tags = array_ops.expand_dims(gen_array_ops.gather_nd(inputs, indices), axis=-1)
        return new_tags, new_tags
