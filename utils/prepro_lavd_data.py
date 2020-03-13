import os
import codecs
import ujson
import random
import numpy as np
from tqdm import tqdm
from collections import Counter

np.random.seed(12345)
glove_sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
PAD = "<PAD>"
UNK = "<UNK>"


def iob_to_iobes(labels):
    """IOB -> IOBES"""
    iob2(labels)
    new_tags = []
    for i, tag in enumerate(labels):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(labels) and labels[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(labels) and labels[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iob2(labels):
    """Check that tags have a valid IOB format. Tags in IOB1 format are converted to IOB2."""
    for i, tag in enumerate(labels):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or labels[i - 1] == 'O':  # conversion IOB1 to IOB2
            labels[i] = 'B' + tag[1:]
        elif labels[i - 1][1:] == tag[1:]:
            continue
        else:
            labels[i] = 'B' + tag[1:]
    return True


def write_json(filename, dataset):
    with codecs.open(filename, mode="w", encoding="utf-8") as f:
        ujson.dump(dataset, f)


def word_convert(word, lowercase=True, char_lowercase=False):
    if char_lowercase:
        char = [c for c in word.lower()]
    else:
        char = [c for c in word]
    if lowercase:
        word = word.lower()
    return word, char


def raw_dataset_iter(filename, lowercase=True, char_lowercase=False):
    with codecs.open(filename, mode="r", encoding="utf-8") as f:
        words, chars, labels = [], [], []
        for line in f:
            line = line.lstrip().rstrip()
            if len(line) == 0 or line.startswith("-DOCSTART-"):
                if len(words) != 0:
                    yield words, chars, labels
                    words, chars, labels = [], [], []
            else:
                word, *_, label = line.split(" ")
                word, char = word_convert(word, lowercase=lowercase, char_lowercase=char_lowercase)
                words.append(word)
                chars.append(char)
                labels.append(label)
        if len(words) != 0:
            yield words, chars, labels


def load_dataset(filename, iobes, lowercase=True, char_lowercase=False):
    dataset = []
    for words, chars, labels in raw_dataset_iter(filename, lowercase, char_lowercase):
        if iobes:
            labels = iob_to_iobes(labels)
        dataset.append({"words": words, "chars": chars, "labels": labels})
    return dataset


def load_emb_vocab(data_path, dim):
    vocab = list()
    with codecs.open(data_path, mode="r", encoding="utf-8") as f:
        if "glove" in data_path:
            total = glove_sizes[data_path.split(".")[-3]]
        else:
            total = int(f.readline().lstrip().rstrip().split(" ")[0])
        for line in tqdm(f, total=total, desc="Load embedding vocabulary"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2:
                continue
            if len(line) != dim + 1:
                continue
            word = line[0]
            vocab.append(word)
    return vocab


def filter_emb(word_dict, data_path, dim):
    vectors = np.zeros([len(word_dict), dim])
    with codecs.open(data_path, mode="r", encoding="utf-8") as f:
        if "glove" in data_path:
            total = glove_sizes[data_path.split(".")[-3]]
        else:
            total = int(f.readline().lstrip().rstrip().split(" ")[0])
        for line in tqdm(f, total=total, desc="Load embedding vectors"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2:
                continue
            if len(line) != dim + 1:
                continue
            word = line[0]
            if word in word_dict:
                vector = [float(x) for x in line[1:]]
                word_idx = word_dict[word]
                vectors[word_idx] = np.asarray(vector)
    return vectors


def build_token_counters(datasets):
    word_counter = Counter()
    char_counter = Counter()
    label_counter = Counter()
    for dataset in datasets:
        for record in dataset:
            for word in record["words"]:
                word_counter[word] += 1
            for char in record["chars"]:
                for c in char:
                    char_counter[c] += 1
            for label in record["labels"]:
                label_counter[label] += 1
    return word_counter, char_counter, label_counter


def build_dataset(data, word_dict, char_dict, label_dict, mode=0, rate=0.5):
    dataset = []
    for record in data:
        chars_list = []
        words = []
        for word in record["words"]:
            words.append(word_dict[word] if word in word_dict else word_dict[UNK])
        for char in record["chars"]:
            chars = [char_dict[c] if c in char_dict else char_dict[UNK] for c in char]
            chars_list.append(chars)
        if mode == 0:  # labeled
            labels = [label_dict[label] for label in record["labels"]]
            dataset.append({"words": words, "chars": chars_list, "labels": labels})
        elif mode == 1:  # partially labeled
            labels = [label_dict[label] for label in record["labels"]]
            label_mask = np.asarray([0 if v < rate else 1 for v in np.random.rand(len(labels))])
            labels = np.asarray(labels) * label_mask
            dataset.append({"words": words, "chars": chars_list, "labels": labels.tolist(),
                            "label_mask": label_mask.tolist()})
        elif mode == 2:  # unlabeled
            labels = [0 for _ in record["labels"]]
            dataset.append({"words": words, "chars": chars_list, "labels": labels})
        else:
            raise ValueError("Unknown label process mode!!! Support: [0: labeled | 1: partial | 2: unlabeled]")
    return dataset


def split_dataset(dataset, n):
    if n is None or type(n) != int or n <= 1 or n >= len(dataset):
        return dataset
    step = len(dataset) // n
    data_list = []
    idx = 0
    for i in range(n):
        if i == n - 1:
            data_list.append(dataset[idx:])
            break
        data_list.append(dataset[idx: idx + step])
        idx = idx + step
    return data_list


def process_word_token(word_counter, config):
    if config.wordvec_path is not None:
        word_vocab = [word for word, _ in word_counter.most_common()]
        emb_vocab = load_emb_vocab(config.wordvec_path, config.word_dim)
        word_vocab = list(set(word_vocab) & set(emb_vocab))
        tmp_word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
        vectors = filter_emb(tmp_word_dict, config.wordvec_path, config.word_dim)
        np.savez_compressed(config.wordvec, embeddings=np.asarray(vectors))
    else:
        word_vocab = [word for word, count in word_counter.most_common() if count >= config.word_threshold]
    word_vocab = [PAD, UNK] + word_vocab
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    return word_dict


def process_char_token(char_counter, config):
    char_vocab = [PAD, UNK] + [char for char, count in char_counter.most_common() if count >= config.char_threshold]
    char_dict = dict([(char, idx) for idx, char in enumerate(char_vocab)])
    return char_dict


def process_label_token(label_counter):
    label_vocab = ["O"] + [label for label, _ in label_counter.most_common() if label != "O"]
    label_dict = dict([(label, idx) for idx, label in enumerate(label_vocab)])
    return label_dict


def write_to_jsons(datasets, files, save_path):
    for dataset, file in zip(datasets, files):
        write_json(os.path.join(save_path, file), dataset)


def process_data(config):
    # load raw datasets
    train_data = load_dataset(config.train_file, config.iobes, config.word_lowercase, config.char_lowercase)
    dev_data = load_dataset(config.dev_file, config.iobes, config.word_lowercase, config.char_lowercase)
    test_data = load_dataset(config.test_file, config.iobes, config.word_lowercase, config.char_lowercase)
    datasets = [train_data, dev_data, test_data]
    # build token counters
    word_counter, char_counter, label_counter = build_token_counters(datasets)
    # create save path
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    # build word vocab
    word_dict = process_word_token(word_counter, config)
    # build char vocab
    char_dict = process_char_token(char_counter, config)
    # build label vocab
    label_dict = process_label_token(label_counter)
    # create indices datasets and write to files
    if "dCRF" in config.model_name:
        random.shuffle(train_data)
        train_folds = split_dataset(train_data, config.folds)  # 10 folds
        train_set = dict()
        for i in range(config.folds):
            train_fold_set = build_dataset(train_folds[i], word_dict, char_dict, label_dict)
            train_set["fold_{}".format(i)] = train_fold_set
        train_set_p = dict()
        for i in range(config.folds):
            train_fold_set = build_dataset(train_folds[i], word_dict, char_dict, label_dict, 1, config.partial_rate)
            train_set_p["fold_{}".format(i)] = train_fold_set
        train_set_u = dict()
        for i in range(config.folds):
            train_fold_set = build_dataset(train_folds[i], word_dict, char_dict, label_dict, 2)
            train_set_u["fold_{}".format(i)] = train_fold_set
        dev_set = build_dataset(dev_data, word_dict, char_dict, label_dict)
        test_set = build_dataset(test_data, word_dict, char_dict, label_dict)
        vocab = {"word_dict": word_dict, "char_dict": char_dict, "label_dict": label_dict}
        write_to_jsons([train_set, train_set_p, train_set_u, dev_set, test_set, vocab],
                       ["train.json", "train_p.json", "train_u.json", "dev.json", "test.json", "vocab.json"],
                       config.save_path)
    else:
        train_set = build_dataset(train_data, word_dict, char_dict, label_dict)
        dev_set = build_dataset(dev_data, word_dict, char_dict, label_dict)
        test_set = build_dataset(test_data, word_dict, char_dict, label_dict)
        vocab = {"word_dict": word_dict, "char_dict": char_dict, "label_dict": label_dict}
        write_to_jsons([train_set, dev_set, test_set, vocab], ["train.json", "dev.json", "test.json", "vocab.json"],
                       config.save_path)
