import os
import re
import codecs
import ujson
import emoji
import string
import numpy as np
from tqdm import tqdm
from collections import Counter

np.random.seed(12345)
emoji_unicode = {v: k for k, v in emoji.EMOJI_UNICODE.items()}
punctuation = string.punctuation
digits = "0123456789"
glove_sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
PAD = "<PAD>"
UNK = "<UNK>"


def iob_to_iobes(labels):
    """IOB -> IOBES"""
    iob1_to_iob2(labels)
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


def iob1_to_iob2(labels):
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
    # create orthographic features
    ortho_char = []
    for char in word:
        if char in punctuation:
            ortho_char.append("p")
        elif char in digits:
            ortho_char.append("n")
        elif char.isupper():
            ortho_char.append("C")
        else:
            ortho_char.append("c")
    ortho_word = "".join(ortho_char)
    # create gazetteers
    # TODO
    if char_lowercase:
        char = [c for c in word.lower()]
    else:
        char = [c for c in word]
    if lowercase:
        word = word.lower()
    return word, char, ortho_word, ortho_char


def remove_emoji(line):
    line = "".join(c for c in line if c not in emoji_unicode)
    try:
        pattern = re.compile(u'([\U00002600-\U000027BF])|([\U0001F1E0-\U0001F6FF])')
    except re.error:
        pattern = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|'
                             u'([\uD83D][\uDE80-\uDEFF])')
    return pattern.sub(r'', line)


def process_token(line):
    word, *_, label = line.split(" ")
    if "page=http" in word or "http" in word:
        return None, None
    return word, label


def process_wnut_token(line):
    line = remove_emoji(line)
    line = line.lstrip().rstrip().split("\t")
    if len(line) != 2:
        return None, None
    word, label = line[0], line[1]
    if word.startswith("@") or word.startswith("https://") or word.startswith("http://"):
        return None, None
    if word in ["&gt;", "&quot;", "&lt;", ":D", ";)", ":)", "-_-", "=D", ":'", "-__-", ":P", ":p", "RT", ":-)", ";-)",
                ":(", ":/"]:
        return None, None
    if "&amp;" in word:
        word = word.replace("&amp;", "&")
    if word in ["/", "<"] and label == "O":
        return None, None
    if word.startswith("#"):
        word = word[1:]
        if label.startswith("B") or label.startswith("I"):
            word = word.capitalize()
    if len(word) == 0:
        return None, None
    return word, label


def raw_dataset_iter(filename, language, lowercase=True, char_lowercase=False):
    with codecs.open(filename, mode="r", encoding="utf-8") as f:
        words, chars, ortho_words, ortho_chars, labels = [], [], [], [], []
        for line in f:
            line = line.lstrip().rstrip()
            if len(line) == 0 or line.startswith("-DOCSTART-") or line.startswith("--------------") or \
                    line.startswith("=============="):
                if len(words) != 0:
                    yield words, chars, ortho_words, ortho_chars, labels
                    words, chars, ortho_words, ortho_chars, labels = [], [], [], [], []
            else:
                if "wnut" in language:
                    word, label = process_wnut_token(line)
                else:
                    word, label = process_token(line)
                if word is None or label is None:
                    continue
                word, char, ortho_word, ortho_char = word_convert(word, lowercase, char_lowercase=char_lowercase)
                words.append(word)
                ortho_words.append(ortho_word)
                chars.append(char)
                ortho_chars.append(ortho_char)
                labels.append(label)
        if len(words) != 0:
            yield words, chars, ortho_words, ortho_chars, labels


def load_dataset(filename, iobes, language, lowercase=True, char_lowercase=False):
    dataset = []
    for words, chars, ortho_words, ortho_chars, labels in raw_dataset_iter(filename, language, lowercase,
                                                                           char_lowercase):
        if iobes:
            labels = iob_to_iobes(labels)
        dataset.append({"words": words, "ortho_words": ortho_words, "chars": chars, "ortho_chars": ortho_chars,
                        "labels": labels})
    return dataset


def load_emb_vocab(data_path, language, dim):
    vocab = list()
    with codecs.open(data_path, mode="r", encoding="utf-8") as f:
        if "glove" in data_path:
            total = glove_sizes[data_path.split(".")[-3]]
        else:
            total = int(f.readline().lstrip().rstrip().split(" ")[0])
        for line in tqdm(f, total=total, desc="Load {} embedding vocabulary".format(language)):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2:
                continue
            if len(line) != dim + 1:
                continue
            word = line[0]
            vocab.append(word)
    return vocab


def filter_emb(word_dict, data_path, language, dim):
    vectors = np.zeros([len(word_dict), dim])
    with codecs.open(data_path, mode="r", encoding="utf-8") as f:
        if "glove" in data_path:
            total = glove_sizes[data_path.split(".")[-3]]
        else:
            total = int(f.readline().lstrip().rstrip().split(" ")[0])
        for line in tqdm(f, total=total, desc="Load {} embedding vectors".format(language)):
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
    ortho_word_counter = Counter()
    char_counter = Counter()
    ortho_char_counter = Counter()
    label_counter = Counter()
    for dataset in datasets:
        for record in dataset:
            for word in record["words"]:
                word_counter[word] += 1
            for ortho_word in record["ortho_words"]:
                ortho_word_counter[ortho_word] += 1
            for char in record["chars"]:
                for c in char:
                    char_counter[c] += 1
            for ortho_char in record["ortho_chars"]:
                for c in ortho_char:
                    ortho_char_counter[c] += 1
            for label in record["labels"]:
                label_counter[label] += 1
    return word_counter, ortho_word_counter, char_counter, ortho_char_counter, label_counter


def build_dataset(data, word_dict, ortho_word_dict, char_dict, ortho_char_dict, label_dict):
    dataset = []
    for record in data:
        words, ortho_words, chars_list, ortho_chars_list = [], [], [], []
        for word in record["words"]:
            words.append(word_dict[word] if word in word_dict else word_dict[UNK])
        for ortho_word in record["ortho_words"]:
            ortho_words.append(ortho_word_dict[ortho_word] if ortho_word in ortho_word_dict else ortho_word_dict[UNK])
        for char in record["chars"]:
            chars = [char_dict[c] if c in char_dict else char_dict[UNK] for c in char]
            chars_list.append(chars)
        for ortho_char in record["ortho_chars"]:
            ortho_chars = [ortho_char_dict[c] for c in ortho_char]
            ortho_chars_list.append(ortho_chars)
        labels = [label_dict[label] for label in record["labels"]]
        dataset.append({"words": words, "ortho_words": ortho_words, "chars": chars_list,
                        "ortho_chars": ortho_chars_list, "labels": labels})
    return dataset


def process_data(config):
    # load raw dataset
    train_data = load_dataset(config.train_file, iobes=config.iobes, language=config.language,
                              lowercase=config.word_lowercase, char_lowercase=config.char_lowercase)
    dev_data = load_dataset(config.dev_file, iobes=config.iobes, language=config.language,
                            lowercase=config.word_lowercase, char_lowercase=config.char_lowercase)
    test_data = load_dataset(config.test_file, iobes=config.iobes, language=config.language,
                             lowercase=config.word_lowercase, char_lowercase=config.char_lowercase)
    datasets = [train_data, dev_data, test_data]

    word_counter, ortho_word_counter, char_counter, ortho_char_counter, label_counter = build_token_counters(datasets)

    # create save path
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    # build word vocab
    word_vocab = [word for word, _ in word_counter.most_common()]
    ortho_word_vocab = [ortho_word for ortho_word, count in ortho_word_counter.most_common()
                        if count >= config.ortho_word_threshold]

    if config.word_vec is not None:
        emb_vocab = load_emb_vocab(config.word_vec_path, config.language, config.word_dim)
        word_vocab = list(set(word_vocab) & set(emb_vocab))
        tmp_word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
        vectors = filter_emb(tmp_word_dict, config.word_vec_path, config.language, config.word_dim)
        np.savez_compressed(config.word_vec, embeddings=np.asarray(vectors))

    word_vocab = [PAD, UNK] + word_vocab
    ortho_word_vocab = [PAD, UNK] + ortho_word_vocab

    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    ortho_word_dict = dict([(ortho_word, idx) for idx, ortho_word in enumerate(ortho_word_vocab)])

    if config.at:
        word_count = dict()
        for word, count in word_counter.most_common():
            if word in word_dict:
                word_count[word] = word_count.get(word, 0) + count
            else:
                word_count[UNK] = word_count.get(UNK, 0) + count
        sum_word_count = float(sum(list(word_count.values())))
        word_weight = [float(word_count[word]) / sum_word_count for word in word_vocab[1:]]
        np.savez_compressed(config.word_weight, embeddings=np.asarray(word_weight))

    # build char vocab
    char_vocab = [PAD, UNK] + [char for char, count in char_counter.most_common() if count >= config.char_threshold]
    ortho_char_vocab = [PAD] + [ortho_char for ortho_char, _ in ortho_char_counter.most_common()]

    char_dict = dict([(char, idx) for idx, char in enumerate(char_vocab)])
    ortho_char_dict = dict([(ortho_char, idx) for idx, ortho_char in enumerate(ortho_char_vocab)])

    if config.at:
        char_count = dict()
        for char, count in char_counter.most_common():
            if char in char_dict:
                char_count[char] = char_count.get(char, 0) + count
            else:
                char_count[UNK] = char_count.get(UNK, 0) + count
        sum_char_count = float(sum(list(char_count.values())))
        char_weight = [float(char_count[char]) / sum_char_count for char in char_vocab[1:]]  # exclude PAD
        np.savez_compressed(config.char_weight, embeddings=np.asarray(char_weight))

    # build label vocab
    label_vocab = ["O"] + [label for label, _ in label_counter.most_common() if label != "O"]
    label_dict = dict([(label, idx) for idx, label in enumerate(label_vocab)])

    if config.dev_for_train:
        train_data = train_data + dev_data

    # create indices dataset
    train_set = build_dataset(train_data, word_dict, ortho_word_dict, char_dict, ortho_char_dict, label_dict)
    dev_set = build_dataset(dev_data, word_dict, ortho_word_dict, char_dict, ortho_char_dict, label_dict)
    test_set = build_dataset(test_data, word_dict, ortho_word_dict, char_dict, ortho_char_dict, label_dict)
    vocab = {"word_dict": word_dict, "ortho_word_dict": ortho_word_dict, "char_dict": char_dict,
             "ortho_char_dict": ortho_char_dict, "label_dict": label_dict}

    # write to file
    write_json(os.path.join(config.save_path, "vocab.json"), vocab)
    write_json(os.path.join(config.save_path, "train.json"), train_set)
    write_json(os.path.join(config.save_path, "dev.json"), dev_set)
    write_json(os.path.join(config.save_path, "test.json"), test_set)
