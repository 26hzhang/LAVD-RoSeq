# LAVD and RoSeq

This is **TensorFlow** implementation for paper (`* indicates equal contributions`):

- Joey Tianyi Zhou*, Hao Zhang*, Di Jin, Xi Peng, Yang Xiao and Zhiguo Cao, "[RoSeq: Robust Sequence 
Labeling](https://ieeexplore.ieee.org/document/8709849)", IEEE Transactions on Neural Networks and Learning Systems 
(IEEE TNNLS), 2019.
- Joey Tianyi Zhou, Meng Fang, Hao Zhang, Chen Gong, Xi Peng, Zhiguo Cao and Rick Siow Mong Goh, "[Learning With 
Annotation of Various Degrees](https://ieeexplore.ieee.org/document/8611308)", IEEE Transactions on Neural Networks and 
Learning Systems (IEEE TNNLS), 2019.

RoSeq: Robust Sequence Labeling Architecture

![roseq-overview](/figures/roseq.jpg)

Learning With Annotation of Various Degrees Architecture

![lavd-overview](/figures/lavd.jpg)

## Requirements
- python 3.x with package tensorflow (`>=1.8.0`), ujson, emoji, matplotlib, tqdm, numpy

## Usage
To train a baseline model, run:
```shell script
$ python3 run_base.py --task conll2003_ner --raw_path <dataset_path> --save_path  <save path> \
                      --wordvec_path <pre-trained word vectors path>
```

To train a LAVD model, run:
```shell script
$ python3 run_lavd.py --task conll2003_ner --partial_rate 0.5 --raw_path <dataset_path> \
                      --wordvec_path <pre-trained word vectors path> --save_path  <save path> 
```
LAVD use 200d GloVe word embeddings, which are available [here](https://nlp.stanford.edu/projects/glove/).

To train a RoSeq model, run:
```shell script
python3 run_roseq.py --language spanish --at true --word_project true --focal_loss true
```
RoSeq used 100d GloVe word embeddings for English ([here](https://nlp.stanford.edu/projects/glove/)), while 50d word2vec 
word embeddings for other languages, which are available 
[here](http://www.limteng.com/research/2018/05/14/pretrained-word-embeddings.html).

**Note**: to obtain the main results of Table 3 in "RoSeq: Robust Sequence Labeling", you can download 

- The trained weights from [Box Drive (ckpt)](https://app.box.com/s/vrw92w0rugqp7mtz5oqdmx1wfo0yf7r8) and save them 
to `./ckpt/` folder.
- The processed dataset from [Box Drive (data)](https://app.box.com/s/9qoqm6mx1it42chukmyogeb0f2ylgqit) and save them 
to `./datasets/data/` folder.

Then run the following commands.

For Spanish (CoNLL-2002 Spanish NER):
```shell script
python3 train_roseq.py --language spanish --at true --word_project true --concat_rnn false --train false
```

For Dutch (CoNLL-2002 Dutch NER):
```shell script
python3 train_roseq.py --language dutch --at true --word_project true --concat_rnn false --train false
```

For English (CoNLL-2003 English NER):
```shell script
python3 train_roseq.py --language english --at true --word_project true --concat_rnn false --word_dim 200 --train false
```

For WNUT-2016 (WNUT-2016 Twitter NER):
```shell script
python3 train_roseq.py --language wnut2016 --dev_for_train true --use_orthographic true --at true --concat_rnn true \
                       --focal_loss true --word_project true --train false
```

For WNUT-2017 (WNUT-2017 Twitter NER):
```shell script
python3 train_roseq.py --language wnut2017 --at true --use_orthographic true --focal_loss true --word_project false \
                       --concat_rnn true --train false
```

## Citation
If you feel this project helpful to your research, please cite our work.
```
@ARTICLE{8709849,
    author={J. T. {Zhou} and H. {Zhang} and D. {Jin} and X. {Peng} and Y. {Xiao} and Z. {Cao}},
    journal={IEEE Transactions on Neural Networks and Learning Systems},
    title={RoSeq: Robust Sequence Labeling},
    year={2019},
    pages={1-11},
    doi={10.1109/TNNLS.2019.2911236},
    ISSN={2162-2388}
}
```
and
```
@article{8611308,
    author={J. T. {Zhou} and M. {Fang} and H. {Zhang} and C. {Gong} and X. {Peng} and Z. {Cao} and R. S. M. {Goh}},
    journal={IEEE Transactions on Neural Networks and Learning Systems},
    title={Learning With Annotation of Various Degrees},
    year={2019},
    volume={30},
    number={9},
    pages={2794-2804},
    doi={10.1109/TNNLS.2018.2885854},
    ISSN={2162-2388},
    month={Sep.}
}
```
