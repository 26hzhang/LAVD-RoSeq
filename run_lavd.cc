#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/globals.h"
//#include "../utils/getpid.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace dynet;

float pdrop = 0.5;
unsigned LAYERS = 1;
//unsigned INPUT_DIM = 128;
unsigned INPUT_DIM = 40;
unsigned HIDDEN_DIM = 128;
unsigned MLP_HIDDEN_DIM = 32;

unsigned TAG_HIDDEN_DIM = 32;
unsigned TAG_DIM = 32;
unsigned TAG_SIZE = 0;

unsigned VOCAB_SIZE = 0;

bool eval = false;

int MAX_EPOCH = 10;

dynet::Dict d;
// use one dictionary
std::map<int, vector<float>> embeddings_all;

dynet::Dict td;
int kNONE;
int kSOS;
int kEOS;

template <class Builder>
struct DeepCRF3
{
  LookupParameter p_w;

  Builder l2rbuilder;
  Builder r2lbuilder;

  //deep information
  Parameter p_W_lstm_fwd;
  Parameter p_W_lstm_rev;
  Parameter p_bias;

  //crf part
  Parameter p_C;
  Parameter p_A_transition;

  //unsupervised part
  Parameter p_decoder;

  enum
  {
    Labelled = 0,
    Unlabelled = 1
  };

  explicit DeepCRF3(Model &model) : l2rbuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model),
                                    r2lbuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)
  {
    p_w = model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM});

    for (std::map<int, vector<float>>::iterator iter = embeddings_all.begin(); iter != embeddings_all.end(); iter++)
    {
      p_w.initialize(iter->first, iter->second);
    }

    //p_l2th = model.add_parameters({TAG_HIDDEN_DIM, HIDDEN_DIM});
    //p_r2th = model.add_parameters({TAG_HIDDEN_DIM, HIDDEN_DIM});
    //p_thbias = model.add_parameters({TAG_HIDDEN_DIM});

    //p_th2t = model.add_parameters({TAG_SIZE, TAG_HIDDEN_DIM});
    //p_tbias = model.add_parameters({TAG_SIZE});

    //crf
    p_W_lstm_fwd = model.add_parameters({MLP_HIDDEN_DIM, HIDDEN_DIM});
    p_W_lstm_rev = model.add_parameters({MLP_HIDDEN_DIM, HIDDEN_DIM});

    p_C = model.add_parameters({TAG_SIZE, MLP_HIDDEN_DIM});
    p_bias = model.add_parameters({MLP_HIDDEN_DIM});

    p_decoder = model.add_parameters({TAG_SIZE, VOCAB_SIZE});
    //p_decoder = model.add_parameters({VOCAB_SIZE, TAG_SIZE});
    p_A_transition = model.add_parameters({TAG_SIZE, TAG_SIZE});
  }

  void add_pretained_embeddings(LookupParameter &p_w, std::map<int, vector<float>> &embeddings)
  {
    // NOW SET THEIR VALUES!
    // iterate over map<int, floats> for english
    // copy floats into p_we[i] for each word index i in map
  }

  // return Expression of total loss
  Expression BuildTaggingGraph(const vector<int> &sent, const vector<int> &tags, ComputationGraph &cg, int labelling_type)
  {
    const unsigned slen = sent.size();
    l2rbuilder.new_graph(cg); // reset RNN builder for new graph
    l2rbuilder.start_new_sequence();
    r2lbuilder.new_graph(cg); // reset RNN builder for new graph
    r2lbuilder.start_new_sequence();

    Expression i_W_lstm_fwd = parameter(cg, p_W_lstm_fwd);
    Expression i_W_lstm_rev = parameter(cg, p_W_lstm_rev);
    Expression i_C = parameter(cg, p_C);
    Expression i_bias = parameter(cg, p_bias);

    Expression i_A_transition = parameter(cg, p_A_transition);

    vector<Expression> i_words(slen);
    vector<Expression> fwds(slen);
    vector<Expression> revs(slen);

    vector<Expression> fs(slen);
    vector<Expression> Gs(slen);

    vector<vector<Expression>> alpha_matrix(slen);
    vector<Expression> numerator_vector;

    Expression i_decoder = parameter(cg, p_decoder);

    vector<vector<dynet::real>> delta_matrix(slen);
    vector<vector<unsigned>> delta_matrix_bp(slen);

    // deep architecture
    // read sequence from left to right
    l2rbuilder.add_input(lookup(cg, p_w, kSOS));
    for (unsigned t = 0; t < slen; ++t)
    {
      i_words[t] = const_lookup(cg, p_w, sent[t]);
      if (!eval)
      {
        i_words[t] = noise(i_words[t], 0.1);
      }
      fwds[t] = l2rbuilder.add_input(i_words[t]);
    }

    // read sequence from right to left
    r2lbuilder.add_input(const_lookup(cg, p_w, kEOS));
    for (unsigned t = 0; t < slen; ++t)
      revs[slen - t - 1] = r2lbuilder.add_input(i_words[slen - t - 1]);

    // crf architecture
    for (unsigned cur_step = 0; cur_step < slen; cur_step++)
    {
      fs[cur_step] = transpose(i_C * tanh(i_W_lstm_fwd * fwds[cur_step] + i_W_lstm_rev * revs[cur_step] + i_bias));
      Gs[cur_step] = i_A_transition;
    }

    //FLAG for the labelling status
    int FLAG = labelling_type;
    //cerr << "\nFLAG = " << FLAG;
    switch (FLAG)
    {
    case Labelled:
    {
      //cerr << "\nProcess the labelled data";
      vector<Expression> alphas(slen);
      //		transpose(colwise_add(transpose(Gs[cur_step]), sum_cols(transpose(fs[0]))));
      alphas[0] = reshape(fs[0], {TAG_SIZE, 1});
      numerator_vector.push_back(pick(alphas[0], tags[0]));
      //cerr << "TAG_SIZE" << TAG_SIZE;
      for (unsigned cur_step = 1; cur_step < slen; cur_step++)
      {
        // phi matrix [y' (previous tag)][y (current tag)]
        //Expression phi_matrix = Gs[cur_step] + f_broadcast;
        Expression f = sum_cols(transpose(fs[cur_step]));
        Expression phi_matrix = transpose(colwise_add(transpose(Gs[cur_step]), f));

        numerator_vector.push_back(pick(select_cols(phi_matrix, {(unsigned)tags[cur_step]}), (unsigned)tags[cur_step - 1]));
        //LOLCAT(numerator_vector.back());

        //Expression alpha_matrix_before_lse = alpha_prev_broadcast + phi_matrix;
        Expression alpha_matrix_before_lse = colwise_add(phi_matrix, sum_cols(alphas[cur_step - 1]));

        vector<Expression> alpha_ys_after_lse_vector;

        // TODO: optimize
        for (unsigned cur_tag = 0; cur_tag < TAG_SIZE; cur_tag++)
        {
          Expression col = select_cols(alpha_matrix_before_lse, {cur_tag});
          //Expression col_phi = select_cols(phi_matrix, {cur_tag});

          //vector<Expression> alpha_y_before_lse_vector;

          //LOLCAT(col);
          //alpha_ys_after_lse_vector.push_back(logsumexp(alpha_y_before_lse_vector));
          alpha_ys_after_lse_vector.push_back(logsumexp_vectorized(col));
        }
        alphas[cur_step] = reshape(concatenate(alpha_ys_after_lse_vector), {TAG_SIZE, 1});
        //LOLCAT(alphas[cur_step]);
      }

      Expression logZ = logsumexp_vectorized(alphas[slen - 1]);
      Expression numerator = sum(numerator_vector);
      Expression loss = -(numerator - logZ);
      return loss;
    }
    case Unlabelled:
    {
      //cerr <<"\nProcess the unlabelled data";
      vector<Expression> alphas(slen);
      vector<Expression> numberator_uns(slen);
      //transpose(colwise_add(transpose(Gs[cur_step]), sum_cols(transpose(fs[0]))));
      alphas[0] = reshape(fs[0], {TAG_SIZE, 1});
      // the unsupervised part
      numberator_uns[0] = reshape(fs[0], {TAG_SIZE, 1});

      for (unsigned cur_step = 1; cur_step < slen; cur_step++)
      {
        //phi matrix [y' (previous tag)][y (current tag)]
        //Expression phi_matrix = Gs[cur_step] + f_broadcast;
        Expression f = sum_cols(transpose(fs[cur_step])); // TAG_SIZE, 1
        Expression phi_matrix = transpose(colwise_add(transpose(Gs[cur_step]), f));

        Expression alpha_matrix_before_lse = colwise_add(phi_matrix, sum_cols(alphas[cur_step - 1]));

        Expression theta = select_cols(i_decoder, {(unsigned)sent[cur_step]}); //TAG_SIZE,1
        Expression theta_dist = log_softmax(theta);

        Expression numberator_uns_matrix_before_lse = colwise_add(alpha_matrix_before_lse, theta_dist);

        vector<Expression> alpha_ys_after_lse_vector;
        vector<Expression> numberator_uns_ys_after_lse_vector;

        //TODO: optimize
        for (unsigned cur_tag = 0; cur_tag < TAG_SIZE; cur_tag++)
        {
          Expression col = select_cols(alpha_matrix_before_lse, {cur_tag});
          Expression col_numberator_uns = select_cols(numberator_uns_matrix_before_lse, {cur_tag});

          alpha_ys_after_lse_vector.push_back(logsumexp_vectorized(col));
          numberator_uns_ys_after_lse_vector.push_back(logsumexp_vectorized(col_numberator_uns));
        }
        alphas[cur_step] = reshape(concatenate(alpha_ys_after_lse_vector), {TAG_SIZE, 1});
        numberator_uns[cur_step] = reshape(concatenate(numberator_uns_ys_after_lse_vector), {TAG_SIZE, 1});
        //LOLCAT(alphas[cur_step]);
      }

      Expression logZ = logsumexp_vectorized(alphas[slen - 1]);
      Expression logNumberator_uns = logsumexp_vectorized(numberator_uns[slen - 1]);
      Expression loss = -(logNumberator_uns - logZ);
      return loss;
    }
    }
  }

  // return Expression of total loss
  Expression Evaluate(const vector<int> &sent, const vector<int> &tags, ComputationGraph &cg, double *cor = 0, unsigned *ntagged = 0)
  {
    //cerr <<"\nStart evaluation";
    const unsigned slen = sent.size();
    l2rbuilder.new_graph(cg); // reset RNN builder for new graph
    l2rbuilder.start_new_sequence();
    r2lbuilder.new_graph(cg); // reset RNN builder for new graph
    r2lbuilder.start_new_sequence();

    Expression i_W_lstm_fwd = parameter(cg, p_W_lstm_fwd);
    Expression i_W_lstm_rev = parameter(cg, p_W_lstm_rev);
    Expression i_C = parameter(cg, p_C);
    Expression i_bias = parameter(cg, p_bias);

    Expression i_A_transition = parameter(cg, p_A_transition);

    vector<Expression> i_words(slen);
    vector<Expression> fwds(slen);
    vector<Expression> revs(slen);

    vector<Expression> fs(slen);
    vector<Expression> Gs(slen);

    vector<vector<Expression>> alpha_matrix(slen);
    vector<Expression> numerator_vector;
    Expression i_decoder = parameter(cg, p_decoder);

    vector<vector<dynet::real>> delta_matrix(slen);
    vector<vector<unsigned>> delta_matrix_bp(slen);

    // deep architecture
    // read sequence from left to right
    l2rbuilder.add_input(lookup(cg, p_w, kSOS));
    for (unsigned t = 0; t < slen; ++t)
    {
      i_words[t] = const_lookup(cg, p_w, sent[t]);
      if (!eval)
      {
        i_words[t] = noise(i_words[t], 0.1);
      }
      fwds[t] = l2rbuilder.add_input(i_words[t]);
    }

    // read sequence from right to left
    r2lbuilder.add_input(const_lookup(cg, p_w, kEOS));
    for (unsigned t = 0; t < slen; ++t)
      revs[slen - t - 1] = r2lbuilder.add_input(i_words[slen - t - 1]);

    // crf architecture
    for (unsigned cur_step = 0; cur_step < slen; cur_step++)
    {
      fs[cur_step] = transpose(i_C * tanh(i_W_lstm_fwd * fwds[cur_step] + i_W_lstm_rev * revs[cur_step] + i_bias));
      Gs[cur_step] = i_A_transition;
    }

    vector<Expression> alphas(slen);
    //		transpose(colwise_add(transpose(Gs[cur_step]), sum_cols(transpose(fs[0]))));
    alphas[0] = reshape(fs[0], {TAG_SIZE, 1});
    numerator_vector.push_back(pick(alphas[0], tags[0]));

    //decoding
    delta_matrix[0].resize(TAG_SIZE);
    delta_matrix_bp[0].resize(TAG_SIZE);
    for (unsigned cur_tag = 0; cur_tag < TAG_SIZE; cur_tag++)
    {
      delta_matrix[0][cur_tag] = as_scalar(pick(alphas[0], cur_tag).value());
    }

    for (unsigned cur_step = 1; cur_step < slen; cur_step++)
    {
      // phi matrix [y' (previous tag)][y (current tag)]
      //Expression phi_matrix = Gs[cur_step] + f_broadcast;
      Expression f = sum_cols(transpose(fs[cur_step]));
      Expression phi_matrix = transpose(colwise_add(transpose(Gs[cur_step]), f));

      numerator_vector.push_back(pick(select_cols(phi_matrix, {(unsigned)tags[cur_step]}), (unsigned)tags[cur_step - 1]));
      //LOLCAT(numerator_vector.back());

      //Expression alpha_matrix_before_lse = alpha_prev_broadcast + phi_matrix;
      Expression alpha_matrix_before_lse = colwise_add(phi_matrix, sum_cols(alphas[cur_step - 1]));

      vector<Expression> alpha_ys_after_lse_vector;

      //decoding
      delta_matrix[cur_step].resize(TAG_SIZE);
      delta_matrix_bp[cur_step].resize(TAG_SIZE);

      // TODO: optimize
      for (unsigned cur_tag = 0; cur_tag < TAG_SIZE; cur_tag++)
      {
        Expression col = select_cols(alpha_matrix_before_lse, {cur_tag});
        Expression col_phi = select_cols(phi_matrix, {cur_tag});

        //decoding
        delta_matrix[cur_step][cur_tag] = numeric_limits<float>::lowest();
        delta_matrix_bp[cur_step][cur_tag] = -1;
        for (unsigned prev_tag = 0; prev_tag < TAG_SIZE; prev_tag++)
        {
          Expression phi = pick(col_phi, prev_tag);
          dynet::real cur = as_scalar(phi.value()) + delta_matrix[cur_step - 1][prev_tag];
          if (delta_matrix[cur_step][cur_tag] < cur)
          {
            delta_matrix[cur_step][cur_tag] = cur;
            delta_matrix_bp[cur_step][cur_tag] = prev_tag;
          }
        }
        assert(delta_matrix_bp[cur_step][cur_tag] != -1);

        //LOLCAT(col);
        //alpha_ys_after_lse_vector.push_back(logsumexp(alpha_y_before_lse_vector));
        alpha_ys_after_lse_vector.push_back(logsumexp_vectorized(col));
      }
      alphas[cur_step] = reshape(concatenate(alpha_ys_after_lse_vector), {TAG_SIZE, 1});
      //LOLCAT(alphas[cur_step]);
    }

    //decoding
    vector<int> predict(slen);
    dynet::real cur_max = numeric_limits<float>::lowest();
    unsigned max_index = -1;
    for (unsigned i = 0; i < delta_matrix[slen - 1].size(); i++)
    {
      if (cur_max < delta_matrix[slen - 1][i])
      {
        cur_max = delta_matrix[slen - 1][i];
        max_index = i;
      }
    }
    bool display = false;
    if (cur_max != numeric_limits<float>::min() && max_index != -1)
    {
      //assert (cur_max != FLT_MIN);
      //assert (max_index != -1);
      predict[slen - 1] = max_index;
      for (unsigned t = slen - 1; t > 0; t--)
      {
        predict[t - 1] = delta_matrix_bp[t][predict[t]];
      }
      assert(predict.size() == slen);
      for (unsigned i = 0; i < slen; i++)
      {
        if (tags[i] == predict[i])
        {
          (*cor)++;
        }
        if (display)
        {
          cerr << d.convert(sent[i]) << "|" << td.convert(tags[i])
               << "|" << td.convert(predict[i]) << " ";
        }
      }
      if (display)
      {
        cerr << endl;
      }
    }

    if (ntagged)
    {
      (*ntagged) += slen;
    }

    Expression logZ = logsumexp_vectorized(alphas[slen - 1]);
    Expression numerator = sum(numerator_vector);
    Expression loss = -(numerator - logZ);
    return loss;
  }

  Expression logsumexp_vectorized(Expression col)
  {
    Expression col_max = pick(kmax_pooling(transpose(col), 1), (unsigned)0);
    Expression diff = colwise_add(transpose(col), -col_max);
    Expression lse = col_max + log(sum_cols(exp(diff)));
    return lse;
  }
};

vector<string> split(const string &s, char c)
{
  vector<string> parts;
  string::size_type i = 0;
  string::size_type j = s.find(c);

  while (j != string::npos)
  {
    parts.push_back(s.substr(i, j - i));
    i = ++j;
    j = s.find(c, j);

    if (j == string::npos)
      parts.push_back(s.substr(i, s.length()));
  }
  return parts;
}

// import pretrained word embeddings and construct a dictionary
std::map<int, vector<float>> importEmbeddings(string inputFile)
{
  std::map<int, vector<float>> w_embeddings;
  unsigned welc = 0;
  unsigned dim = 0;
  cerr << "Loading word embeddings " << inputFile << "...\n";

  string line;
  ifstream in(inputFile);
  assert(in);
  while (getline(in, line))
  {

    // word, (val1, val2, ...)
    // if use multiCCA, then the space is ' '
    vector<string> splitedline = split(line, ' ');

    string word = splitedline[0];
    auto w_id = d.convert(word);
    vector<float> w_embedding;

    for (unsigned i = 1; i < splitedline.size(); i++)
    {
      w_embedding.push_back(std::stof(splitedline[i]));
    }
    if (dim == 0)
      dim = splitedline.size() - 1;
    else
      assert(dim == splitedline.size() - 1);
    w_embeddings[w_id] = w_embedding;

    ++welc;
  }

  cerr << welc << " word embeddings, " << dim << " dimensions \n";
  return w_embeddings;
}

vector<pair<vector<int>, vector<int>>> loadCorpus(string inputFile)
{
  cerr << "Loading data " << inputFile << "...\n";
  vector<pair<vector<int>, vector<int>>> data;
  string line;
  int lc = 0;
  int toks = 0;
  ifstream in(inputFile);
  assert(in);
  while (getline(in, line))
  {
    ++lc;
    int nc = 0;
    vector<int> x, y;
    read_sentence_pair(line, x, d, y, td);
    assert(x.size() == y.size());
    if (x.size() == 0)
    {
      cerr << line << endl;
      abort();
    }
    data.push_back(make_pair(x, y));
    for (unsigned i = 0; i < y.size(); ++i)
    {
      if (y[i] != kNONE)
      {
        ++nc;
      }
    }
    if (nc == 0)
    {
      cerr << "No tagged tokens in line " << lc << endl;
      abort();
    }
    toks += x.size();
  }
  cerr << lc << " lines, " << toks << " tokens, " << d.size() << " types\n";
  cerr << "Tags: " << td.size() << endl;
  return data;
}

void trainLabelled(vector<pair<vector<int>, vector<int>>> &training, DeepCRF3<LSTMBuilder> &lm, Trainer *sgd)
{
  int report = 0;
  int report_every_i = 100;
  double loss = 0;
  int ttags = 0;
  unsigned budget = training.size();
  cerr << "\nTraining labelled data: size = " << budget << " sentences";
  for (unsigned i = 0; i < budget; ++i)
  {
    // build graph for this instance
    ComputationGraph cg;
    auto &sent = training[i];
    int l_type = 0;
    //cerr << "Compute loss\n";
    Expression loss_expr = lm.BuildTaggingGraph(sent.first, sent.second, cg, l_type);
    loss += as_scalar(cg.forward(loss_expr));
    cg.backward(loss_expr);
    sgd->update(1.0);
    ttags += sent.first.size();
    ++report;
    if (report % report_every_i == 0)
    {
      cerr << "\n***TRAIN:labelled E = " << (loss / ttags);
    }
  }
}

void trainUnlabelled(vector<pair<vector<int>, vector<int>>> &training, DeepCRF3<LSTMBuilder> &lm, Trainer *sgd)
{
  int report = 0;
  int report_every_i = 100;
  int ttoks = 0;
  double loss = 0;
  unsigned budget = training.size();
  cerr << "\nTraining unlabelled data: size = " << budget << " sentences";
  for (unsigned i = 0; i < budget; ++i)
  {
    // build graph for this instance
    ComputationGraph cg;
    auto &sent = training[i];
    int l_type = 1;
    Expression loss_expr = lm.BuildTaggingGraph(sent.first, sent.second, cg, l_type);
    loss += as_scalar(cg.forward(loss_expr));
    cg.backward(loss_expr);
    sgd->update(1.0);
    ttoks += sent.first.size();
    ++report;
    if (report % report_every_i == 0)
    {
      cerr << "\n***TRAIN:unlabelled E = " << (loss / ttoks);
    }
  }
}

// evaluate the dev and test data
void testData(vector<pair<vector<int>, vector<int>>> &test, vector<pair<vector<int>, vector<int>>> &dev, DeepCRF3<LSTMBuilder> &lm, double &best)
{
  //dev data
  double loss = 0;
  unsigned tags = 0;
  double corr = 0;
  eval = true;
  //lm.p_th2t->scale_parameters(pdrop);
  for (auto &sent : dev)
  {
    ComputationGraph cg;
    Expression loss_expr = lm.Evaluate(sent.first, sent.second, cg, &corr, &tags);
    loss += as_scalar(cg.forward(loss_expr));
  }
  //lm.p_th2t->scale_parameters(1/pdrop);
  eval = false;
  cerr << "\n***DEV  E = " << (loss / tags) << " ppl=" << exp(loss / tags) << " acc=" << (corr / tags) << ' ';
  // test data
  loss = 0;
  tags = 0;
  corr = 0;
  eval = true;
  //lm.p_th2t->scale_parameters(pdrop);
  for (auto &sent : test)
  {
    ComputationGraph cg;
    Expression loss_expr = lm.Evaluate(sent.first, sent.second, cg, &corr, &tags);
    loss += as_scalar(cg.forward(loss_expr));
  }
  //lm.p_th2t->scale_parameters(1/pdrop);
  eval = false;
  if (loss < best)
  {
    best = loss;
    /*ofstream out(fname);
    boost::archive::text_oarchive oa(out);
    oa << model;*/
  }
  cerr << "\n***TEST  E = " << (loss / tags) << " ppl=" << exp(loss / tags) << " acc=" << (corr / tags) << ' ';
}

/*
$./train_nncrf3 embeddings_file labelled_file unlabelled_file dev_file test_file max_epoch
*/

int main(int argc, char **argv)
{
  dynet::initialize(argc, argv);

  vector<pair<vector<int>, vector<int>>> data_labelled, data_unlabelled;
  vector<pair<vector<int>, vector<int>>> dev, test;
  string line;
  int tlc = 0;
  int ttoks = 0;

  // load embeddings and build dictionaries
  // import one language
  kNONE = td.convert("*");
  kSOS = d.convert("<s>");
  kEOS = d.convert("</s>");

  string inputEmbeddingFile = argv[1];
  embeddings_all = importEmbeddings(inputEmbeddingFile);
  d.freeze(); // no new word types allowed
  d.set_unk("<UNK>");
  VOCAB_SIZE = d.size();
  cerr << "Size of Dictionary of all languages: " << VOCAB_SIZE << endl;

  string inputFile = argv[2];
  data_labelled = loadCorpus(inputFile);
  inputFile = argv[3];
  data_unlabelled = loadCorpus(inputFile);
  td.freeze();
  TAG_SIZE = td.size();

  //the dev and test data; they are from the target language
  inputFile = argv[4];
  dev = loadCorpus(inputFile);
  inputFile = argv[5];
  test = loadCorpus(inputFile);

  // initialise the model
  ostringstream os;
  os << "tagger"
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;
  double best = 9e+99;

  Model model;
  bool use_momentum = true;
  Trainer *sgd = nullptr;
  if (use_momentum)
    sgd = new MomentumSGDTrainer(model);
  else
    sgd = new SimpleSGDTrainer(model);

  DeepCRF3<LSTMBuilder> lm(model);
  if (argc == 7)
  {
    MAX_EPOCH = atoi(argv[6]);
  }
  /*if (argc == 4)
  {
    string fname = argv[3];
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }*/

  unsigned report_every_i = 50;
  unsigned dev_every_i_reports = 25;
  /*unsigned si = training.size(); // a combined training datasets
  vector<unsigned> order(training.size());
  for (unsigned i = 0; i < order.size(); ++i)
    order[i] = i;
  bool first = true;
  int report = 0;
  unsigned lines = 0;*/

  int epoch = 0;
  while (1)
  {
    Timer iteration("completed in");
    double loss = 0;
    //epoch++;
    cerr << "\n***Training [epoch=" << epoch << "]";
    // train labelled data and unlabelled data
    trainLabelled(data_labelled, lm, sgd);
    trainUnlabelled(data_unlabelled, lm, sgd);

    //test
    testData(test, dev, lm, best);

    cerr << "\n***Best = " << best;
    sgd->status();
    sgd->update_epoch();
    epoch++;
    if (epoch > MAX_EPOCH)
      break;
  }
  delete sgd;
}
