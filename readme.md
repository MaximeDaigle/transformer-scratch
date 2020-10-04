**Date: March 17th, 2020**

# GRU and Transformer

4 parts project. Part 1 consists of implementing a stacked **gated
recurrent units (GRUs)** from a simple RNN implementation, . Part 2 is
the implementation of the **attention module of a transformer network**.
Part 3 consist of training these 3 models (RNN, GRU, and Transformer)
using a variety of different optimizers and hyperparameter settings and
Part 4 consist of generating samples from the trained models.

#### Summary

This repo implements and trains **sequential language models** on the
Penn Treebank dataset. Language models learn to assign a likelihood to
sequences of text. The elements of the sequence (typically words or
individual characters) are called tokens, and can be represented as
one-hot vectors with length equal to the vocabulary size, e.g. 26 for a
vocabulary of English letters with no punctuation or spaces, in the case
of characters, or as indices in the vocabulary for words. In this
representation an entire dataset (or a mini-batch of examples) can be
represented by a 3-dimensional tensor, with axes corresponding to: (1)
the example within the dataset/mini-batch, (2) the time-step within the
sequence, and (3) the index of the token in the vocabulary. Sequential
language models do **next-step prediction**, in other words, they
predict tokens in a sequence one at a time, with each prediction based
on all the previous elements of the sequence. A trained sequential
language model can also be used to generate new sequences of text, by
making each prediction conditioned on the past *predictions* (instead of
the ground-truth input sequence).

#### The Penn Treebank Dataset

This is a dataset of about 1 million words from about 2,500 stories from
the Wall Street Journal. It has Part-of-Speech annotations and is
sometimes used for training parsers, but it's also a very common
benchmark dataset for training RNNs and other sequence models to do
next-step prediction.

#### Preprocessing

The version of the dataset you will work with has been preprocessed:
lower-cased, stripped of non-alphabetic characters, tokenized (broken up
into words, with sentences separated by the `<eos>` (end of sequence)
token), and cut down to a vocabulary of 10,000 words; any word not in
this vocabulary is replaced by `<unk>`. For the transformer network,
positional information (an embedding of the position in the source
sequence) for each token is also included in the input sequence.

Part
=====

#### Implementing an RNN with Gated Recurrent Units (GRU)

\vspace{-\baselineskip}
The model is implemented **from scratch** using PyTorch Tensors,
Variables, and associated operations. It doesn't use built-in Recurrent
modules.

The use of gating can significantly improve the performance of RNNs. The
Long-Short Term Memory (LSTM) RNN is the best known example of gating in
RNNs; GRU-RNNs are a slightly simpler variant (with fewer gates).

The equations for a GRU are: $$\begin{aligned}
{\bm{r}}_t &= \sigma_r ({\bm{W}}_r {\bm{x}}_t + {\bm{U}}_r {\bm{h}}_{t-1} + {\bm{b}}_r) \\
{\bm{z}}_t &= \sigma_z ({\bm{W}}_z {\bm{x}}_t + {\bm{U}}_z {\bm{h}}_{t-1} + {\bm{b}}_z) \\
\tilde{{\bm{h}}}_t &= \sigma_h ({\bm{W}}_h {\bm{x}}_t + {\bm{U}}_h ({\bm{r}}_t \odot {\bm{h}}_{t-1}) + {\bm{b}}_h) \\
{\bm{h}}_t &= (1 - {\bm{z}}_t) \odot {\bm{h}}_{t-1} + {\bm{z}}_t \odot \tilde{{\bm{h}}}_t \\
P({\bm{y}}_t | {\bm{x}}_1, ...., {\bm{x}}_t) &= \sigma_y({\bm{W}}_y {\bm{h}}_t + {\bm{b}}_y)\end{aligned}$$
${\bm{r}}_t$ is called the "reset gate" and ${\bm{z}}_t$ the "forget
gate". The trainable parameters are
${\bm{W}}_r, {\bm{W}}_z, {\bm{W}}_h, {\bm{W}}_y, \\ {\bm{U}}_r, {\bm{U}}_z, {\bm{U}}_h, {\bm{b}}_r, {\bm{b}}_z, {\bm{b}}_h,$
and ${\bm{b}}_y$, as well as the initial hidden state parameter
${\bm{h}}_0$. GRUs use the $\mathrm{sigmoid}$ activation function for
$\sigma_r$ and $\sigma_z$, and $\mathrm{tanh}$ for $\sigma_h$.

Specifically about the implementation, it uses matrix and tensor
operations (e.g. dot, multiply, add, etc.) to implement the recurrent
unit calculations. It uses subclass `nn.module`, built-in Linear
modules, and built-in implementations of nonlinearities (tanh, sigmoid,
and softmax), initializations, loss functions, and optimization
algorithms.

\nobreak
\extramarks{Problem \arabic{homeworkProblemCounter} (continued)}{Problem \arabic{homeworkProblemCounter} continued on next page\ldots}
\nobreak{}
\stepcounter{homeworkProblemCounter}
\nobreak
\extramarks{Problem \arabic{homeworkProblemCounter}}{}
\nobreak{}
-1\>0

Part
=====

\setcounter{partCounter}{1}
\nobreak
\extramarks{}{Problem \arabic{homeworkProblemCounter} continued on next page\ldots}
\nobreak{}
\nobreak
\extramarks{Problem \arabic{homeworkProblemCounter} (continued)}{Problem \arabic{homeworkProblemCounter} continued on next page\ldots}
\nobreak{}
#### Implementing the multi-head attention module of a transformer

While prototypical RNNs "remember" past information by taking their
previous hidden state as input at each step, recent years have seen a
profusion of methodologies for making use of past information in
different ways. The transformer [^1] is one such architecture which uses
several self-attention networks ("heads") in parallel, among other
architectural specifics. The transformer is quite complicated to
implement compared to the RNNs described so far. The project uses
open-source code to implements the different modules that are not the
multi-head attention module (e.g. word embedding map, positional
encoding, and mlp layers).

The attention vector for $m$ heads indexed by $i$ is calculated as
follows: $$\begin{aligned}
{\bm{A}}_i &=\text{softmax}\left(\frac{{\bm{Q}}_i{\bm{W}}_{Q_i}({\bm{K}}_i{\bm{W}}_{K_i})^\top}{\sqrt{d_k}} \right) \\
{\bm{H}}_i &= {\bm{A}}_i {\bm{V}}{\bm{W}}_{V_i}\\
A({\bm{Q}}, {\bm{K}}, {\bm{V}}) &= \text{concat}({\bm{H}}_1, ..., {\bm{H}}_m){\bm{W}}_O\end{aligned}$$
where ${\bm{Q}}, {\bm{K}}, {\bm{V}}$ are queries, keys, and values
respectively, ${\bm{W}}_{Q_i}, {\bm{W}}_{K_i}, {\bm{W}}_{V_i}$ are their
corresponding embedding matrices, ${\bm{W}}_O$ is the output embedding,
and $d_k$ is the dimension of the keys. ${\bm{Q}}, {\bm{K}},$ and
${\bm{V}}$ are determined by the output of the feed-forward layer of the
main network. ${\bm{A}}_i$ are the attention values, which specify which
elements of the input sequence each attention head attends to.

Note that the implementation of multi-head attention requires binary
masks, so that attention is computed only over the past, not the future.
A mask value of $1$ indicates an element which the model is allowed to
attend to (i.e. from the past); a value of $0$ indicates an element it
is not allowed to attend to. This can be implemented by modifying the
$\text{softmax}$ function to account for the mask ${\bm{s}}$ as follows:
$$\begin{aligned}
\tilde{{\bm{x}}} &= \exp({\bm{x}}) \odot {\bm{s}}\\
\text{softmax}({\bm{x}}, {\bm{s}}) &\doteq \frac{\tilde{{\bm{x}}}}{\sum_i \tilde{x}_i}\end{aligned}$$
To avoid potential numerical stability issues, a different
implementation is: $$\begin{aligned}
\tilde{{\bm{x}}} &= {\bm{x}}\odot {\bm{s}}- 10^{9} (1 - {\bm{s}}) \\
\text{softmax}({\bm{x}}, {\bm{s}}) &\doteq \frac{\exp(\tilde{{\bm{x}}})}{\sum_i \exp(\tilde{x}_i)}\end{aligned}$$
This second version is equivalent (up to numerical precision) as long as
${\bm{x}}>> -10^9$, which should be the case in practice.

\nobreak
\extramarks{Problem \arabic{homeworkProblemCounter} (continued)}{Problem \arabic{homeworkProblemCounter} continued on next page\ldots}
\nobreak{}
\stepcounter{homeworkProblemCounter}
\nobreak
\extramarks{Problem \arabic{homeworkProblemCounter}}{}
\nobreak{}
-1\>0

Part
=====

\setcounter{partCounter}{1}
\nobreak
\extramarks{}{Problem \arabic{homeworkProblemCounter} continued on next page\ldots}
\nobreak{}
\nobreak
\extramarks{Problem \arabic{homeworkProblemCounter} (continued)}{Problem \arabic{homeworkProblemCounter} continued on next page\ldots}
\nobreak{}
#### Training language models and model comparison

\vspace{-\baselineskip}
Unlike in classification problems, where the performance metric is
typically accuracy, in language modelling, the performance metric is
typically based directly on the cross-entropy loss, i.e. the negative
log-likelihood ($NLL$) the model assigns to the tokens. For word-level
language modelling it is standard to report **perplexity (PPL)**, which
is the exponentiated average per-token NLL (over all tokens):
$$\exp\left(\frac{1}{TN}\sum_{t=1}^T \sum_{n=1}^N - \log P({\bm{x}}^{(n)}_t | {\bm{x}}^{(n)}_1, ...., {\bm{x}}^{(n)}_{t-1})\right),$$
where $t$ is the index with the sequence, and $n$ indexes different
sequences. For Penn Treebank in particular, the test set is treated as a
single sequence (i.e. $N=1$). The purpose of this part is to perform
model exploration.

The three architectures are trained using either stochastic gradient
descent or the ADAM optimizer. The training loop is provided in
*run\_exp.py*. For each experiment (3.1, 3.2, 3.3, 3.4), the learning
curves (train and validation) of PPL over both epochs and
wall-clock-time are in the folder images.

![Best Validation PPL for each experiment](img/table_result.png)

[\[table\_result\]]{#table_result label="table_result"}

\nobreak
\extramarks{Problem \arabic{homeworkProblemCounter} (continued)}{Problem \arabic{homeworkProblemCounter} continued on next page\ldots}
\nobreak{}
\stepcounter{homeworkProblemCounter}
\nobreak
\extramarks{Problem \arabic{homeworkProblemCounter}}{}
\nobreak{}
-1\>0

Part
=====

\setcounter{partCounter}{1}
\nobreak
\extramarks{}{Problem \arabic{homeworkProblemCounter} continued on next page\ldots}
\nobreak{}
\nobreak
\extramarks{Problem \arabic{homeworkProblemCounter} (continued)}{Problem \arabic{homeworkProblemCounter} continued on next page\ldots}
\nobreak{}
#### Comparison of samples generated by both the RNN and GRU models

\vspace{-\baselineskip}
The samples are generated by recursively making
$$\hat{{\bm{x}}}_{t+1} = \argmax P({\bm{x}}_{t+1} | \hat{{\bm{x}}}_1, ...., \hat{{\bm{x}}}_t)$$
It's conditioned on the sampled $\hat{{\bm{x}}}_t$, *not* the ground
truth. 20 samples are generated from both the RNN and GRU: 10 sequences
of the same length as the training sequences, and 10 sequences of
*twice* the length of the training sequences. All 40 samples are
available in the folder images. Here are the 3 "best", 3 "worst", and 3
that are "interesting".

Best three (all from the GRU model):

1.  nov. N N $<$eos$>$ the company said it will redeem \$ N million of
    assets and N N of the N N convertible subordinated debentures due
    nov. N N $<$eos$>$ the notes are rated triple-a

2.  N N $<$eos$>$ the key rate of the bills was quoted at N N to yield N
    N $<$eos$>$ the N N N notes due N was N N to N N N to yield

3.  macy & co. said it agreed to acquire its N N stake in the u.s. and
    $<$unk$>$ concern $<$eos$>$ the company said it will sell its N N
    stake in the u.s. and N N

Three interesting:

1.  sachs & co. and salomon brothers inc $<$eos$>$ the move is a
    $<$unk$>$ of the $<$unk$>$ of the $<$unk$>$ $<$unk$>$ of the
    $<$unk$>$ $<$unk$>$ of the $<$unk$>$ $<$unk$>$ $<$eos$>$ the
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ and $<$unk$>$ $<$unk$>$
    $<$eos$>$ the $<$unk$>$ of the $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ and\
    It starts promising, but quickly gets in a loop and deteriorates.

2.  at par to yield N N $<$eos$>$ the N N N notes due N fell N to N N to
    yield N N $<$eos$>$ the N N N notes due N fell N to N N to yield N N
    $<$eos$>$ the N N N notes due N fell N to N N to yield N N $<$eos$>$
    the N N N notes due N fell N to N N\
    The words generated are good, but it gets stuck in a loop.

3.  n't be able to pay the debt $<$eos$>$ the company said it would n't
    identify the offer $<$eos$>$ the company said it would n't elaborate
    $<$eos$>$ the company said it would n't elaborate $<$eos$>$ the
    company said it would n't elaborate $<$eos$>$ the company said it
    would n't elaborate $<$eos$>$ the company said it would n't
    elaborate $<$eos$>$ the company said it will sell its stake in
    navigation mixte to\
    It gets stuck in a loop, but, eventually, gets out of it.

Worst three:

1.  and $<$unk$>$ $<$unk$>$ $<$eos$>$ the $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ and $<$unk$>$ $<$unk$>$ $<$eos$>$ the
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ and $<$unk$>$ $<$unk$>$
    $<$eos$>$ the $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ and $<$unk$>$
    $<$unk$>$ $<$eos$>$ the

2.  of the $<$unk$>$ $<$eos$>$ the $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$eos$>$ the $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$eos$>$ the $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$

3.  $<$eos$>$ but the company 's $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$eos$>$ the $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$eos$>$ the $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$eos$>$ the $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$eos$>$ the
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$ $<$unk$>$ $<$eos$>$ the $<$unk$>$ $<$unk$>$
    $<$unk$>$ $<$unk$>$

\nobreak
\extramarks{Problem \arabic{homeworkProblemCounter} (continued)}{Problem \arabic{homeworkProblemCounter} continued on next page\ldots}
\nobreak{}
\stepcounter{homeworkProblemCounter}
\nobreak
\extramarks{Problem \arabic{homeworkProblemCounter}}{}
\nobreak{}

[^1]: See <https://arxiv.org/abs/1706.03762> for more details.
