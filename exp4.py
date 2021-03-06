#!/bin/python
# coding: utf-8

# based on code from:
#    https://github.com/deeplearningathome/pytorch-language-model/blob/master/reader.py
#    https://github.com/ceshine/examples/blob/master/word_language_model/main.py
#    https://github.com/teganmaharaj/zoneout/blob/master/zoneout_word_ptb.py
#    https://github.com/harvardnlp/annotated-transformer
# By Tegan Maharaj, David Krueger, and Chin-Wei Huang
# Edits by Jessica Thompson, Jonathan Cornford and Lluis Castrejon

import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy
np = numpy

from models import RNN, GRU
from models import make_model as TRANSFORMER


##############################################################################
#
# ARG PARSING AND EXPERIMENT SETUP
#
##############################################################################

parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')

parser.add_argument('--data', type=str, default='data',
                    help='location of the data corpus. We suggest you change the default\
                    here, rather than passing as an argument, to avoid long file paths.')
parser.add_argument('--model', type=str, default='GRU',
                    help='type of recurrent net (RNN, GRU, TRANSFORMER)')
parser.add_argument('--optimizer', type=str, default='SGD_LR_SCHEDULE',
                    help='optimization algo to use; SGD, SGD_LR_SCHEDULE, ADAM')
parser.add_argument('--seq_len', type=int, default=35,
                    help='number of timesteps over which BPTT is performed')
parser.add_argument('--batch_size', type=int, default=20,
                    help='size of one minibatch')
parser.add_argument('--initial_lr', type=float, default=20.0,
                    help='initial learning rate')
parser.add_argument('--hidden_size', type=int, default=200,
                    help='size of hidden layers. IMPORTANT: for the transformer\
                    this must be a multiple of 16.')
parser.add_argument('--save_best', action='store_true',
                    help='save the model for the best validation performance')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of hidden layers in RNN/GRU, or number of transformer blocks in TRANSFORMER')

# Other hyperparameters you may want to tune in your exploration
parser.add_argument('--emb_size', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs to stop after')
parser.add_argument('--dp_keep_prob', type=float, default=0.35,
                    help='dropout *keep* probability. drop_prob = 1-dp_keep_prob \
                    (dp_keep_prob=1 means no dropout)')

parser.add_argument('--debug', action='store_true')
parser.add_argument('--save_dir', type=str, default='',
                    help='path to save the experimental config, logs, model \
                    This is automatically generated based on the command line \
                    arguments you pass and only needs to be set if you want a \
                    custom dir name')
parser.add_argument('--evaluate', action='store_true',
                    help="use this flag to run on the test set.")

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

def set_seed(seed):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

args = parser.parse_args()
argsdict = args.__dict__
argsdict['code_file'] = sys.argv[0]

# Use the model, optimizer, and the flags passed to the script to make the
# name for the experimental dir
print("\n########## Setting Up Experiment ######################")
flags = [flag.lstrip('--').replace('/', '').replace('\\', '') for flag in sys.argv[1:]]
experiment_path = os.path.join(args.save_dir, '_'.join([argsdict['model'],
                                         argsdict['optimizer']]
                                         + flags))
# Increment a counter so that previous results with the same args will not
# be overwritten. Comment out the next four lines if you only want to keep
# the most recent results.
i = 0
while os.path.exists(experiment_path + "_" + str(i)):
    i += 1
experiment_path = experiment_path + "_" + str(i)

# Creates an experimental directory and dumps all the args to a text file
os.makedirs(experiment_path, exist_ok=True)

print ("\nPutting log in %s"%experiment_path)
argsdict['save_dir'] = experiment_path
with open (os.path.join(experiment_path,'exp_config.txt'), 'w') as f:
    for key in sorted(argsdict):
        f.write(key+'    '+str(argsdict[key])+'\n')

# Set the random seed manually for reproducibility.
set_seed(args.seed)


# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")


###############################################################################
#
# LOADING & PROCESSING
#
###############################################################################

# HELPER FUNCTIONS
def _read_words(filename):
    with open(filename, "r") as f:
      return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

# Processes the raw data from text files
def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word

# Yields minibatches of data
def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)


class Batch:
    "Data processing for the transformer. This class adds a mask to the data."
    def __init__(self, x, pad=-1):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."

        def subsequent_mask(size):
            """ helper function for creating the masks. """
            attn_shape = (1, size, size)
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
            return torch.from_numpy(subsequent_mask) == 0

        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


# LOAD DATA
print('Loading data from '+args.data)
if args.data == 'SLURM_TMPDIR':
    raw_data = ptb_raw_data(data_path=os.environ['SLURM_TMPDIR'])
else:
    raw_data = ptb_raw_data(data_path=args.data)
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
vocab_size = len(word_to_id)
print('  vocabulary size: {}'.format(vocab_size))


###############################################################################
#
# MODEL SETUP
#
###############################################################################

if args.model == 'RNN':
    model = RNN(emb_size=args.emb_size, hidden_size=args.hidden_size,
                seq_len=args.seq_len, batch_size=args.batch_size,
                vocab_size=vocab_size, num_layers=args.num_layers,
                dp_keep_prob=args.dp_keep_prob)
elif args.model == 'GRU':
    model = GRU(emb_size=args.emb_size, hidden_size=args.hidden_size,
                seq_len=args.seq_len, batch_size=args.batch_size,
                vocab_size=vocab_size, num_layers=args.num_layers,
                dp_keep_prob=args.dp_keep_prob)
elif args.model == 'TRANSFORMER':
    if args.debug:  # use a very small model
        model = TRANSFORMER(vocab_size=vocab_size, n_units=16, n_blocks=2)
    else:
        # Note that we're using num_layers and hidden_size to mean slightly
        # different things here than in the RNNs.
        # Also, the Transformer also has other hyperparameters
        # (such as the number of attention heads) which can change it's behavior.
        model = TRANSFORMER(vocab_size=vocab_size, n_units=args.hidden_size,
                            n_blocks=args.num_layers, dropout=1.-args.dp_keep_prob)
    # these 3 attributes don't affect the Transformer's computations;
    # they are only used in run_epoch
    model.batch_size=args.batch_size
    model.seq_len=args.seq_len
    model.vocab_size=vocab_size
else:
    print("Model type not recognized.")

model = model.to(device)

# LOAD MODEL
#model.load_state_dict(torch.load(os.path.join(args.save_dir[:-1]+'0', 'best_params.pt')))
## model.eval()
## torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_params.pt'))

# LOSS FUNCTION
loss_fn = nn.CrossEntropyLoss()
if args.optimizer == 'ADAM':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)

# LEARNING RATE
lr = args.initial_lr
# These variables are for learning rate schedule
# see SGD_LR_SCHEDULE in the main loop
lr_decay_base = 1 / 1.15
m_flat_lr = 14.0 # we will not touch lr for the first m_flat_lr epochs

###############################################################################
#
# DEFINE COMPUTATIONS FOR PROCESSING ONE EPOCH
#
###############################################################################

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.

    This prevents Pytorch from trying to backpropagate into previous input
    sequences when we use the final hidden states from one mini-batch as the
    initial hidden states for the next mini-batch.

    Using the final hidden states in this way makes sense when the elements of
    the mini-batches are actually successive subsequences in a set of longer sequences.
    This is the case with the way we've processed the Penn Treebank dataset.
    """
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)


def run_epoch(model, data, is_train=False, lr=1.0):
    """
    One epoch of training/validation (depending on flag is_train).
    """
    model.train()
    epoch_size = ((len(data) // model.batch_size) - 1) // model.seq_len
    start_time = time.time()
    if args.model != 'TRANSFORMER':
        hidden = model.init_hidden()
        hidden = hidden.to(device)
    costs = 0.0
    iters = 0
    losses = []

    # LOOP THROUGH MINIBATCHES
    for step, (x, y) in enumerate(ptb_iterator(data, model.batch_size, model.seq_len)):
        if args.model == 'TRANSFORMER':
            batch = Batch(torch.from_numpy(x).long().to(device))
            model.zero_grad()
            outputs = model.forward(batch.data, batch.mask).transpose(1,0)
        else:
            inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
            model.zero_grad()
            hidden = repackage_hidden(hidden)
            # hidden.requires_grad_(True)
            # hidden.retain_grad()
            # for i in range(len(hidden)):
            #     hidden[i].requires_grad_(True)
            #     hidden[i].retain_grad()
            outputs, hidden, all_hidden = model(inputs, hidden)

        targets = torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
        tt = torch.squeeze(targets.view(-1, model.batch_size * model.seq_len))

        # LOSS COMPUTATION
        # averages across all the sequences in a mini-batch
        # and all time-steps of the sequences.
        # For experiment 4.1, need to compute the average loss
        # at each time-step separately. to do that I use the method retain_grad to keep
        # gradients for intermediate nodes of the computational graph.
        #
        loss = loss_fn(outputs.contiguous().view(-1, model.vocab_size), tt)
        costs += loss.data.item() * model.seq_len
        losses.append(costs)
        iters += model.seq_len

        #hidden[-1].mean().backward()
        loss.backward(retain_graph=True)
        norm = np.array([None, None,None])
        for i in range(2, len(all_hidden),2):
            #hidden[i].mean().backward(retain_graph=True)
            #print('Timestep', i//2)
            # # print(all_hidden[i])
            all_layers = all_hidden[i].grad.numpy()
            all_layers = np.stack((all_hidden[i+1].grad.numpy(),all_layers))
            #print(np.linalg.norm(all_layers))
            x1, x2, x3 = np.linalg.norm(all_layers), np.linalg.norm(all_hidden[i].grad), np.linalg.norm(all_hidden[i+1].grad)
            #print(np.linalg.norm(all_hidden[i].grad), np.linalg.norm(all_hidden[i+1].grad))
            #print([x1,x2,x3])
            norm = np.vstack((norm, np.array([x1,x2,x3])))
        norm = norm[1:,:]
        # print(norm)
        min = np.amin(norm, axis=0)
        max = np.amax(norm, axis=0)
        # print(min,max)
        print((norm - min)/(max-min))
        import sys
        sys.exit()

        if args.debug:
            print(step, loss)
        if is_train:  # Only update parameters if training
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            if args.optimizer == 'ADAM':
                optimizer.step()
            else:
                for p in model.parameters():
                    if p.grad is not None:
                        p.data.add_(-lr, p.grad.data)
            if step % (epoch_size // 10) == 10:
                print('step: '+ str(step) + '\t' \
                    + "loss (sum over all examples' seen this epoch):" + str(costs) + '\t' \
                    + 'speed (wps):' + str(iters * model.batch_size / (time.time() - start_time)))
    return np.exp(costs / iters), losses



###############################################################################
#
# RUN MAIN LOOP (TRAIN AND VAL)
#
###############################################################################

print("\n########## Running Main Loop ##########################")
train_ppls = []
train_losses = []
val_ppls = []
val_losses = []
best_val_so_far = np.inf
times = []

# In debug mode, only run one epoch
if args.debug:
    num_epochs = 1
else:
    num_epochs = args.num_epochs

# MAIN LOOP
for epoch in range(num_epochs):
    t0 = time.time()
    print('\nEPOCH '+str(epoch)+' ------------------')
    if args.optimizer == 'SGD_LR_SCHEDULE':
        lr_decay = lr_decay_base ** max(epoch - m_flat_lr, 0)
        lr = lr * lr_decay # decay lr if it is time

    # RUN MODEL ON TRAINING DATA
    train_ppl, train_loss = run_epoch(model, train_data, True, lr)

    # RUN MODEL ON VALIDATION DATA
    val_ppl, val_loss = run_epoch(model, valid_data)


    # SAVE MODEL IF IT'S THE BEST SO FAR
    # if val_ppl < best_val_so_far:
    #     best_val_so_far = val_ppl
    #     if args.save_best:
    #         print("Saving model parameters to best_params.pt")
    #         torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_params.pt'))

    # LOC RESULTS
    train_ppls.append(train_ppl)
    val_ppls.append(val_ppl)
    train_losses.extend(train_loss)
    val_losses.extend(val_loss)
    times.append(time.time() - t0)
    log_str = 'epoch: ' + str(epoch) + '\t' \
            + 'train ppl: ' + str(train_ppl) + '\t' \
            + 'val ppl: ' + str(val_ppl)  + '\t' \
            + 'best val: ' + str(best_val_so_far) + '\t' \
            + 'time (s) spent in epoch: ' + str(times[-1])
    # print(log_str)
    # with open (os.path.join(args.save_dir, 'log.txt'), 'a') as f_:
    #     f_.write(log_str+ '\n')

# # SAVE LEARNING CURVES
# lc_path = os.path.join(args.save_dir, 'learning_curves.npy')
# print('\nDONE\n\nSaving learning curves to '+lc_path)
# np.save(lc_path, {'train_ppls':train_ppls,
#                   'val_ppls':val_ppls,
#                   'train_losses':train_losses,
#                   'val_losses':val_losses,
#                   'times':times})
# NOTE ==============================================
# To load these, run
# >>> x = np.load(lc_path, allow_pickle=True)[()]


## Plotting related
# import numpy as np
# from numpy import cumsum
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()
# sns.set_style("whitegrid")
#
#
# norms_rnn = [1.0, 0.8371797829070176, 0.7088431242235455, 0.6215466868565448,
#  0.5611284766977119, 0.5027503670833419, 0.4337970931378192,
#  0.38393500629638866, 0.3588040887328191, 0.3458262535746043,
#  0.3243722857197341, 0.3030781041760983, 0.29401237807147584,
#  0.29611287874560965, 0.2731497551718317, 0.27723617142845036,
#  0.28125083074897717, 0.2497645087086506, 0.2618885493304216,
#  0.24800103962805045, 0.24475434902827972, 0.221535104044026,
#  0.22032489789115123, 0.20931506733216046, 0.21763740287917643,
#  0.22858988439081823, 0.22225465095830116, 0.21672778483654462,
#  0.2179763555245425, 0.20351484683496116, 0.1708024417133805,
#  0.15237712507530593, 0.11677178539873478, 0.0]
#
# norms_gru = [1.0, 0.8095128757267991, 0.773129425096312, 0.7487689304854905,
#  0.7486208472825466, 0.7339944270098162, 0.7170263535868255,
#  0.7059299137816514, 0.7047547254830892, 0.6873855888980892, 0.68069890533127,
#  0.6646528248540263, 0.6411453971890362, 0.6204127795050263,
#  0.6060907106006734, 0.6047428842116974, 0.5773747389764305,
#  0.5702205432758083, 0.5560419266087799, 0.5547510718666092,
#  0.5316548308698661, 0.5173488087998686, 0.4969250119045433,
#  0.4947756515243604, 0.48987269744754636, 0.4837822237799977,
#  0.46473560067205005, 0.4197315536927347, 0.37240997766312967,
#  0.32445702870460963, 0.2864252774042641, 0.22497333290757168,
#  0.13426294762792518, 0.0]
#
#
#
# ax = sns.lineplot(x=[i for i in range(34)], y=norms_rnn, label="RNN")
# ax = sns.lineplot(x=[i for i in range(34)], y=norms_gru, label="GRU")
#
#
# # plt.ylim(bottom=0, top=1200)
# ax.set(xlabel='Time-step', ylabel='Norm of gradients')
# ax.grid(False)
# plt.legend()
# plt.show()
