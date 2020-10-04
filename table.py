import numpy as np
from numpy import cumsum

exp311 = np.load("RNN_SGD_model=RNN_optimizer=SGD_initial_lr=1.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_save_best_0/learning_curves.npy", allow_pickle=True)[()]
exp312 = np.load("RNN_SGD_model=RNN_optimizer=SGD_initial_lr=1.0_batch_size=20_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]
exp313 = np.load("RNN_SGD_model=RNN_optimizer=SGD_initial_lr=10.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]
exp314 = np.load("RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]
exp315 = np.load("RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]

exp321 = np.load("GRU_SGD_model=GRU_optimizer=SGD_initial_lr=10.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.5_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]
exp322 = np.load("GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=20_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.5_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]
exp323 = np.load("GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.5_num_epochs=20_save_best_0/learning_curves.npy", allow_pickle=True)[()]

exp331 = np.load("GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=256_num_layers=2_dp_keep_prob=0.2_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]
exp332 = np.load("GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=2048_num_layers=2_dp_keep_prob=0.5_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]
exp333 = np.load("GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=4_dp_keep_prob=0.5_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]

exp341 = np.load("TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.0001_batch_size=128_seq_len=35_hidden_size=512_num_layers=6_dp_keep_prob=0.9_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]
exp342 = np.load("TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.0001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.9_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]
exp343 = np.load("TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.0001_batch_size=128_seq_len=35_hidden_size=2048_num_layers=2_dp_keep_prob=0.6_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]
exp344 = np.load("TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.0001_batch_size=128_seq_len=35_hidden_size=1024_num_layers=6_dp_keep_prob=0.9_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]


t = [exp311, exp312 , exp313, exp314, exp315, exp321 , exp322, exp323, exp331, exp332, exp333, exp341, exp342, exp343 , exp344]

#print(cumsum(exp332['times']))

for e in t[-4:]:
    ## epoch
    # epoch = np.argmin(np.array(e['val_ppls'])) + 1
    # print(epoch)
    ## val ppl
    #print(round(np.min(np.array(e['val_ppls'])),2))

    index = np.argmin(np.array(e['val_ppls']))
    ## train ppl
    #print(round(e['train_ppls'][index],2))

    ## time (minutes)
    print(round(cumsum(e['times'][:index+1])[-1]/60))


