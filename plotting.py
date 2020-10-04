import numpy as np
from numpy import cumsum
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_palette("Paired")
#sns.palplot(sns.color_palette("Paired"))
sns.set_style("whitegrid")


"""
For each experiment (3.1, 3.2, 3.3, 3.4), plot learning curves 
(train and validation) of PPL over both epochs and wall-clock-time.   
Figures  should  have  labeled  axes  and  a  legend  and  an  explanatory caption.

Experiment 3.X has a plot y=train & val PPL         x=epochs Then x=clock-time
"""

epochs = [i+1 for i in range(20)]

# """
# 3.1 RNN
# "Val   SGD  lr=1.0    batch_size=128" 3.1.1
# "Train SGD  lr=1.0    batch_size=20 "
# "SGD  lr=10.0   batch_size=128"
# "ADAM lr=0.001  batch_size=128"
# "ADAM lr=0.0001 batch_size=128"
# """
#
# #x = np.load(lc_path, allow_pickle=True)[()] # 'train_ppls', 'val_ppls', 'train_losses', 'val_losses', 'times'
#
# exp311 = np.load("RNN_SGD_model=RNN_optimizer=SGD_initial_lr=1.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_save_best_0/learning_curves.npy", allow_pickle=True)[()]
# exp312 = np.load("RNN_SGD_model=RNN_optimizer=SGD_initial_lr=1.0_batch_size=20_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]
# exp313 = np.load("RNN_SGD_model=RNN_optimizer=SGD_initial_lr=10.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]
# exp314 = np.load("RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]
# exp315 = np.load("RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]

# """
# x = epochs
# """
# ax = sns.lineplot(x=epochs, y=exp311['val_ppls'], label="Val    SGD    lr=1.0       batch_size=128")
# ax = sns.lineplot(x=epochs, y=exp311['train_ppls'], label="Train SGD    lr=1.0       batch_size=128")
#
# ax = sns.lineplot(x=epochs, y=exp312['val_ppls'], label="Val    SGD    lr=1.0       batch_size=20")
# ax = sns.lineplot(x=epochs, y=exp312['train_ppls'], label="Train SGD    lr=1.0       batch_size=20")
#
# ax = sns.lineplot(x=epochs, y=exp313['val_ppls'], label="Val    SGD    lr=10.0     batch_size=128")
# ax = sns.lineplot(x=epochs, y=exp313['train_ppls'], label="Train SGD    lr=10.0     batch_size=128")
#
# ax = sns.lineplot(x=epochs, y=exp314['val_ppls'], label="Val    ADAM lr=0.001   batch_size=128")
# ax = sns.lineplot(x=epochs, y=exp314['train_ppls'], label="Train ADAM lr=0.001   batch_size=128")
#
# ax = sns.lineplot(x=epochs, y=exp315['val_ppls'], label="Val    ADAM lr=0.0001 batch_size=128")
# ax = sns.lineplot(x=epochs, y=exp315['train_ppls'], label="Train ADAM lr=0.0001 batch_size=128")
#
# print(min(exp313['train_ppls']))
# print(min(exp313['val_ppls']))
#
# plt.xticks(range(1,21,2))
# plt.ylim(bottom=0, top=1200)
# ax.set(xlabel='Epochs', ylabel='PPL')
# ax.grid(False)
# plt.legend()
# plt.show()
#
#
# """
# x = times
# """
#
# ax = sns.lineplot(x=cumsum(exp311['times']), y=exp311['val_ppls'], label="Val    SGD    lr=1.0       batch_size=128")
# ax = sns.lineplot(x=cumsum(exp311['times']), y=exp311['train_ppls'], label="Train SGD    lr=1.0       batch_size=128")
#
# ax = sns.lineplot(x=cumsum(exp312['times']), y=exp312['val_ppls'], label="Val    SGD    lr=1.0       batch_size=20")
# ax = sns.lineplot(x=cumsum(exp312['times']), y=exp312['train_ppls'], label="Train SGD    lr=1.0       batch_size=20")
#
# ax = sns.lineplot(x=cumsum(exp313['times']), y=exp313['val_ppls'], label="Val    SGD    lr=10.0     batch_size=128")
# ax = sns.lineplot(x=cumsum(exp313['times']), y=exp313['train_ppls'], label="Train SGD    lr=10.0     batch_size=128")
#
# ax = sns.lineplot(x=cumsum(exp314['times']), y=exp314['val_ppls'], label="Val    ADAM lr=0.001   batch_size=128")
# ax = sns.lineplot(x=cumsum(exp314['times']), y=exp314['train_ppls'], label="Train ADAM lr=0.001   batch_size=128")
#
# ax = sns.lineplot(x=cumsum(exp315['times']), y=exp315['val_ppls'], label="Val    ADAM lr=0.0001 batch_size=128")
# ax = sns.lineplot(x=cumsum(exp315['times']), y=exp315['train_ppls'], label="Train ADAM lr=0.0001 batch_size=128")
#
# print(min(exp314['times']))
# print(max(exp314['times']))
# print(len(exp314['times']))
# print(exp314['times'])
# print(cumsum(exp311['times']))
#
# plt.ylim(bottom=0, top=1200)
# ax.set(xlabel='Seconds', ylabel='PPL')
# ax.grid(False)
# plt.legend()
# plt.show()



### END 3.1

### START 3.2



# """
# 3.2 GRU
# "SGD  lr=10.0  batch_size=128"
# "ADAM lr=0.001 batch_size=20"
# "ADAM lr=0.001 batch_size=128"
# """
#
# #x = np.load(lc_path, allow_pickle=True)[()] # 'train_ppls', 'val_ppls', 'train_losses', 'val_losses', 'times'
#
# exp311 = np.load("GRU_SGD_model=GRU_optimizer=SGD_initial_lr=10.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.5_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]
# exp312 = np.load("GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=20_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.5_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]
# exp313 = np.load("GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.5_num_epochs=20_save_best_0/learning_curves.npy", allow_pickle=True)[()]

# """
# x = epochs
# """
# ax = sns.lineplot(x=epochs, y=exp311['val_ppls'], label="Val     SGD       lr=10.0       batch_size=128")
# ax = sns.lineplot(x=epochs, y=exp311['train_ppls'], label="Train  SGD       lr=10.0       batch_size=128")
#
# ax = sns.lineplot(x=epochs, y=exp312['val_ppls'], label="Val     ADAM    lr=0.001     batch_size=20")
# ax = sns.lineplot(x=epochs, y=exp312['train_ppls'], label="Train  ADAM    lr=0.001     batch_size=20")
#
# ax = sns.lineplot(x=epochs, y=exp313['val_ppls'], label="Val     ADAM    lr=0.001     batch_size=128")
# ax = sns.lineplot(x=epochs, y=exp313['train_ppls'], label="Train  ADAM    lr=0.001     batch_size=128")
#
#
# print(min(exp313['train_ppls']))
# print(min(exp313['val_ppls']))
#
# plt.xticks(range(1,21,2))
# # plt.ylim(bottom=0, top=1200)
# ax.set(xlabel='Epochs', ylabel='PPL')
# ax.grid(False)
# plt.legend()
# plt.show()
#
#
# """
# x = times
# """
#
# ax = sns.lineplot(x=cumsum(exp311['times']), y=exp311['val_ppls'], label="Val     SGD      lr=10.0        batch_size=128")
# ax = sns.lineplot(x=cumsum(exp311['times']), y=exp311['train_ppls'], label="Train  SGD      lr=10.0        batch_size=128")
#
# ax = sns.lineplot(x=cumsum(exp312['times']), y=exp312['val_ppls'], label="Val     ADAM    lr=0.001     batch_size=20")
# ax = sns.lineplot(x=cumsum(exp312['times']), y=exp312['train_ppls'], label="Train  ADAM    lr=0.001     batch_size=20")
#
# ax = sns.lineplot(x=cumsum(exp313['times']), y=exp313['val_ppls'], label="Val     ADAM    lr=0.001     batch_size=128")
# ax = sns.lineplot(x=cumsum(exp313['times']), y=exp313['train_ppls'], label="Train  ADAM    lr=0.001     batch_size=128")
#
#
# # plt.ylim(bottom=0, top=1200)
# ax.set(xlabel='Seconds', ylabel='PPL')
# ax.grid(False)
# plt.legend()
# plt.show()


### END 3.2

### START 3.3



"""
3.3 GRU with ADAM, lr=0.001, and batch_size=128 
"hidden_size=256  num_layers=2  dp_keep_prob=0.2" 
"hidden_size=2048 num_layers=2  dp_keep_prob=0.5"
"hidden_size=512  num_layers=4  dp_keep_prob=0.5"
"""

# exp311 = np.load("GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=256_num_layers=2_dp_keep_prob=0.2_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]
# exp312 = np.load("GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=2048_num_layers=2_dp_keep_prob=0.5_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]
# exp313 = np.load("GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=4_dp_keep_prob=0.5_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]

# """
# x = epochs
# """
# ax = sns.lineplot(x=epochs, y=exp311['val_ppls'], label="Val     hidden_size=256   num_layers=2  dp_keep_prob=0.2")
# ax = sns.lineplot(x=epochs, y=exp311['train_ppls'], label="Train  hidden_size=256   num_layers=2  dp_keep_prob=0.2")
#
# ax = sns.lineplot(x=epochs, y=exp312['val_ppls'], label="Val     hidden_size=2048 num_layers=2  dp_keep_prob=0.5")
# ax = sns.lineplot(x=epochs, y=exp312['train_ppls'], label="Train  hidden_size=2048 num_layers=2  dp_keep_prob=0.5")
#
# ax = sns.lineplot(x=epochs, y=exp313['val_ppls'], label="Val     hidden_size=512   num_layers=4  dp_keep_prob=0.5")
# ax = sns.lineplot(x=epochs, y=exp313['train_ppls'], label="Train  hidden_size=512   num_layers=4  dp_keep_prob=0.5")
#
#
# print(min(exp313['train_ppls']))
# print(min(exp313['val_ppls']))
#
# plt.xticks(range(1,21,2))
# # plt.ylim(bottom=0, top=1200)
# ax.set(xlabel='Epochs', ylabel='PPL')
# ax.grid(False)
# plt.legend()
# plt.show()
#
#
# """
# x = times
# """
#
# ax = sns.lineplot(x=cumsum(exp311['times']), y=exp311['val_ppls'], label="Val     hidden_size=256   num_layers=2  dp_keep_prob=0.2")
# ax = sns.lineplot(x=cumsum(exp311['times']), y=exp311['train_ppls'], label="Train  hidden_size=256   num_layers=2  dp_keep_prob=0.2")
#
# ax = sns.lineplot(x=cumsum(exp312['times']), y=exp312['val_ppls'], label="Val     hidden_size=2048 num_layers=2  dp_keep_prob=0.5")
# ax = sns.lineplot(x=cumsum(exp312['times']), y=exp312['train_ppls'], label="Train  hidden_size=2048 num_layers=2  dp_keep_prob=0.5")
#
# ax = sns.lineplot(x=cumsum(exp313['times']), y=exp313['val_ppls'], label="Val     hidden_size=512   num_layers=4  dp_keep_prob=0.5")
# ax = sns.lineplot(x=cumsum(exp313['times']), y=exp313['train_ppls'], label="Train  hidden_size=512   num_layers=4  dp_keep_prob=0.5")
#
#
# # plt.ylim(bottom=0, top=1200)
# ax.set(xlabel='Seconds', ylabel='PPL')
# ax.grid(False)
# plt.legend()
# plt.show()


### END 3.3

### START 3.4



"""
3.4 Transfomer --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 

"hidden_size=512  num_layers=6 dp_keep_prob=0.9"
"hidden_size=512  num_layers=2 dp_keep_prob=0.9"
"hidden_size=2048 num_layers=2 dp_keep_prob=0.6"
"hidden_size=1024 num_layers=6 dp_keep_prob=0.9"

"""

exp311 = np.load("TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.0001_batch_size=128_seq_len=35_hidden_size=512_num_layers=6_dp_keep_prob=0.9_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]
exp312 = np.load("TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.0001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.9_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]
exp313 = np.load("TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.0001_batch_size=128_seq_len=35_hidden_size=2048_num_layers=2_dp_keep_prob=0.6_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]
exp314 = np.load("TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.0001_batch_size=128_seq_len=35_hidden_size=1024_num_layers=6_dp_keep_prob=0.9_num_epochs=20_0/learning_curves.npy", allow_pickle=True)[()]

"""
x = epochs
"""
ax = sns.lineplot(x=epochs, y=exp311['val_ppls'], label="Val     hidden_size=512  num_layers=6 dp_keep_prob=0.9")
ax = sns.lineplot(x=epochs, y=exp311['train_ppls'], label="Train  hidden_size=512  num_layers=6 dp_keep_prob=0.9")

ax = sns.lineplot(x=epochs, y=exp312['val_ppls'], label="Val     hidden_size=512  num_layers=2 dp_keep_prob=0.9")
ax = sns.lineplot(x=epochs, y=exp312['train_ppls'], label="Train  hidden_size=512  num_layers=2 dp_keep_prob=0.9")

ax = sns.lineplot(x=epochs, y=exp313['val_ppls'], label="Val     hidden_size=2048 num_layers=2 dp_keep_prob=0.6")
ax = sns.lineplot(x=epochs, y=exp313['train_ppls'], label="Train  hidden_size=2048 num_layers=2 dp_keep_prob=0.6")

ax = sns.lineplot(x=epochs, y=exp314['val_ppls'], label="Val     hidden_size=1024 num_layers=6 dp_keep_prob=0.9")
ax = sns.lineplot(x=epochs, y=exp314['train_ppls'], label="Train  hidden_size=1024 num_layers=6 dp_keep_prob=0.9")


print(min(exp313['train_ppls']))
print(min(exp313['val_ppls']))

plt.xticks(range(1,21,2))
# plt.ylim(bottom=0, top=1200)
ax.set(xlabel='Epochs', ylabel='PPL')
ax.grid(False)
plt.legend()
plt.show()


"""
x = times 
"""

ax = sns.lineplot(x=cumsum(exp311['times']), y=exp311['val_ppls'], label="Val     hidden_size=512  num_layers=6 dp_keep_prob=0.9")
ax = sns.lineplot(x=cumsum(exp311['times']), y=exp311['train_ppls'], label="Train  hidden_size=512  num_layers=6 dp_keep_prob=0.9")

ax = sns.lineplot(x=cumsum(exp312['times']), y=exp312['val_ppls'], label="Val     hidden_size=512  num_layers=2 dp_keep_prob=0.9")
ax = sns.lineplot(x=cumsum(exp312['times']), y=exp312['train_ppls'], label="Train  hidden_size=512  num_layers=2 dp_keep_prob=0.9")

ax = sns.lineplot(x=cumsum(exp313['times']), y=exp313['val_ppls'], label="Val     hidden_size=2048 num_layers=2 dp_keep_prob=0.6")
ax = sns.lineplot(x=cumsum(exp313['times']), y=exp313['train_ppls'], label="Train  hidden_size=2048 num_layers=2 dp_keep_prob=0.6")

ax = sns.lineplot(x=cumsum(exp314['times']), y=exp314['val_ppls'], label="Val     hidden_size=1024 num_layers=6 dp_keep_prob=0.9")
ax = sns.lineplot(x=cumsum(exp314['times']), y=exp314['train_ppls'], label="Train  hidden_size=1024 num_layers=6 dp_keep_prob=0.9")


# plt.ylim(bottom=0, top=1200)
ax.set(xlabel='Seconds', ylabel='PPL')
ax.grid(False)
plt.legend()
plt.show()