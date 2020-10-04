#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000M

module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
echo $SLURM_TMPDIR
source $SLURM_TMPDIR/env/bin/activate


# pip install Pillow-SIMD-7.0.0.post3.whl
# pip install --no-index torch torchvision
pip install --no-index torch
pip install --no-index numpy

echo ""
echo "Calling python train script."

# echo "Starting 3.1.1"
# stdbuf -oL python -u ./run_exp.py --model=RNN --optimizer=SGD --initial_lr=1.0 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20 --save_best
# echo "Starting 3.1.2"
# stdbuf -oL python -u ./run_exp.py --model=RNN --optimizer=SGD --initial_lr=1.0 --batch_size=20  --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20
# echo "Starting 3.1.3"
# stdbuf -oL python -u ./run_exp.py --model=RNN --optimizer=SGD --initial_lr=10.0 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20
# echo "Starting 3.1.4"
# stdbuf -oL python -u ./run_exp.py --model=RNN --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20
# echo "Starting 3.1.5"
# stdbuf -oL python -u ./run_exp.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20

# echo "Starting 3.2.1"
# stdbuf -oL python -u ./run_exp.py --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.5  --num_epochs=20 --save_best
# echo "Starting 3.2.2"
# stdbuf -oL python -u ./run_exp.py --model=GRU --optimizer=SGD  --initial_lr=10.0  --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.5  --num_epochs=20
# echo "Starting 3.2.3"
# stdbuf -oL python -u ./run_exp.py --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=20 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.5  --num_epochs=20

# echo "Starting 3.3.1"
# stdbuf -oL python -u ./run_exp.py --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=256 --num_layers=2 --dp_keep_prob=0.2  --num_epochs=20
# echo "Starting 3.3.2"
# stdbuf -oL python -u ./run_exp.py --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=2048 --num_layers=2 --dp_keep_prob=0.5  --num_epochs=20
# echo "Starting 3.3.3"
# stdbuf -oL python -u ./run_exp.py --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=4 --dp_keep_prob=0.5  --num_epochs=20

#echo "Starting 3.4.1"
#stdbuf -oL python -u ./run_exp.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=512  --num_layers=6 --dp_keep_prob=0.9 --num_epochs=20
#echo "Starting 3.4.2"
#stdbuf -oL python -u ./run_exp.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=512  --num_layers=2 --dp_keep_prob=0.9 --num_epochs=20
#echo "Starting 3.4.3"
#stdbuf -oL python -u ./run_exp.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=2048 --num_layers=2 --dp_keep_prob=0.6 --num_epochs=20
#echo "Starting 3.4.4"
#stdbuf -oL python -u ./run_exp.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=1024 --num_layers=6 --dp_keep_prob=0.9 --num_epochs=20

#echo "Starting generate RNN"
#stdbuf -oL python -u ./exp42.py --model=RNN --optimizer=SGD --initial_lr=1.0 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20 --save_best
#
#echo "Starting generate GRU"
#stdbuf -oL python -u ./exp42.py --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.5  --num_epochs=20 --save_best

echo "RNN gradients"
stdbuf -oL python -u ./exp4.py --model=RNN --optimizer=SGD --initial_lr=1.0 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20 --save_best

echo "GRU gradients"
stdbuf -oL python -u ./exp4.py --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.5  --num_epochs=20 --save_best
