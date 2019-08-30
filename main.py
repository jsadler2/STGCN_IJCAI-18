# @Time     : Jan. 02, 2019 22:17
# @Author   : Veritas YIN
# @FileName : main.py
# @Version  : 1.0
# @Project  : Orion
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=228)
# [jeff] how many past observations are used to predict the future observations
parser.add_argument('--n_his', type=int, default=12)
# [jeff] how many future observations predicted
parser.add_argument('--n_pred', type=int, default=9)
# [jeff] don't know what "batch_size" means
parser.add_argument('--batch_size', type=int, default=50)
# [jeff] don't know what "epoch" means
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--save', type=int, default=10)
# [jeff] I think ks is the number of spatial convolutions (this is true see line 18 of "base_model")
parser.add_argument('--ks', type=int, default=3)
# [jeff] I think kt is the number of temporal convolutions
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
# [jeff] the optimizer
parser.add_argument('--opt', type=str, default='RMSProp')
# [jeff] you can specify a weighted matrix file
parser.add_argument('--graph', type=str, default='default')
# [jeff] inference mode
parser.add_argument('--inf_mode', type=str, default='merge')

args = parser.parse_args()
print(f'Training configs: {args}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[1, 32, 64], [64, 32, 128]]

n = 228
# Load wighted adjacency matrix W
W = weight_matrix(pjoin('./data_loader/PeMS-M/PeMS-M', f'W_{n}.csv'))
# if args.graph == 'default':
#     W = weight_matrix(pjoin('./data_loader/PeMS-M/PeMS-M', f'W_228.csv'))
# else:
#     # load customized graph weight matrix
#     W = weight_matrix(pjoin('./dataset', args.graph))

# Calculate graph kernel
L = scaled_laplacian(W)
# Alternative approximation method: 1st approx - first_approx(W, n).
Lk = cheb_poly_approx(L, Ks, n)
# [jeff] Lk shape is (228, 684), so now there are 3x the number of columns compared to L, I think this has to do with
# Ks being 3. I think from the manuscript I am gathering that K is the number of convolutions
# [jeff] here they are storing Lk as a kernel in the collection. They later use this in the layers.py, line 22
tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

# Data Preprocessing
data_file = f'V_{n}.csv'
n_train, n_val, n_test = 34, 5, 5
PeMS = data_gen(pjoin('./data_loader/PeMS-M/PeMS-M', data_file), (n_train, n_val, n_test), n, n_his + n_pred)
# [jeff] PeMs train.shape = (9112, 21, 228, 1), test.shape = (1340, 21, 228, 1), val.shape = (1340, 21, 228, 1),
# I really only understand one of these numbers: 228 - that's the number of sensors. I suppose the first dimension
# must be the time. but I don't know where/how they got 9112 or 1340. Then I have *no* idea what the 21 is... is that
# the number of features...?
print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')

if __name__ == '__main__':
    model_train(PeMS, blocks, args)
    model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode)
