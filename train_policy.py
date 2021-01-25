import os
import argparse
import random
import numpy as np
import tensorflow as tf
from tensorflow.python import keras as K

from network import SLPolicyNetwork, RolloutPolicy
from load import read_kifu

# 訓練用データをnumpy形式に変換する
def dataset(train_data):
    states = []
    actions = []
    for data in train_data:
        state, action, win = data
        state = state.reshape(22, 4, 3)
        states.append(state)
        actions.append(action)
    return np.array(states), np.array(actions)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=30, help='Number of passes of the entire dataset training')
    parser.add_argument('--model', '-m', default=None, help='initial model for training')
    parser.add_argument('--policy', '-p', type=str, default='sl', help='Policy to train: ssl or rollout')
    parser.add_argument('--kifu', '-k', type=str, default='./test_play_results', help='filepath of kifu to load')
    args = parser.parse_args()

    if args.policy == 'rollout':
        model = RolloutPolicy()
    else:
        model = SLPolicyNetwork()

    if args.model != None:
        model.model = K.models.load_model(args.model)

    # 棋譜を読み込む
    train_data = read_kifu(args.kifu)
    train_size = len(train_data)

    # 学習ループ
    for epoch in range(args.epoch):
        train_data = random.sample(train_data, train_size)
        x, y = dataset(train_data)
        history = model(x, y)
        
        print("epoch : {}   history: {}\n".format(epoch, history.history))

        # save model
        if args.policy == 'rollout':
            model.model.save('./model/rollout_model.hdf5')
        else:
            model.model.save('./model/sl_model.hdf5')
        

if __name__ == "__main__":
    main()