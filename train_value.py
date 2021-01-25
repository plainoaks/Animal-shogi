import os
import argparse
import random
import numpy as np
import tensorflow as tf
from tensorflow.python import keras as K

from network import ValueNetwork
from load import read_kifu

# 訓練データをnumpy形式に変換
def dataset(train_data):
    states = []
    wins = []
    for data in train_data:
        state, action, win = data
        state = state.reshape(22, 4, 3)
        states.append(state)
        wins.append(win)
    return np.array(states), np.array(wins)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=30, help='Number of passes of the entire dataset training')
    parser.add_argument('--model', '-m', default=None, help='initial model for training')
    parser.add_argument('--kifu', '-k', type=str, default='./test_play_results/', help='filepath of kifu to load')
    args = parser.parse_args()

    model = ValueNetwork()
    if args.model != None:
        model.model = K.models.load_model(args.model)

    # 棋譜を読み込む
    train_data = read_kifu(args.kifu)

    # 学習ループ
    for epoch in range(args.epoch):
        x, y = dataset(train_data)
        history = model(x, y)
        print("epoch : {}   history: {}\n".format(epoch, history.history))

        # save model
        model.model.save('./model/value_model.hdf5')
        

if __name__ == "__main__":
    main()