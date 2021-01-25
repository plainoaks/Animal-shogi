import numpy as np
import argparse
import tensorflow as tf
from tensorflow.python import keras as K

from network import SLPolicyNetwork
from MCTS import MCTS
from game import Game


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--player1', '-p1', type=str, default=None, help='network model of player1')
    parser.add_argument('--player2', '-p2', type=str, default=None, help='network model of player2')
    parser.add_argument('--mode1', type=str, default='mcts', help='mode of the game for p1')
    parser.add_argument('--mode2', type=str, default='mcts', help='mode of the game for p2')
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--filename', '-f', type=str, default='test_play_results', help='filename to save kifu')
    args = parser.parse_args()
    
    model1 = None
    model2 = None
    if args.player1 != None:
        model1 = K.models.load_model(args.player1)
    if args.player2 != None:
        model2 = K.models.load_model(args.player2)
    play = Game(args.filename)
    play.reset()

    print("game start")

    # start game loop
    while True:
        if play.turn(0, model1, args.mode1):
            break
        if play.turn(1, model2, args.mode2):
            break
    
    print("game is over")

    # save kifu in file
    if args.save:
        play.save_kifu()

if __name__ == "__main__":
    main()
