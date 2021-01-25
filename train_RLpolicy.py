import os
import argparse
import copy
import numpy as np
import random
from datetime import datetime
import tensorflow as tf
from tensorflow.python import keras as K
from tqdm import tqdm

import environment
import features
import reinforce
from network import SLPolicyNetwork

def main():
    tf.compat.v1.disable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', '-s', type=int, default=1000, help='Number of game sets played to train')
    parser.add_argument('--episode', '-e', type=int, default=32, help='Number of episodes played in 1 set')
    parser.add_argument('--model', '-m', type=str, default='./model/sl_model.hdf5', help='init model to train')
    args = parser.parse_args()

    model1 = SLPolicyNetwork()
    model1.model = K.models.load_model(args.model)
    agent = reinforce.REINFORCE(model1.model)

    for i in range(args.set):
        folder_path = "./model/rl/"
        files = [filename for filename in os.listdir(folder_path) if not filename.startswith('.')]
        if len(files) == 0:
            model2 = None
        else:
            model2_path = random.choice(files)
            model2 = K.models.load_model(folder_path + model2_path)
        
        env = environment.gameEnv()
        result = 0

        states, actions, rewards = [], [], []
        for e in range(args.episode):
            env.reset()
            reward = 0
            player_model = [agent.model, model2]
            color = 0
            flg = 100000
            episode_rewards = []
            while flg != 0 and abs(flg) != 1:
                state, action, reward, flg = env.turn(color, player_model[color])
                if color == 0:
                    states.append(state)
                    actions.append(action)
                    episode_rewards.append(reward)
                color = (color+1) % 2
            episode_rewards = [flg] * len(episode_rewards)
            rewards = rewards + episode_rewards
            if flg == 1:
                result += 1
        agent.update_policy(states, actions, rewards)
        del states, actions, rewards
        
        print("Set:" + str(i) + ", Result:" + str(result/args.episode))
            
        if (i + 1) % 64 == 0:
            print('saving rl_model...')
            date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            agent.model.save('./model/rl/model' + date + '.hdf5')
            print('successfuly saved')


if __name__ == "__main__":
    main()
