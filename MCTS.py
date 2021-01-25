import copy
import time
import numpy as np
from tensorflow.python import keras as K
from network import SLPolicyNetwork, ValueNetwork, RolloutPolicy
from game import GameState as gs
from environment import gameEnv as ge
from features import state_to_input, action_to_output

class Node:

    def __init__(self, parent=None, prob=0):
        self.parent = parent
        self.children = {}
        self.visited = 0
        self.Q = 0
        self.u = prob
        self.P = prob
        self.pieces = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0], [2, 4, 3]], dtype=np.int)
        self.opponent_pieces = np.array([[-3, -4, -2], [0, -1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int)
        self.board = self.pieces + self.opponent_pieces
        self.store = []
        self.opponent_store = []
    
    def is_Leaf(self):
        return len(self.children) < 1
    
    def select(self):
        for k, v in self.children.items():
            v.u = v.P * np.sqrt(v.parent.visited) / (1 + v.visited)
        return max(self.children.items(), key=lambda node : node[1].Q + node[1].u)

    def expand(self, action_probs):
        for action, prob in action_probs:
            if action not in self.children:
                self.children[action] = Node(self, prob)

    def update(self, leaf_value):
        self.visited += 1
        self.Q += (leaf_value - self.Q) / self.visited
        if self.parent != None:
            self.parent.update(leaf_value)
    
class MCTS:
    
    def __init__(self, lmda=0.5, thr=5, time_limit=10):
        self.root = Node(prob=1.0)
        self.policy_net = SLPolicyNetwork()
        self.policy_net.model = K.models.load_model('./model/sl_model.hdf5')
        self.value_net = ValueNetwork()
        self.value_net.model = K.models.load_model('./model/value_model.hdf5')
        self.rollout = RolloutPolicy()
        self.rollout.model = K.models.load_model('./model/rollout_model.hdf5')
        self.lmda = lmda
        self.thr = thr
        self.time_limit = time_limit
        self.env = ge()
        self.gs = gs()

    def make_action_probs(self, state, stores, actions):
        state = state_to_input(state, stores)
        state = np.array([state])
        prob = self.policy_net.model.predict(state)[0]
        action_probs = []
        for action in actions:
            promoted = 1 if action[-1] == '+' else 0
            action_label = action.split(' ')
            action_label = action_to_output(action_label, promoted)
            action_probs.append((action, prob[action_label]))
        return action_probs

    def get_value(self, state, stores):
        state = state_to_input(state, stores)
        state = np.array([state])
        return self.value_net.model.predict(state)

    def playout(self, state, pieces, opponent, stores, color, node):
        c = copy.deepcopy(color)
        _state, _pieces, _opponent_pieces ,_stores = copy.deepcopy(state), copy.deepcopy(pieces), copy.deepcopy(opponent), copy.deepcopy(stores)
        last_state = [_state, _pieces, _opponent_pieces ,_stores]
        if node.is_Leaf():
            if node.visited >= self.thr:
                if self.env.isEnd((c+1)%2) > 1:
                    if c == 0:
                        e_pieces = _pieces
                        e_opponent_pieces = _opponent_pieces
                    elif c == 1:
                        e_pieces = _opponent_pieces
                        e_opponent_pieces = _pieces
                    actions = self.gs.legal_moves(c, e_pieces, e_opponent_pieces, _stores[c])
                    if len(actions) == 1:
                        node.children[actions] = Node(node, 1.0)
                    else:
                        action_probs = self.make_action_probs(_state, _stores, actions)
                        node.expand(action_probs)
                    self.playout(state, pieces, opponent, stores, c, node)
            else:
                v = self.get_value(state, stores)
                move_num = self.env.move_num
                z = self.rollout_res(c, move_num, last_state)
                leaf_value = (1 - self.lmda) * v + self.lmda * z
                node.update(leaf_value)
        
        else:
            action, node = node.select()
            if c == 0:
                e_pieces = self.env.pieces
                e_opponent_pieces = self.env.opponent_pieces
                e_store = self.env.store
                e_opponent_store = self.env.opponent_store

            elif c == 1:
                e_pieces = self.env.opponent_pieces
                e_opponent_pieces = self.env.pieces
                e_store = self.env.opponent_store
                e_opponent_store = self.env.store

            self.env.update_state(c, action, e_pieces, e_opponent_pieces, e_store)
            last_state = [self.env.board, self.env.pieces, self.env.opponent_pieces, [self.env.store, self.env.opponent_store]]
            c = (c + 1) % 2
            self.playout(last_state[0], last_state[1], last_state[2], last_state[3], c, node)
    
    def rollout_res(self, color, move_num, last_state):
        c = copy.deepcopy(color)
        self.env.board = copy.deepcopy(last_state[0])
        self.env.pieces = copy.deepcopy(last_state[1])
        self.env.opponent_pieces = copy.deepcopy(last_state[2])
        self.env.store = copy.deepcopy(last_state[3][0])
        self.env.opponent_store = copy.deepcopy(last_state[3][1])
        flg = self.env.isEnd((c+1)%2)
        while flg > 1:
            board, prob, reward, flg = self.env.turn(c, self.rollout.model)
            c = (c + 1) % 2
        self.env.move_num = move_num
        return flg

    def select_move(self, color):
        start = time.time()
        elapsed = 0
        while elapsed < self.time_limit:
            c = copy.deepcopy(color)
            self.env.board = copy.deepcopy(self.root.board)
            self.env.pieces = copy.deepcopy(self.root.pieces)
            self.env.opponent_pieces = copy.deepcopy(self.root.opponent_pieces)
            self.env.store = copy.deepcopy(self.root.store)
            self.env.opponent_store = copy.deepcopy(self.root.opponent_store)

            self.playout(self.env.board, self.env.pieces, self.env.opponent_pieces, [self.env.store, self.env.opponent_store], c, self.root)

            elapsed = time.time() - start
        return max(self.root.children.items(), key=lambda node: node[1].visited)[0]

    def update_root(self, last_move, last_state):
        self.root = self.root.children[last_move]
        self.root.parent = None
        self.root.children = {}
        self.root.board = copy.deepcopy(last_state[0])
        self.root.pieces = copy.deepcopy(last_state[1])
        self.root.opponent_pieces = copy.deepcopy(last_state[2])
        self.root.store = copy.deepcopy(last_state[3][0])
        self.root.opponent_store = copy.deepcopy(last_state[3][1])
