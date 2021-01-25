import os
import copy
import numpy as np
import random
from datetime import datetime
from tensorflow.python import keras as K

from features import state_to_input, action_to_output, rotate_action, rotate_state
import MCTS


class Game:

    def __init__(self, filename):
        self.State = GameState()
        self.piece_types = {'P':1, 'B':2, 'R':3, 'K':4, 'G':5}      # Pawn, Bishop, Rook, King, Gold
        self.move_num = 0
        self.kifu = ""
        self.filename = filename
        self.date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.mcts = MCTS.MCTS()
    
    def reset(self):
        self.pieces = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0], [2, 4, 3]], dtype=np.int)
        self.store = []
        self.opponent_pieces = np.array([[-3, -4, -2], [0, -1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int)
        self.opponent_store = []
        self.board = self.pieces + self.opponent_pieces
        self.last_state = []

    def isEnd(self, color):
        store = self.store if color == 0 else self.opponent_store
        for p in store:
            if p == 4 or p == -4:
                return 1
        if self.move_num > 100:
            return 0
        else:
            return 10
    
    def turn(self, color, model=None, mode='mcts'):
        self.color = color
        self.model = model
        self.mode = mode
        self.move_num += 1

        if self.color == 0:
            pieces = self.pieces
            opponent_pieces = self.opponent_pieces
            store = self.store
            opponent_store = self.opponent_store

        elif self.color == 1:
            pieces = self.opponent_pieces
            opponent_pieces = self.pieces
            store = self.opponent_store
            opponent_store = self.store

        next_move = self.best_move()
        #print("move_num {} : turn {} : {}".format(self.move_num, self.color, next_move))

        self.update_state(self.color, next_move, pieces, opponent_pieces, store)

        #print(self.opponent_store)
        #print(self.board)
        #print(self.store)

        board, _store, _opponent_store = copy.deepcopy(self.board), copy.deepcopy(self.store), copy.deepcopy(self.opponent_store)
        if self.model == None and self.mode == 'mcts':
            _pieces, _opponent_pieces = copy.deepcopy(self.pieces), copy.deepcopy(self.opponent_pieces)

            self.last_state = [board, _pieces, _opponent_pieces, [_store, _opponent_store]]
            self.mcts.update_root(next_move, self.last_state)

        # add action info to kifu
        self.kifu += str(self.color)  + ' ' + next_move + ":"

        # add board state info to kifu
        s = []
        for i in range(4):
            for j in range(3):
                s.append(str(self.board[i][j]))
        self.kifu += ' '.join(s) + ',' + ' '.join([str(d) for d in self.store]) + ',' + ' '.join([str(d) for d in self.opponent_store]) + "\n"

        flg = self.isEnd(self.color)

        if flg == 0:
            print("move_num {} : turn {} : draw".format(self.move_num, self.color))
            self.kifu += "draw\n"
            return True

        elif flg == 1:
            print("move_num {} : turn {} : win".format(self.move_num, self.color))
            self.kifu += str(color) + " win\n"
            return True
        
        point = self.mcts.get_value(board, [_store, _opponent_store])
        #print('eval : 0 - {} | 1 - {}'.format(point, 1- point))

    def best_move(self):
        if self.color == 0:
            pieces = self.pieces
            opponent_pieces = self.opponent_pieces
            store = self.store

        elif self.color == 1:
            pieces = self.opponent_pieces
            opponent_pieces = self.pieces
            store = self.opponent_store

        actions = self.State.legal_moves(self.color, pieces, opponent_pieces, store)
        Board, Store, Opponent_store = copy.deepcopy(self.board), copy.deepcopy(self.store), copy.deepcopy(self.opponent_store)

        if self.model == None:
            if self.mode == "random":
                bestmove = random.choice(actions)
            if self.mode == "mcts":
                action_probs = self.mcts.make_action_probs(Board, [Store, Opponent_store], actions)
                self.mcts.root.expand(action_probs)
                bestmove = self.mcts.select_move(self.color)

        else:
            x = np.array([state_to_input(Board, [Store, Opponent_store])])
            probs = self.model.predict(x)
            legal_probability = []
            for move in actions:
                promoted = 1 if len(move) == 6 else 0
                from_pos = (move[0], move[1])
                to_pos = (int(move[3]), int(move[4]))
                action = [from_pos, to_pos]
                if self.color == 1:
                    action = rotate_action(from_pos, to_pos)
                idx = action_to_output(action, promoted)
                legal_probability.append(probs[0][idx])
            
            legal_probability = np.array(legal_probability)
            #print(actions)
            if np.count_nonzero(legal_probability) == 0:
                bestmove = random.choice(actions)
                probs = np.array([[1/ len(actions)]*len(actions)])
                #print(probs)
            else:
                #print(legal_probability)
                best_idx = np.random.choice(range(len(legal_probability)), p=legal_probability/np.sum(legal_probability))
                bestmove = actions[best_idx]
        
        return bestmove
    
    def update_state(self, color, next_move, pieces, opponent, store):
        to_x, to_y = int(next_move[3]), int(next_move[4])
        
        # place piece from store
        if next_move[1] == '*':
            pieces[to_y][to_x] = self.piece_types[next_move[0]] * (1 if color == 0 else -1)
            store.remove(pieces[to_y][to_x])

        else :
            from_x, from_y = int(next_move[0]), int(next_move[1])
            # promote Pawn to Gold
            if len(next_move) == 6:
                pieces[from_y][from_x] = 5 if color == 0 else -5
        
            # caputre opponent's piece
            if opponent[to_y][to_x] != 0:
                # if Gold is captured, store as Pawn
                if abs(opponent[to_y][to_x]) == 5:
                    opponent[to_y][to_x] //= 5
                
                store.append(-opponent[to_y][to_x])
                opponent[to_y][to_x] = 0
                pieces[from_y][from_x], pieces[to_y][to_x] = pieces[to_y][to_x], pieces[from_y][from_x]

            else:
                pieces[from_y][from_x], pieces[to_y][to_x] = pieces[to_y][to_x], pieces[from_y][from_x]
        
        self.board = pieces + opponent


    def save_kifu(self):
        filename = "./" + self.filename + "/" + self.date + ".txt"
        file_path = os.path.dirname(filename)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        with open(filename,'w') as f:
            f.write(self.kifu)


class GameState:
    
    dxy = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
    piece_types = {1:'P', 2:'B', 3:'R', 4:'K', 5:'G'}

    # search legal moves
    
    def legal_moves(self, color, pieces, opponent, store):
        self.moves = []
        # search legal directions for each piece
        for j in range(4):
            for i in range(3):
                piece = abs(pieces[j][i])
                if piece != 0:
                    if piece == 1:
                        directions = [0] if color == 0 else [4]
                        self.make_legal_move(color, pieces, opponent, directions, piece, i, j, (i, j))
                    
                    if piece == 2:
                        directions = [1, 3, 5, 7]
                        self.make_legal_move(color, pieces, opponent, directions, piece, i, j, (i, j))

                    if piece == 3:
                        directions = [0, 2, 4, 6]
                        self.make_legal_move(color, pieces, opponent, directions, piece, i, j, (i, j))

                    if piece == 4:
                        directions = [0, 1, 2, 3, 4, 5, 6, 7]
                        self.make_legal_move(color, pieces, opponent, directions, piece, i, j, (i, j))

                    if piece == 5:
                        directions = [0, 1, 2, 4, 6, 7] if color == 0 else [0, 2, 3, 4, 5, 6]
                        self.make_legal_move(color, pieces, opponent, directions, piece, i, j, (i, j))

                elif opponent[j][i] == 0 and len(store) != 0:
                    for p in store:
                        self.moves.append("{}* {}{}".format(self.piece_types[abs(p)], i, j))
        return self.moves

    
    def make_legal_move(self, color, pieces, opponent, directions, ptype, from_x, from_y, root):
        
        for a in directions:
            to_x = from_x + self.dxy[a][0]
            to_y = from_y + self.dxy[a][1]
            if self.is_inside(to_x, to_y) and pieces[to_y][to_x] == 0:
                if ptype == 1:
                    # promote Pawn to Gold
                    # when promoting, add '+' at the end of info
                    if color == 0:
                        if to_y < 2:
                            self.moves.append("{}{} {}{}+".format(root[0], root[1], to_x, to_y))
                    if color == 1:
                        if to_y >= 2:
                            self.moves.append("{}{} {}{}+".format(root[0], root[1], to_x, to_y))

                self.moves.append("{}{} {}{}".format(root[0], root[1], to_x, to_y))

    
    def is_inside(self, x, y):
        # return if position is inside game board
        if x >= 0 and x <= 2 and y >= 0 and y <= 3:
            return True
        else: return False
