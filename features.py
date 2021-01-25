import numpy as np

MOVE_DIRECTION = [RIGHT, RIGHT_UP, UP, LEFT_UP, LEFT, LEFT_DOWN, DOWN, RIGHT_DOWN, UP_PROMOTED] = range(9)
PIECE_TYPE = [PAWN, BISHOP, ROOK] = range(3)
PIECE = {'P':PAWN, 'B':BISHOP, 'R':ROOK}
BOARD_LABEL = [
    [1, 2, 4],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
]

class Board:
    def __init__(self, state, color):
        self.board = np.array([int(state[0].split(' ')[i]) for i in range(12)]).reshape(4, 3)
        self.store = state[color+1].split(' ')
        self.store = np.array([int(c) for c in self.store if not c == ''])
        self.opponent_store = state[2-color].split(' ')
        self.opponent_store = np.array([int(c) for c in self.opponent_store if not c == ''])

def rotate_state(board, store):
    store *= -1
    board *= -1
    board = np.rot90(board, 2)
    return board, store

def rotate_action(from_pos, to_pos):
    if from_pos[1] != '*':
        from_pos = (2 - int(from_pos[0]), 3 - int(from_pos[1]))

    return [from_pos, (2 - int(to_pos[0]), 3 - int(to_pos[1]))]

def state_to_input(board, stores):
    channels = []

    for turn in range(2):
        if turn == 1:
            _board, store = rotate_state(board, stores[turn])
        else:
            _board, store = board, stores[turn]
        # for each types of pieces on board
        for i in range(1, 6): # 1:Pawn 2:Bishop 3:Rook 4:King 5:Gold
            bb = np.zeros((4,3))
            for j in range(4):
                for k in range(3):
                    if _board[j][k] == i:
                        bb[j][k] = 1
            channels.append(bb)
        
        # for each types of pieces in store
        for i in range(1, 4):
            n = np.count_nonzero(store == i)
            for j in range(2):
                if j < n:
                    bb = np.ones((4,3))
                else:
                    bb = np.zeros((4,3))
                channels.append(bb)

    return channels

def action_to_output(action, promoted):
    from_x, from_y = action[0][0], action[0][1]
    to_x, to_y = int(action[1][0]), int(action[1][1])

    to_label = BOARD_LABEL[to_y][to_x]
    
    if from_y == '*':
        move_direction = len(MOVE_DIRECTION) + PIECE[from_x]
    else:
        from_x, from_y = int(action[0][0]), int(action[0][1])
        dx, dy = to_x - from_x, to_y - from_y

        if promoted:
            move_direction = UP_PROMOTED
        elif dx > 0 and dy == 0:
            move_direction = RIGHT
        elif dx > 0 and dy > 0:
            move_direction = RIGHT_UP
        elif dx == 0 and dy > 0:
            move_direction = UP
        elif dx < 0 and dy > 0:
            move_direction = LEFT_UP
        elif dx < 0 and dy == 0:
            move_direction = LEFT
        elif dx < 0 and dy < 0:
            move_direction = LEFT_DOWN
        elif dx == 0 and dy < 0:
            move_direction = DOWN
        elif dx > 0 and dy < 0:
            move_direction = RIGHT_DOWN
    

    return 4 * 3 * move_direction + to_label - 1