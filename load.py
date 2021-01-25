import numpy as np
import os
import argparse
import features

def transform(line):
    data = line.replace("\n", "").split(':')
    action = data[0].split(' ')
    promoted = 1 if len(action[2]) == 3 else 0
    state = data[1].split(',')
    return state, action, promoted

def read_kifu(path):
    labels = []
    res = [0, 0]
    path = path
    files = [filename for filename in os.listdir(path) if not filename.startswith('.')]
    
    for file in files:
        with open(path + '/' + file, 'r') as f:
            lines = f.readlines()
        
        if "draw" in lines[-1]:
            continue
        else:
            win_color = int(lines[-1][0])
            res[win_color] += 1
        del lines[-1]

        for line in lines:
            state, action, promoted = transform(line)
            color = int(action[0])
            del action[0]
            board = features.Board(state, color)
            if color == 1:
                board.board, board.store = features.rotate_state(board.board, board.store)
                action = features.rotate_action(action[0], action[1])

            input_channels = features.state_to_input(board.board, [board.store, board.opponent_store])
            output_label = features.action_to_output(action, promoted)
            output_vector = np.zeros(144)
            output_vector[output_label] = 1

            win = 1.0 if color == win_color else 0
            win = [win]
            
            labels.append((np.array(input_channels), output_vector, np.array(win)))
    print("result black : {}  white : {}".format(res[0], res[1]))
        
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./test_play_results')
    args = parser.parse_args()

    labels = read_kifu(args.path)