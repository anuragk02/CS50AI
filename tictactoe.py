"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    xctr = 0
    octr = 0
    for row in board:
        xctr += row.count(X)
        octr += row.count(O)
    
    if xctr == octr:
        return X
    else:
        return O

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actionset = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                actionset.append([i,j])
    return actionset

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    bcopy = copy.deepcopy(board)
    try:
        if bcopy[action[0]][action[1]] != EMPTY:
            raise IndexError
        else:
            bcopy[action[0]][action[1]] = player(board)
            return bcopy
    
    except IndexError:
        print('Already Filled')
        
        
def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    columns = []
    for row in board:
        ctrx = row.count(X)
        ctro = row.count(O)
        if ctrx == 3:
            return X
        if ctro == 3:
            return O
        
    for i in range(len(board)):
        col = [row[i] for row in board]
        columns.append(col)
            
    for i in columns:
        ctrx = i.count(X)
        ctro = i.count(O)
        if ctrx == 3:
            return X
        if ctro == 3:
            return O
        
    if board[0][0] == X and board[1][1] == X and board[2][2] == X:
        return X
    if board[0][0] == O and board[1][1] == O and board[2][2] == O:
        return O
    if board[0][2] == X and board[1][1] == X and board[2][0] == X:
        return X
    if board[0][2] == O and board[1][1] == O and board[2][0] == O:
        return O
    
    return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    emptyctr = 0
    for row in board:
        emptyctr += row.count(EMPTY)
    if emptyctr == 0:
        return True
    elif winner(board) is not None:
        return True
    else:
        return False
    
def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    currentplayer = player(board)
    
    alpha = -math.inf
    beta = math.inf
    
    if currentplayer == X:
        v = -math.inf
        for action in actions(board):
            a = minval(result(board, action), alpha, beta)
            if a > v:
                v = a
                optimalaction = action
                
    else:
        v = math.inf
        for action in actions(board):
            a = maxval(result(board, action), alpha, beta)
            if a < v:
                v = a
                optimalaction = action
    
    return optimalaction

def maxval(board, alpha, beta):
    if terminal(board):
        return utility(board)
    v = -math.inf
    for action in actions(board):
        v = max(v, minval(result(board, action), alpha, beta))
        alpha = max(alpha, v)
        if beta <= alpha:
            break
    return v

def minval(board, alpha, beta):
    if terminal(board):
        return utility(board)
    v = math.inf
    for action in actions(board):
        v = min(v, maxval(result(board, action), alpha, beta))
        beta = min(beta, v)
        if beta <= alpha:
            break
    return v
    