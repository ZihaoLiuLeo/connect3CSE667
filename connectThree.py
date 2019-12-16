import numpy as np
import copy as cp
from mdp import *
import matplotlib.pyplot as pt

BLANK = 0
WHITE = 1
BLACK = -1
###################################################
#   Zihao Liu, 12,16,2019                         #
#   mdp on connect four                           #
#   mdp.py from in-class material, Garret Katz    #
###################################################


###################################################
# Helper Function                                 #
# Algorithm to get any number of peices connected #
# in four ways learnede from:                     #
# https://www.cnblogs.com/mq0036/p/7831616.html   #
# Tree search algorithm learned from CIS667 class #
#  material                                       #
###################################################

def count_num(state, x, y, xd, yd, num_of_connectors, size_of_map, color):
    """
    helper function to count same pieces in any directions
    """
    total_count=0
    for step in range(1, num_of_connectors):
        if xd!=0 and(y+xd*step<0 or y+xd*step >=size_of_map):
            break
        if yd!=0 and(x+yd*step<0 or x+yd*step >=size_of_map):
            break
        """
        only calculate three connected ones
        """
        if state[x+yd*step][y+xd*step] == color:
            total_count+=1
        else:
            break
    return total_count

def have_three(state, x, y, color):
    """
    helper function to find if there is any three pieces
    """
    if(state[x][y] != color):
        return False
    
    directions = [[(-1, 0), (1, 0)],
                  [(0, -1), (0, 1)],
                  [(-1, 1), (1, -1)],
                  [(-1, -1), (1, 1)]]
    for axis in directions:
        axis_count=1
        for(xd,yd) in axis:
            axis_count += count_num(state, x, y, xd, yd, 2, 3, color)
            if axis_count>=3:
                return True
    return False

def must_win_helper(state, x, y, xd, yd, size_of_map):
    total_count=0
    if xd!=0 and(y+xd*3<0 or y+xd*3 >=size_of_map):
        return total_count
    if yd!=0 and(x+yd*3<0 or x+yd*3 >=size_of_map):
        return total_count
    if state[x+yd*3][y+xd*3] == 0:
        total_count+=1
    return total_count

def must_win_cond(state, x, y, color):
    if(state[x][y] == 0):
        directions = [[(-1, 0), (1, 0)],
                      [(-1, 1), (1, -1)],
                      [(-1, -1), (1, 1)]]
        for axis in directions:
            axis_count=1
            for(xd, yd) in axis:
                axis_count += count_num(state, x, y, xd, yd, 3, 4, color)
                if(axis_count>=3):
                    return True
                    axis_count+=must_win_helper(state, x, y, xd, yd, 4)
                    if axis_count>=4:
                        return True
    return False
                          

#############
# game part #
#############

class ConnectFour(object):
    def __init__(self, n):
        self.size = n
        self.gameMap = np.full((n,n), BLANK)
        self.hasWinner = False
        self.score = 0
        self.col_flag = np.zeros(n, dtype=bool)

    def printMap(self):
        print (self.gameMap)
    
    def getScore(self):
        for i in range(self.size):
            for j in range(self.size):                
                if have_three(self.gameMap, self.size - i - 1, j, 1):
                    self.score = 1
                    self.hasWinner = True
                    break
                elif have_three(self.gameMap, self.size - i - 1, j, -1):
                    self.score = -1
                    self.hasWinner = True
                    break                

    def move(self, col_index, player):
        if self.hasWinner == False:
            if col_index < self.size:
                col_array = self.gameMap[:, col_index]
                for index in range(col_array.size):
                    if col_array[col_array.size - index - 1] == BLANK:
                        col_array[col_array.size - index - 1] = player
                        self.gameMap[:, col_index] = col_array
                        self.getScore()
                        
                        if(self.hasWinner):
                            player = self.score
                        return True                    
                self.col_flag[col_index] = True
                return False
            else:
                return False
        else:
            return False

    def mapIsFull(self):
        return (self.col_flag == True).all()



####################
# Tree search part #
####################

def score(state):
    size = len(state)
    for i in range(size):
        for j in range(size):                
            if have_three(state, size - i - 1, j, 1):
                return 1
            elif have_three(state, size - i - 1, j, -1):
                return -1
            #elif must_win_cond(state, size - i - 1, j, 1):
                #return 80
            #elif must_win_cond(state, size - i - 1, j, -1):
                #return -80
    return 0

def move(state, col_index, player):
    size = len(state)
    new_state = cp.deepcopy(state)
    if col_index < size:
        col_array = new_state[:, col_index]
        for index in range(col_array.size):
            if col_array[col_array.size - index - 1] == 0:
                col_array[col_array.size - index - 1] = player
                new_state[:, col_index] = col_array
                return new_state                    
        return False
    else:
        return False
    
def DFS(state, player):
    """
    only used to compare performance
    From inclass materials of CIS667, Garret Katz
    """
    if score(state) in [-1, 1]: return 1

    size = len(state)
    has_child = False
    leaf_count = 0

    for col in range(size):
        child = move(state, col, player)
        if child is False: continue
        has_child = True
        num_leaves = DFS(child, 1 if player == -1 else -1)
        leaf_count += num_leaves

    if has_child: return leaf_count
    else: return 1
    

def mnx_ab(state, player, depth=0, alpha=-np.inf, beta=np.inf):
    """
    using alpha-beta pruning minimax tree to reduce the number of states
    and also keep a index-state dictionary to help create P table
    Pseufo code from inclass materials of CIS667, Garret Katz
    """
    global count   
    global state_dict
    global player_record_dict
    global valid_move_record_dict
    flag = False

    """
    check for all available actions for current state
    """
    valid_cols = (state == 0).any(axis=0)
    valid_moves = []   
    
    for col in range(len(valid_cols)):
        if(valid_cols[col]):
            valid_moves.append(col)

    """
    setup the dict for P table setup information passing 
    """
    for index, s in state_dict.items():
        if np.array_equal(state, s):
            flag = True
            
    if(flag == False):
        state_dict[count] = state
        player_record_dict[count] = player
        valid_move_record_dict[count] = valid_moves
        count+=1

    """
    start the minimax tree
    """
    v = score(state)
    if v in [-1, 1] or (state != 0).all():
        return v, [], 1

    v, a, n = [], [], 0

    #if(valid_moves != []):
        #print(count)
        #state_dict[count] = state
        #count+=1

    for vm in valid_moves:
        child = move(state, vm, player)
        v_c, a_c, n_c = mnx_ab(child, 1 if player == -1 else -1, depth+1, alpha, beta)
        v.append(v_c)
        a.append(a_c)
        n += n_c
        if player == 1:
            alpha = max(alpha, v_c)
            if alpha > beta: break
        else:
            beta = min(beta, v_c)
            if beta < alpha: break

    best = np.argmax(v) if player is 1 else np.argmin(v)
    return v[best], [valid_moves[best]] + a[best], n

#################
# Learning Part #
#################
      
def state_to_index(state):
    global state_dict
    for index, s in state_dict.items():
        if np.array_equal(s, state):
            return index
    return False

def index_to_state(index):
    global state_dict
    if(index < len(state_dict)):
        return state_dict[index]
    return False

def get_next_player(index):
    global player_record_dict
    return player_record_dict[index]

def get_valid_moves(index):
    global valid_move_record_dict
    return valid_move_record_dict[index]

def reward(state):
    return 100*score(state)

def r_shift(choice):
    if choice+1>2:
        return 0
    return choice+1

def plot_state(state):
    print(state)

def get_available_moves(state):
    v_cols = (state == 0).any(axis=0)
    va_moves = []   
    
    for col in range(len(v_cols)):
        if(v_cols[col]):
            va_moves.append(col)
    return va_moves


actions = [0, 1, 2]

def build():
    """
    dummy player for player 1
    smart AI for player -1
    """
    global state_dict
    
    num_states = len(state_dict)
    P = np.zeros((3, num_states, num_states))
    r = np.zeros(num_states)
    for i in range(num_states):
        state = index_to_state(i)
        r[i] = reward(state)
        next_player = get_next_player(i)
        choices = get_valid_moves(i)
        if(choices==[]):
            j = state_to_index(state)
            P[choice,i,j] = 1
        else:
            if next_player == 1:
                """
                if next step is smart AI's turn
                """
                for choice in choices:
                    """
                    AI has a slightly chance to move to a different col
                    """
                    
                    if(len(choices)==1):
                        new_state = move(state, choice, 1)
                        j = state_to_index(new_state)
                        P[choice,i,j] = 1
                    elif(len(choices)==2):
                        new_state = move(state, choice, 1)
                        j = state_to_index(new_state)
                        P[choice,i,j] = 0.7
                        """
                        shift accident right by 1
                        """
                        choice = r_shift(choice)
                        new_state = move(state, choice, 1)
                        j = state_to_index(new_state)
                        P[choice,i,j] = 0.3
                    else:
                        new_state = move(state, choice, 1)
                        j = state_to_index(new_state)
                        P[choice,i,j] = 0.8
                        choice = r_shift(choice)
                        new_state = move(state, choice, 1)
                        j = state_to_index(new_state)
                        P[choice,i,j] = 0.15
                        choice = r_shift(choice)
                        new_state = move(state, choice, 1)
                        j = state_to_index(new_state)
                        P[choice,i,j] = 0.05            
            else:
                """
                if next step is dummy AI's turn
                """
                for choice in choices:
                    new_state = move(state, choice, -1)
                    j = state_to_index(new_state)
                    P[choice,i,j] = 1
    """
    add pruned condition
    """
    for a in range(len(actions)):
        for i in range(len(state_dict)):
            prune_dict={}
            total = 0
            if(np.fabs(P[a,i,:].sum() - 1.) > .001):
                for j in range(len(state_dict)):
                    if P[a,i,j]>0:
                        prune_dict[j]=P[a,i,j]
                if(prune_dict=={}):
                    P[a,i,i]=1
                else:
                    for ind in prune_dict:
                        total += prune_dict[ind]
                    for x in prune_dict:
                        prune_dict[x]=prune_dict[x]/total
                        P[a,i,x]=prune_dict[x]
            
    return P,r        
        

if __name__ == '__main__':
    c4 = ConnectFour(3)
    s = c4.gameMap
    
    
    count = 0
    state_dict = {}
    player_record_dict = {}
    valid_move_record_dict = {}

    v_ab, actions_ab, n_ab = mnx_ab(s, 1)
    
    P, r = build()
    g = .5
    print(state_to_index(s))

    #test p table
    #num_states = len(state_dict)
    #for a in range(len(actions)):
    #    for i in range(num_states):
    #        assert(np.fabs(P[a,i,:].sum() - 1.) < .001)
                
    
    pi, u = policy_iteration(r,g,P,num_iters=10)
    mins = np.flatnonzero(np.fabs(u - u.min()) < 0.001)
    maxs = np.flatnonzero(np.fabs(u - u.max()) < 0.001)
    print(mins, maxs)
    pt.plot(u)
    pt.show()

    # Show min/max states
    for m in [mins, maxs]:
        for idx in m:
            #plot_state(index_to_state(idx))
            pt.show()

    pt.ion()
    rewards = []
    for t in range(200):
        state = c4.gameMap
        #print("###")
        #print("Game Start")
        re = 0
        prune_flag = False
        
        while(state_to_index(state)!=False or reward(state)==0):
            """
            dummy AI's turn
            """
            #print("dummy's turn")
            v_move = get_available_moves(state)
            v = np.random.choice(v_move, size=1)
            state = move(state, v[0], 1)
            #print(state)
            """
            smart AI's turn
            """
            new_v = get_available_moves(state)
            i = state_to_index(state)
            if(reward(state)!=100):
                if(len(new_v)>0 and i==False):
                    prune_flag = True
                else:
                    #print("smart's turn")
                    if(i!=False):
                        a = actions[pi[i]]
                        new_state = move(state, a, -1)
                        if(state_to_index(new_state)!=False):
                            state = new_state
                            #print(state)
                        else:
                            break
                    else:
                        break
            else:
                break

        # show reward
        if(prune_flag==True):
            re=100
        else:
            re=reward(state) 
        rewards.append(re)
        print(t, np.mean(rewards))
    
    

    

