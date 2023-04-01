import numpy as np
from tct import ticTacToe
import matplotlib.pyplot as plt

def get_child_states(_state: np.chararray) -> dict:

    # Input:
    # _state - 3x3 matrix representing a valid state

    # Output:
    # _child_states - Dictionary with key as position of player's move and value as list of corresponding child states

    _child_states   = {}
    indices         = np.where(_state == '') # Get indices of all empty slots

    # Return empty dictionary if no empty slots are left in a state
    if len(indices[0]) == 0:
        return {}

    # Get reward for the current state
    _reward         = get_reward(_state)

    # No child state if current state is a terminal state; proceed otherwise
    if _reward != -10 and _reward != 10:

        # Formulation of initial child states, when no player has played a move
        if len(indices[0]) == _state.shape[0]**2:
            for i in range(len(indices[0])):
                x                   = indices[0][i]
                y                   = indices[1][i]
                new_states          = []
                _new_state          = np.copy(_state)
                _new_state[x, y]    = 'X'               # Opponent plays the first move
                new_states.append(np.copy(_new_state))
                _child_states[(x,y)]= new_states

        # Formulation of child states for subsequent states in the tree
        else:
            # Vary all the positions for player's move
            for i in range(len(indices[0])):

                x                   = indices[0][i]
                y                   = indices[1][i]
                _intm_state         = np.copy(_state)
                _intm_state[x, y]   = 'O'

                new_states          = []

                # Check if it is a terminal state after player's move
                # If so, add the state as a child state and continue to a new position of player's move
                if get_reward(_intm_state) == 10:
                    _new_state              = np.copy(_intm_state)
                    new_states.append(np.copy(_new_state))
                    _child_states[(x,y)]    = new_states
                    continue

                # Collect all indices where opponent can play after player's move
                new_indices         = np.where(_intm_state == '')

                # Formulate the list of all child states and add to the dictionary
                for j in range(len(new_indices[0])):

                    x1                  = new_indices[0][j]
                    y1                  = new_indices[1][j]
                    _new_state          = np.copy(_intm_state)
                    _new_state[x1, y1]  = 'X'
                    new_states.append(np.copy(_new_state))

                _child_states[(x,y)]    = new_states

    return _child_states

def get_reward(_state: np.chararray) -> int:

    # Input:
    # _state - 3x3 matrix representing a valid state

    # Output:
    # Reward according to the winner and state

    empty_slots     = len(np.where(_state == '')[0]) # Find the number of empty slots in a state

    # Compute both the diagonal values
    diag_value_1    = _state[0, 0] + _state[1, 1] + _state[2, 2]
    diag_value_2    = _state[0, 2] + _state[1, 1] + _state[2, 0]

    # Check if opponent is a winner across any diagonal
    if diag_value_1 == 'XXX' or diag_value_2 == 'XXX':
        return -10

    # Check if player is a winner across any diagonal
    elif diag_value_1 == 'OOO' or diag_value_2 == 'OOO':
        return 10

    # Check winner across rows and columns if there is no winner across the diagonal
    for i in range(np.shape(_state)[0]):

        # Compute values across rows and columns
        row_value = _state[i, 0] + _state[i, 1] + _state[i, 2]
        col_value = _state[0, i] + _state[1, i] + _state[2, i]

        # Check if opponent is a winner across any row or column
        if row_value == 'XXX' or col_value == 'XXX':
            return -10

        # Check if player is a winner across any row or column
        if row_value == 'OOO' or col_value == 'OOO':
            return 10

    # No reward if there is no winner and it is not a draw state
    if empty_slots != 0:
        return 1

    # Draw state if there is no winner up till here and no empty slots left in state
    else:
        return 0

def matrix_to_tuple(_state: np.chararray) -> tuple:

    # Input:
    # _state - 3x3 matrix representing a valid state

    # Output:
    # _tuple - Valid state converted into a tuple

    row1    = (_state[0, 0], _state[0, 1], _state[0, 2])
    row2    = (_state[1, 0], _state[1, 1], _state[1, 2])
    row3    = (_state[2, 0], _state[2, 1], _state[2, 2])

    _tuple  = (row1, row2, row3)

    return _tuple

def tuple_to_matrix(_tuple: tuple) -> np.chararray:

    # Input:
    # _tuple - Tuple form of a valid state

    # Output:
    # _state - State converted into a 3x3 matrix form

    _state  = np.chararray((3,3), unicode=True)

    for i in range(3):
        for j in range(3):
            _state[i,j] = _tuple[i][j]

    return _state

# Declaration of global variables
state               = np.chararray((3,3), unicode=True)
state[:,:]          = ""
state_value_dict    = {}
state_policy_dict   = {}
num_states          = 0
num_iterations      = 0
counter             = 1
gamma               = 0.9

# Loop to form all the valid states
indices             = np.where(state == '')

# Execute while loop until all states are covered
while len(indices[0]) != 0:

    # Obtain child states for a particular state
    child_states    = get_child_states(state)

    # Initialize the value for each valid state
    for key in child_states.keys():
        for i in range(len(child_states[key])):
            state_tuple = matrix_to_tuple(child_states[key][i])

            # Check if the state already exists in dictionary, initiate its value otherwise
            if not state_tuple in state_value_dict.keys():
                num_states                      += 1
                state_value_dict[state_tuple]   = 1

    # Trick to obtain the state to be expanded next; indexing in dict.keys() does not work as list / array
    temp_counter    = 1
    for _state in state_value_dict.keys():
        if temp_counter == counter:
            state = tuple_to_matrix(_state)
            break
        temp_counter += 1

    indices = np.where(state == '')
    counter += 1

print('Total number of valid states: ', num_states)

# Temp list of values to store optimal values for each state
prev_max_value          = [0]*num_states

# Declarations for the given state to be tracked
tracked_state           = np.chararray((3,3), unicode=True)
tracked_state[:,:]      = ""
tracked_state[0,0]      = 'X'
tracked_state[0,2]      = 'X'
tracked_state[1,0]      = 'O'
tracked_state[1,1]      = 'O'
tracked_state[2,0]      = 'X'

tracked_state           = matrix_to_tuple(tracked_state)

tracked_state_val       = []
tracked_state_val.append(state_value_dict[tracked_state])

# Value Iteration Loop
while(True):
    num_iterations  += 1
    delta           = 0
    state_counter   = 0
    print('Iteration: ', num_iterations)

    for _state in state_value_dict.keys():
        state           = tuple_to_matrix(_state)
        indices         = np.where(state == '')
        reward          = get_reward(state)

        child_states    = get_child_states(state)
        sum_list        = []

        # Loop across all the actions possible, i.e., each position where player can move
        for key in child_states.keys():
            # Calculation of sum of P(s'|s,a)*U[s'] for each action - as part of Bellman Update
            sum = 0
            for j in range(len(child_states[key])):
                temp_state      = matrix_to_tuple(child_states[key][j])
                value           = state_value_dict[temp_state]
                transition_prob = 1 / len(child_states[key])
                sum             = sum + (transition_prob * value)

            sum_list.append(sum)

        # Calculate new value for the state and the action for which sum is maximum
        if len(sum_list) != 0:
            new_val     = reward + (gamma * np.max(sum_list))
            max_index   = np.argwhere(sum_list == np.max(sum_list))

            # If there is only one action for which the sum is maximum, take that as the optimal action
            if len(max_index) == 1:
                max_index                       = np.argmax(sum_list)
                max_value                       = sum_list[max_index]
                prev_max_value[state_counter]   = max_value
                temp_counter                    = 0

                for key in child_states.keys():
                    if temp_counter == max_index:
                        state_policy_dict[_state] = key
                        break
                    temp_counter += 1

            # Find best action if there are multiple actions for which the sum is maximum
            # Store best action as the optimal action
            else:
                max_index   = np.argmax(sum_list)
                max_value   = sum_list[max_index]
                if max_value > prev_max_value[state_counter]:
                    prev_max_value[state_counter]   = max_value
                    temp_counter                    = 0

                    for key in child_states.keys():
                        if temp_counter == max_index:
                            state_policy_dict[_state] = key
                            break
                        temp_counter += 1

        # Value update is just the reward of that state if the sum_list is empty
        else:
            new_val                     = reward
            state_policy_dict[_state]   = ()

        # Update the value of delta
        if np.abs(new_val - state_value_dict[_state]) > delta:
            delta = np.abs(new_val - state_value_dict[_state])

        # Update the value for each state as per Bellman Update
        state_value_dict[_state]    = new_val
        state_counter               += 1

    # Track the value of tracked_state across all iterations
    tracked_state_val.append(state_value_dict[tracked_state])

    # Break condition
    if delta < 0.1 * ((1 - gamma)/ gamma):
        break

# Required Outputs for the state to be tracked
print(tracked_state_val)
print(state_policy_dict[tracked_state])

# Plot showing value of Tracked State across iterations
plt.plot(range(0, num_iterations+1), tracked_state_val)
plt.xlabel('Iterations')
plt.ylabel('Value of Tracked State (Fig 1)')
plt.title('Plot showing value of Tracked State across iterations')
plt.show()

# Test with random trajectory generator
state               = np.chararray((3,3), unicode=True)
state[:,:]          = ""
rewards_to_go       = []
avg_rewards_to_go   = []
gamma               = 0.9

tct                 = ticTacToe(state, state_policy_dict)
# tct.opponent_move()

# Monte-carlo update for 100 iterations
for i in range(100):
    # Initialize state everytime
    state               = np.chararray((3,3), unicode=True)
    state[:,:]          = ""
    state[0,0]          = 'X'

    tct                 = ticTacToe(state, state_policy_dict)
    traj, action_list, transition_prob, rewards_list = tct.random_trajectory_generator(optimal_action = True)

    reward = 0
    for k in range(len(rewards_list)):
        reward = reward + ( gamma**(k) * rewards_list[k] )

    rewards_to_go.append(reward)
    avg = np.sum(rewards_to_go) / len(rewards_to_go)
    avg_rewards_to_go.append(avg)

# Plot for average reward-to-go of the state for K random trajectories
plt.plot(range(0, 100), avg_rewards_to_go)
plt.xlabel('k - Random Trajectory')
plt.ylabel('Average Reward-to-go')
plt.title('Monte-Carlo Plot')
plt.show()

# Following is the code to enable user input and play the game against the optimal actions determined
val = 0
for i in range(3):
    for j in range(3):
        val += 1
        state[i,j] = str(val)

print('Refer the state below to determine a number corresponding to a square')
print(state)

state               = np.chararray((3,3), unicode=True)
state[:,:]          = ""
reward              = get_reward(state)

while reward == 1:

    print('')
    opp_action = input('Choose a number between 1 to 9 to play in a square: ')

    try:
        if opp_action == str(1):
            assert(state[0,0] == '')
            state[0,0] = 'X'

        elif opp_action == str(2):
            assert(state[0,1] == '')
            state[0,1] = 'X'

        elif opp_action == str(3):
            assert(state[0,2] == '')
            state[0,2] = 'X'

        elif opp_action == str(4):
            assert(state[1,0] == '')
            state[1,0] = 'X'

        elif opp_action == str(5):
            assert(state[1,1] == '')
            state[1,1] = 'X'

        elif opp_action == str(6):
            assert(state[1,2] == '')
            state[1,2] = 'X'

        elif opp_action == str(7):
            assert(state[2,0] == '')
            state[2,0] = 'X'

        elif opp_action == str(8):
            assert(state[2,1] == '')
            state[2,1] = 'X'

        elif opp_action == str(9):
            assert(state[2,2] == '')
            state[2,2] = 'X'

    except:
        print('Square is already occupied! Choose again!')
        print('')
        continue

    reward                  = get_reward(state)

    if reward == 1:
        state_tuple             = matrix_to_tuple(state)
        player_action           = state_policy_dict[state_tuple]
        state[(player_action)]  = 'O'

    reward                  = get_reward(state)

    print(state)
