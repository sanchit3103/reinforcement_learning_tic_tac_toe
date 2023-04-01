import numpy as np

class ticTacToe():
    def __init__(self, _state, _state_policy_dict):

        # Initialization of all parameters
        self.state              = _state
        self.optimal_actions    = _state_policy_dict
        self.traj               = []
        self.action_list        = []
        self.transition_prob    = []
        self.rewards_list       = []
        self.gen_reward         = 1
        self.win_reward         = 10
        self.lose_reward        = -10
        self.draw_reward        = 0
        self.winner             = False
        self.terminal_state     = False

    # Implementation of random trajectory generator for the MDP of tic-tac-toe game
    def random_trajectory_generator(self, optimal_action = False):

        # Initialize trajectory with the initial state
        self.traj.append(np.copy(self.state))

        # Check if the initial state is a terminal state
        self.check_terminal_state()

        if not self.terminal_state:
            self.rewards_list.append(self.gen_reward)

        # Loop for the game until the terminal state is encountered
        while not self.terminal_state:

            # Player's move (Player plays 'O')
            if optimal_action:
                self.player_move_with_optimal_action()

            else:
                self.player_move()

            # Check for terminal state after player's move
            self.check_terminal_state()

            # Add win reward if its a terminal state after player's move
            if self.terminal_state:
                self.traj.append(np.copy(self.state))
                self.rewards_list.append(self.win_reward)
                break

            # Opponent's move (Opponent plays 'X')
            self.opponent_move()

            # Check for terminal state after Opponent's move
            self.check_terminal_state()

            # Add lose reward if its a terminal state after opponent's move, otherwise add general reward
            if self.terminal_state:
                self.traj.append(np.copy(self.state))

                if self.winner:
                    self.rewards_list.append(self.lose_reward)

                break

            else:
                self.rewards_list.append(self.gen_reward)

            self.traj.append(np.copy(self.state))

        return self.traj, self.action_list, self.transition_prob, self.rewards_list

    # Function to check if a given state is a terminal state
    def check_terminal_state(self):

        # Check if there is a winner in the state; determine terminal state accordingly
        self.check_winner()

        ind_empty_space = np.where(self.state == '')

        # Case when there are empty spaces in the game but there is a winner
        if len(ind_empty_space[0]) != 0 and self.winner:

            self.terminal_state = True

        # Case when there is a winner and the game has ended as well
        elif len(ind_empty_space[0]) == 0 and self.winner:

            self.terminal_state = True

        # Case when there is no winner but the game has ended
        elif len(ind_empty_space[0]) == 0 and not self.winner:

            self.terminal_state = True
            self.rewards_list.append(self.draw_reward)

        else:
            self.terminal_state = False


    def check_winner(self):

        # Check if there is a winner across the two diagonals
        diag_value_1 = self.state[0, 0] + self.state[1, 1] + self.state[2, 2]
        diag_value_2 = self.state[0, 2] + self.state[1, 1] + self.state[2, 0]

        if diag_value_1 == 'XXX' or diag_value_1 == 'OOO':
            self.winner = True

        elif diag_value_2 == 'XXX' or diag_value_2 == 'OOO':
            self.winner = True

        # Check if there is a winner across rows or columns
        elif not self.winner:
            for i in range(np.shape(self.state)[0]):
                row_value = self.state[i, 0] + self.state[i, 1] + self.state[i, 2]
                col_value = self.state[0, i] + self.state[1, i] + self.state[2, i]

                if row_value == 'XXX' or row_value == 'OOO' or col_value == 'XXX' or col_value == 'OOO':
                    self.winner = True

        else:
            self.winner = False

    # Function to implement player's random move
    def player_move(self):
        indices = np.where(self.state == '')
        index   = np.random.choice(len(indices[0]))

        self.transition_prob.append(1 / len(indices[0]))
        self.action_list.append([indices[0][index], indices[1][index]])

        self.state[indices[0][index], indices[1][index]] = 'O'

    # Function to implement player's move with optimal actions
    def player_move_with_optimal_action(self):
        _state  = self.matrix_to_tuple(self.state)
        _action = self.optimal_actions[_state]

        self.action_list.append(_action)

        self.state[_action[0], _action[1]] = 'O'

    # Function to implement opponent's random move
    def opponent_move(self):
        indices = np.where(self.state == '')
        index   = np.random.choice(len(indices[0]))

        self.state[indices[0][index], indices[1][index]] = 'X'

    # Helper function to convert a matrix to tuple, as states are added as tuple in state_action_dict
    def matrix_to_tuple(self, _state):
        row1 = (_state[0, 0], _state[0, 1], _state[0, 2])
        row2 = (_state[1, 0], _state[1, 1], _state[1, 2])
        row3 = (_state[2, 0], _state[2, 1], _state[2, 2])

        _tuple = (row1, row2, row3)

        return _tuple


# state           = np.chararray((3,3), unicode=True)
# state[:,:]      = ""
# rewards_to_go   = []
# gamma           = 0.9
#
# tct             = ticTacToe(state, {})
# tct.opponent_move()
#
# traj, action_list, transition_prob, rewards_list = tct.random_trajectory_generator()
#
# for i in range(len(traj)):
#     print('State: ')
#     print(traj[i])
#
#     if i != len(traj)-1:
#         print('Action taken by player: ', action_list[i])
#         print('Transition Probability: ', transition_prob[i])
#
#     reward = 0
#     for j in range(i, len(rewards_list)):
#         reward = reward + ( gamma**(j - i) * rewards_list[j] )
#
#     print('Reward-to-go: ', reward)
#     print('')
#     rewards_to_go.append(reward)
#
# print('Action: ', action_list)
# print('Transition Prob: ', transition_prob)
# print('Rewards: ', rewards_list)
# print('Rewards-to-go: ', rewards_to_go)
