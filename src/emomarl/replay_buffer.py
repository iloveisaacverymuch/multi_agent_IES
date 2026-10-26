import numpy as np


class ReplayBuffer:
    def __init__(self, args):
        self.N = args.N
        self.state_dim = args.state_dim
        self.obs_dim_per_agent = args.obs_dim_per_agent
        self.action_dim_per_agent = args.action_dim_per_agent
        self.buffer_size = args.buffer_size
        self.buffer = None
        self.current_size = 0
        self.is_full = False
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {}
        for i in range(self.N):
            self.buffer[i] = {
                'states': np.zeros((self.buffer_size, self.obs_dim_per_agent[i])),
                'actions': np.zeros((self.buffer_size, self.action_dim_per_agent[i])),
                'rewards': np.zeros((self.buffer_size, 1)),
                'next_states': np.zeros((self.buffer_size, self.obs_dim_per_agent[i])),
                'state_value': np.zeros((self.buffer_size, 1)),
                'a_log_prob': np.zeros((self.buffer_size, 1)),
                'dones': np.ones((self.buffer_size, 1))
            }
        self.current_size = 0
        self.is_full = False

    def store_transition(self, num_buffer, index, state, action, r, next_state, state_value, a_log_prob, done):
        self.buffer[index]['states'][num_buffer] = state
        self.buffer[index]['actions'][num_buffer] = action
        self.buffer[index]['rewards'][num_buffer] = r
        self.buffer[index]['next_states'][num_buffer] = next_state
        self.buffer[index]['state_value'][num_buffer] = state_value
        self.buffer[index]['a_log_prob'][num_buffer] = a_log_prob
        self.buffer[index]['dones'][num_buffer] = done


        if num_buffer >= self.buffer_size - 1:
            self.is_full = True

        if not self.is_full:
            self.current_size = num_buffer + 1
        else:
            self.current_size = self.buffer_size

    def clear(self):
        self.reset_buffer()

    def buffer_len(self):
        return self.current_size

    def get_data(self):
        """返回有效数据（只取前 current_size 条）"""
        if self.current_size == 0:
            return None

        valid_buffer = {}
        for i in range(self.N):
            valid_buffer[i] = {
                'states': self.buffer[i]['states'][:self.current_size].copy(),
                'actions': self.buffer[i]['actions'][:self.current_size].copy(),
                'rewards': self.buffer[i]['rewards'][:self.current_size].copy(),
                'next_states': self.buffer[i]['next_states'][:self.current_size].copy(),
                'state_value': self.buffer[i]['state_value'][:self.current_size].copy(),
                'a_log_prob': self.buffer[i]['a_log_prob'][:self.current_size].copy(),
                'dones': self.buffer[i]['dones'][:self.current_size].copy()
            }
        return valid_buffer