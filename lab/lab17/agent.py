import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.score = 0
        self.alpha = .01
        self.gamma = 1.0
        self.Q = defaultdict(lambda: np.zeros(self.nA))
    
    def update_Q(Qsa, Qsa_next,reward, alpha, gamma):
         return Qsa + (alpha * (reward + gamma*np.max(Qsa_next) - Qsa))
    
    def epsilon_greedy(env, Qs, i_episode, eps=None):
        epsilon = 1.0 / i_episode

        #if epsilon is not passed by parameter
        if eps is not None:
            epsilon = eps

        policy_s = np.ones(env.nA) * epsilon / env.nA
        policy_s[np.argmax(Qs)] = 1 - epsilon + (epsilon/env.nA)
        
        return policy_s
    
    def select_action():
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment
        
        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.random.choice(np.arange(env.nA), p=self.policy)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """       
        self.score += reward

        if done:
            # update TD estimate of Q
            self.Q[state][action] = update_Q(self.Q[state][action], 0, reward, self.alpha, self.gamma)
            # append score
            tmp_scores.append(score)
            return
        if not done:
            self.policy = epsilon_greedy(env, self.Q[next_state], i_episode)

            next_action = select_action()

            self.Q[state][action] = update_Q(self.Q[state][action], self.Q[next_state][next_action], reward, alpha,gamma)

            state = next_state
            action = next_action
            
        if (i_episode % plot_every == 0):
                scores.append(np.mean(tmp_scores))
        return self.Q