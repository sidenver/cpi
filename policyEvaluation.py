from collections import Counter
import numpy as np

class PolicyEvaluation(object):

    def __init__(self, env, discount = 0.9):
        self.env = env
        self.discount = discount

    def getScore(self, policy, theta=0.00001):
        stateList = self.env.getAllStates()
        values = np.zeros(len(stateList))
        while True:
            delta = 0
            # For each state, perform a "full backup"
            oldV = np.copy(values)
            for state in stateList:
                v = 0
                # Look at the possible next actions
                if state.isTerminal():
                    values[state.getIndex()] = 0
                    continue
                actionProbDict = policy.getActionsWithProb(state)
                for action in actionProbDict.keys():
                    actionProb = actionProbDict[action]
                    # For each action, look at the possible next states...
                    for  nextState, prob in self.env.getNextStateListWithAction(state, action):
                        # Calculate the expected value
                        reward = self.env.getReward(state, action, nextState)
                        v += actionProb * prob * (reward + self.discount * oldV[nextState.getIndex()])
                # How much our value function changed (across any states)
                delta = max(delta, np.abs(v - oldV[state.getIndex()]))
                values[state.getIndex()] = v
            # Stop evaluating once our value function change is below a threshold
            if delta < theta:
                break
        return np.array(values)[0]
