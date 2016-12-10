from env import Env

class State(object):
	def __init__(self, index, reward, isTerminal = False):
		self.index = index
		self.reward = reward
		self.isTerminal = isTerminal

	def isTerminal(self):
		return self.isTerminal

	def getReward(self):
		return self.reward


class LongGrid(Env):

	def __init__(self, width, reward):
		self.state = list()
		for i in range(width - 1):
			state = State(i , 0)
			self.state.append(state)
		# Terminal state
		self.state.append(State(width, reward, True))
