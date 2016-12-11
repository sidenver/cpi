# from game import game
from greedyPolicy import GreedyPolicy
from advantage import AdvantageEstimator
from policy import Policy
from longGrid import LongGrid

def policyEvaluate(env, policy, discount, epoch=100.0):
    startState = env.getStartState()
    score = 0.0
    for i in range(epoch):
        k = 0
        state = startState
        discountFactor = discount
        reward = 0.0
        while not state.isTerminal() and k < 500:
            action = policy.getAction(state)
            nextState = env.getNextStateWithAction(state, action)
            reward += discountFactor * env.getReward(state, action, nextState)
            state = nextState
            discountFactor = discountFactor * discount
            k = k + 1
        score += reward * (1 - discount)

    return score / epoch

if __name__ == '__main__':
    env = LongGrid(10, 100, 0.1)
    dist = 0  # restart distribution
    discount = 0.9  # discount factor
    iteration = 1000  # number of iteration of learning
    # TODO accuracy?
    accuracy = 0.3  # accuracy
    sampleSize = 100  # number of sample's states
    horizon = 500  # finite horizon
    greedyChooser = GreedyPolicy(env)
    # TODO initalize policy
    policy = Policy(env)  # initalized here
    advantageEstimator = AdvantageEstimator(env, dist, discount)
    k = 0
    while k < iteration:
        estimate = advantageEstimator.estimateAdvantage(greedyChooser, policy, sampleSize, horizon)
        alpha = (estimate['advantage'] - (accuracy / 3.)) * (1 - discount) / 4.
        policy.conservativeUpdate(estimate['newPolicy'], alpha)
        k = k + 1
        score = policyEvaluate(env, policy, discount, 100)
        print('Iteration {}: advantage {} , policy score {}'.format(str(k), str(estimate['advantage']), str(score)))
        # if estimate['advantage'] < (accuracy * 2. / 3.):
            # break
