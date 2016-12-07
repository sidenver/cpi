from game import game
from greedyPolicy import GreedyPolicy
from advantage import AdvantageEstimator
from policy import Policy

if __name__ == '__main__':
    env = game()
    dist = 0  # restart distribution
    discount = 0.5  # discount factor
    iteration = 1000  # number of iteration of learning
    # TODO accuracy?
    accuracy = 0.9  # accuracy
    sampleSize = 100  # number of sample's states
    horizon = 1000  # finite horizon
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
        if estimate['advantage'] < (accuracy * 2. / 3.):
            break
