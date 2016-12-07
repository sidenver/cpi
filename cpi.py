from game import game
from greedyPolicy import GreedyPolicy

if __name__ == '__main__':
    env = game()
    greedyChooser = GreedyPolicy(env)
    # TODO initalize policy
    policy = greedyChooser.getGreedyPolicy([])  # initalized here
    
