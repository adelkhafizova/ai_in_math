import copy 
import math
import random
from burau_package.scripts.free_scripts import tiered_sampling, min_invariant_in_array, largest_power_range, extend_in_all_ways_p
from burau_package.classes.generators import Generators

gens = Generators(2) #creates generators object
helper_dict = {1:"a",2:"b",3:"B",4:"A"}
helper_dict1 = {"A":0,"B":1,"b":2,"a":3}

class Node:
    def __init__(self, env_state, parent=None, action=None, reward = 0, is_terminal = False):
        self.env_state = env_state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        if parent == None:
            self.total_reward = reward
        else:    
            self.total_reward = self.parent.total_reward + reward
        self.is_terminal = is_terminal
        self.untried_actions = self.env_state.legal_actions()
        self.total_score = 0

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_term(self):
        return self.is_terminal

    def expand(self):
        action = self.untried_actions.pop()
        new_env = copy.deepcopy(self.env_state)
        _, reward, term, trunc, _ = new_env.step(action)
        child = Node(new_env, parent=self, action=action, reward=reward, is_terminal=term or trunc)
        self.children.append(child)
        return child

    def best_child(self, c=1.4):
        """Select child with best UCB1 score."""
        return max(self.children, key=lambda child:
                   (child.total_score / child.visits) +
                   c * math.sqrt(math.log(self.visits) / child.visits))
    

    def rollout(self):
        """Play random moves until the episode ends."""
        env = copy.deepcopy(self.env_state)
        total_reward = self.total_reward
        term = self.is_terminal
        trunc = False
        while not (term or trunc):
            actions = env.legal_actions()
            action = random.choice(actions)
            _, reward, term, trunc, _ = env.step(action)
            total_reward += reward
        return total_reward
    
    def rollout_baseline(self):
        base_dict = {"A" : gens.A, 
        "a" : gens.a, 
        "B" : gens.B,
        "b" : gens.b}
        tiered_dict = {"A" : self.env_state.matrix*gens.A, 
        "a" : self.env_state.matrix*gens.a, 
        "B" : self.env_state.matrix*gens.B,
        "b" : self.env_state.matrix*gens.b}
        #self.env_state.render()
        tiered_dict.pop(helper_dict[self.env_state.word[-1]])
        for _ in range(31-self.env_state.turn):
            m = 100
            tiered_dict = tiered_sampling(tiered_dict,min_num=m,max_num=m) 
            tiered_dict = extend_in_all_ways_p(base_dict,tiered_dict,1)
            #print(len(next(iter(tiered_dict))),min_invariant_in_array(tiered_dict,largest_power_range))
        key = tiered_sampling(tiered_dict,min_num=1,max_num=1)
        key = next(iter(key))
        #print(key)

        env = copy.deepcopy(self.env_state)
        total_reward = self.total_reward


        for letter in key:
            _, reward, term, _, _ = env.step(helper_dict1[letter])
            total_reward += reward
            #env.render()
        #print(total_reward)
        return total_reward
            
    def backpropagate(self, result):
        """Update stats up the tree."""
        self.visits += 1
        self.total_score += result
        if self.parent:
            self.parent.backpropagate(result)


    



def mcts_search(root_state, reward = 0, iterations=500, c = 1.4):
    root = Node(root_state,reward = reward)
    best_result = -100
    for _ in range(iterations):
        node = root

        # Selection
        while not node.is_term() and node.is_fully_expanded():
            node = node.best_child(c=c)

        # Expansion
        if not node.is_term():
            node = node.expand()

        # Simulation
        
        result = node.rollout_baseline()
        if result >= best_result:
            best_result = result
        # Backpropagation
        node.backpropagate(result)
    return root.best_child(c=0).action  # Return best move