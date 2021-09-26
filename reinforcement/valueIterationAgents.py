# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    from gridworld import Gridworld
    def __init__(self, mdp : Gridworld, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        iter = 0
        while(iter < self.iterations):
            values_copy = self.values.copy()
            states = self.mdp.getStates()
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                q_values = []
                for action in actions:
                    q_values.append(self.getQValue(state, action))
                if(len(q_values) > 0):
                    values_copy[state] = max(q_values)
            iter += 1
            self.values = values_copy

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        Q = 0
        for next_state, prob in transitions:
            q_value = (self.mdp.getReward(state,action, next_state) + self.discount * self.getValue(next_state))
            q_value = q_value * prob
            Q += q_value
        return Q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actions = self.mdp.getPossibleActions(state)
        actions_value = util.Counter()
        for action in actions:
            actions_value[action] = self.getQValue(state, action)
        return actions_value.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        iter = 0
        while(iter < self.iterations):
            state = states[iter % len(states)]
            iter += 1
            possible_actions = self.mdp.getPossibleActions(state)
            if(len(possible_actions) == 0):
                continue
            q_values = []
            for action in possible_actions:
                q_values.append(self.computeQValueFromValues(state, action))
            self.values[state] = max(q_values)
        return

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
    def computeHighestQValue(self, state):
        q_values = []
        possible_actions = self.mdp.getPossibleActions(state)
        if(len(possible_actions) == 0):
            return self.values[state]
        for action in possible_actions:
            q_values.append(self.computeQValueFromValues(state, action))
        return max(q_values)
    def computePredecessors(self):
        states = self.mdp.getStates()
        predecessors = {}
        for state in states:
            predecessors[state] = set()
        for state in states:
            possible_actions = self.mdp.getPossibleActions(state)
            for action in possible_actions:
                transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                for next_state, _ in transitions:
                    predecessors[next_state].add(state)
        return predecessors
    def runValueIteration(self):
        predecessors = self.computePredecessors()
        pq = util.PriorityQueue()
        states = self.mdp.getStates()
        for state in states:
            if(state == "TERMINAL_STATE"):
                continue
            max_q = self.computeHighestQValue(state)
            diff = abs(self.values[state] - max_q)
            pq.push(state, -diff)
        iter = 0
        while (iter < self.iterations):
            if(pq.isEmpty()):
                return
            state = pq.pop()
            self.values[state] = self.computeHighestQValue(state)
            for p in predecessors[state]:
                max_q = self.computeHighestQValue(p)
                diff = abs(self.values[p] - max_q)
                if(diff > self.theta):
                    pq.update(p, -diff)
            iter += 1

        return

        "*** YOUR CODE HERE ***"

