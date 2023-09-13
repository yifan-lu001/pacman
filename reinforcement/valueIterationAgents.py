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
    def __init__(self, mdp, discount = 0.9, iterations = 100):
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

        # Iterate over given iterations
        for i in range (self.iterations):
            prevValues = self.values.copy()
            states = self.mdp.getStates()

            # Iterate over the state space
            for state in states:
                self.values[state] = float('-inf') # Initialize values to -infinity
                actions = self.mdp.getPossibleActions(state)

                # Iterate over possible actions
                for action in actions:
                    qValue = 0
                    for s, p in self.mdp.getTransitionStatesAndProbs(state, action):
                        qValue += p * (self.mdp.getReward(state, action, s) + self.discount * prevValues[s])
                    aValue = max(self.values[state], qValue)
                    self.values[state] = aValue

                # Check to see if there was no possible action
                if self.values[state] == float('-inf'):
                    self.values[state] = 0.0

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
        "*** YOUR CODE HERE ***"
        # Just applies the equation that we learned in class
        q = 0
        for s, p in self.mdp.getTransitionStatesAndProbs(state, action):
            q += p * (self.mdp.getReward(state, action, s) + self.discount * self.values[s])
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # Check to make sure we aren't at a terminal state
        if self.mdp.isTerminal(state):
            return None
        else:
            maxNumber, decision = float('-inf'), None
            # Check all possible actions from the given state
            for action in self.mdp.getPossibleActions(state):
                qValue = self.computeQValueFromValues(state, action)
                # We want to find the maximum q-value
                if qValue > maxNumber:
                    maxNumber, decision = qValue, action
            return decision

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
        "*** YOUR CODE HERE ***"

        # Iterate over the given iterations
        for i in range(0, self.iterations):
            numStates = len(self.mdp.getStates())
            iteration = i % numStates
            s = self.mdp.getStates()[iteration]
            a = self.computeActionFromValues(s)
            # Make sure the action exists, then compute it
            if a != None:
                qValue = self.computeQValueFromValues(s, a)
            else:
                qValue = 0
            self.values[s] = qValue

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

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        pdcs = {}
        states = self.mdp.getStates()

        # Initialize states to sets to avoid duplicates
        for s in states:
            pdcs[s] = set()

        # Add data to sets
        for s in states:
            for action in self.mdp.getPossibleActions(s):
                for ns, pred in self.mdp.getTransitionStatesAndProbs(s, action):
                    # If it exists
                    if pred > 0:
                        pdcs[ns].add(s)

        # Initialize a priority queue, as recommended in the instructions
        priorityQueue = util.PriorityQueue()

        # Push data to the priority queue
        for s in states:

            # Calculate the q-values
            actions = self.mdp.getPossibleActions(s)
            qValues = util.Counter()
            for a in actions:
                qValues[a] = self.computeQValueFromValues(s, a)

            if len(qValues) > 0: # Make sure there exists q-values
                # Push the negative absolute differences to the queue
                priorityQueue.push(s, -(abs(self.values[s] - qValues[qValues.argMax()])))

        # Iterate over given iterations
        for i in range(0, self.iterations):
            # Check to see if the queue is empty; if so, return None
            if priorityQueue.isEmpty():
                return None
            else:
                s = priorityQueue.pop()

                # Calculate the q-values
                actions = self.mdp.getPossibleActions(s)
                qValues = util.Counter()
                for a in actions:
                    qValues[a] = self.computeQValueFromValues(s, a)     

                # Set the value to the max q-value
                self.values[s] = qValues[qValues.argMax()]

                # Iterate over the predecessors
                for pred in pdcs[s]:

                    # Calculate the q-values
                    actions = self.mdp.getPossibleActions(pred)
                    pqValues = util.Counter()
                    for a in actions:
                        pqValues[a] = self.computeQValueFromValues(pred, a)
                    d = abs(self.values[pred] - pqValues[pqValues.argMax()])

                    # Update the priority queue if needed
                    if d > self.theta:
                        priorityQueue.update(pred, -d)

