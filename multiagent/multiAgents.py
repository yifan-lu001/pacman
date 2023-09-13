# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        distance = 0                   # Initialize a distance variable
        # Tries to find location of ghost(s), taking into account how many are 
        ghostDist = 1                  # Initialize a distance to ghost variable
        ghostsAdjacent = 0             # Shows how many ghosts are right next to Pac-Man
        for ghost in successorGameState.getGhostPositions():
            distance = util.manhattanDistance(newPos, ghost)
            ghostDist += distance
            if distance <= 1:   # If there is a ghost right next to Pac-Man
                ghostsAdjacent += 1

        # Tries to find distance to furthest food pellet
        foodList = newFood.asList()     # Represent the food positions as a list
        foodDist = -1                   # Initialize the distance to food pellet
        for food in foodList:
            distance = util.manhattanDistance(newPos, food)    # Use the Manhattan distance to approximate distance to food
            if foodDist >= distance or foodDist == -1:         # If no food or if there is further food
                foodDist = distance

        # We return higher numbers for higher ghost distances, lower food distances
        return successorGameState.getScore() - (1 / ghostDist) + (1 / foodDist) - ghostsAdjacent

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Helper function to perform minimax computations
        def minimax(agent, depth, gameState):
            # Special cases: we've won, lost, or have defined depth
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # If the agent is pacman, we should maximize
            if agent == 0:
                return max(minimax(1, depth, gameState.generateSuccessor(agent, nextState)) for nextState in gameState.getLegalActions(agent))
            
            # If the agent is one of the ghosts, we should minimize
            else:
                nextAgent = agent + 1
                if nextAgent == gameState.getNumAgents(): # Loop back around to Pac-Man
                    nextAgent = 0
                if nextAgent == 0:
                    depth = depth + 1
                return min(minimax(nextAgent, depth, gameState.generateSuccessor(agent, nextState)) for nextState in gameState.getLegalActions(agent))        
        
        # Run minimax for Pac-Man
        maxAction = float("-inf") # Nothing can be smaller than negative infinity, so we set max
        action = Directions.WEST
        for state in gameState.getLegalActions(0):
            util = minimax(1, 0, gameState.generateSuccessor(0, state))
            if util > maxAction or maxAction == float("-inf"):
                maxAction = util
                action = state

        return action
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Alpha-Beta Pruning requires a minimizing and a maximizing function

        # Minimizing helper function
        def minimize(agent, depth, state, alpha, beta):
            minValue = float("inf")
            nextAgent = agent + 1
            if nextAgent == state.getNumAgents(): # Loop back around to Pac-Man
                nextAgent = 0
            if nextAgent == 0:
                depth = depth + 1
            for s in state.getLegalActions(agent):
                minValue = min(minValue, prune(nextAgent, depth, state.generateSuccessor(agent, s), alpha, beta))
                if minValue < alpha:
                    return minValue
                else:
                    beta = min(beta, minValue)
            return minValue

        # Maximizing helper function
        def maximize(agent, depth, state, alpha, beta):
            maxValue = float("-inf")
            for s in state.getLegalActions(agent):
                maxValue = max(maxValue, prune(1, depth, state.generateSuccessor(agent, s), alpha, beta))
                if maxValue > beta:
                    return maxValue
                else:
                    alpha = max(alpha, maxValue)
            return maxValue

        # Alpha-Beta Pruning Function
        def prune(agent, depth, state, alpha, beta):
            # Special cases: we've won, lost, or have defined depth
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            elif agent == 0:
                return maximize(agent, depth, state, alpha, beta)
            return minimize(agent, depth, state, alpha, beta)

        # Run Alpha-Beta Pruning for Pac-Man
        util = float("-inf")
        action = Directions.WEST
        a = util
        b = float("inf")
        for s in gameState.getLegalActions(0):
            g = prune(1, 0, gameState.generateSuccessor(0, s), a, b)
            if g > util:
                action = s
                util = g
            if util > b:
                return util
            else:
                a = max(a, util)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Expectimax helper function
        # The only thing changed from minimax is to take the sum of expectimax instead of min
        # if considering one of the ghosts
        def expectimax(agent, depth, gameState):
            # Special cases: we've won, lost, or have defined depth
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # If the agent is pacman, we should maximize
            if agent == 0:
                return max(expectimax(1, depth, gameState.generateSuccessor(agent, nextState)) for nextState in gameState.getLegalActions(agent))
            
            # If the agent is one of the ghosts, we should minimize
            else:
                nextAgent = agent + 1
                if nextAgent == gameState.getNumAgents(): # Loop back around to Pac-Man
                    nextAgent = 0
                if nextAgent == 0:
                    depth = depth + 1
                return sum(expectimax(nextAgent, depth, gameState.generateSuccessor(agent, nextState)) for nextState in gameState.getLegalActions(agent))        
        
        # Run minimax for Pac-Man
        maxAction = float("-inf") # Nothing can be smaller than negative infinity, so we set max
        action = Directions.WEST
        for state in gameState.getLegalActions(0):
            util = expectimax(1, 0, gameState.generateSuccessor(0, state))
            if util > maxAction or maxAction == float("-inf"):
                maxAction = util
                action = state

        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Similar to the reflex agent evaluation function, we take into account ghost locations,
    ghost proximities, and food pellet locations. Score is proportional to ghost distances and inversely
    proportional to food distances. The only thing I really change from the reflex agent is to also 
    consider number of capsules available.
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    distance = 0
    numCapsules = len(currentGameState.getCapsules())

    # Tries to find location of ghost(s), taking into account how many are 
    ghostDist = 1                  # Initialize a distance to ghost variable
    ghostsAdjacent = 0             # Shows how many ghosts are right next to Pac-Man
    for ghost in currentGameState.getGhostPositions():
        distance = util.manhattanDistance(newPos, ghost)
        ghostDist += distance
        if distance <= 1:   # If there is a ghost right next to Pac-Man
            ghostsAdjacent += 1
        
    # Tries to find distance to furthest food pellet
    foodList = newFood.asList()     # Represent the food positions as a list
    foodDist = -1                   # Initialize the distance to food pellet
    for food in foodList:
        distance = util.manhattanDistance(newPos, food)    # Use the Manhattan distance to approximate distance to food
        if foodDist >= distance or foodDist == -1:         # If no food or if there is further food
            foodDist = distance

    # We return higher numbers for higher ghost distances, lower food distances
    return currentGameState.getScore() - (1 / ghostDist) + (1 / foodDist) - ghostsAdjacent - numCapsules

# Abbreviation
better = betterEvaluationFunction
