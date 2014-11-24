# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from util import manhattanDistance
from game import Directions
import random, util, sys
import pdb

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        print "GAME INFORMATION"
        successorGameState = currentGameState.generatePacmanSuccessor(action); print repr(successorGameState)
        pacman = successorGameState.agentStates[0]
        newPos = successorGameState.getPacmanPosition(); print repr(newPos)
        newFood = successorGameState.getFood(); print repr(newFood)
        newGhostStates = successorGameState.getGhostStates(); print repr(newGhostStates)
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]; print repr(newScaredTimes)

        closestGhost = min(newGhostStates, lambda ghost: manhattanDistance(pacman.pos, ghost.pos))
        ghostDist = manhattanDistance(closestGhost.pos, pacman.pos)
        ghostFactor = (closestGhost.scaredTimer - ghostDist)

        closestFood = (sys.maxint, sys.maxint)
        foodDist = sys.maxint
        foodCount = 0

        for i in range(newFood.width):
            for j in range(newFood.height):
                newDist = manhattanDistance(pacman.pos, (i, j))
                if newFood[i][j]:
                    foodCount += 1
                    if newDist < _foodDist:
                        foodDist = newDist
                        closestFood = (i, j)

        if foodDist == sys.maxint:
            return foodDist

        
        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

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

    def nextNode(self, gameState, agentIndex, depth):
        # returns tuple of (newAgent, newDepth)
        newAgentIndex = agentIndex + 1
        nextNode = (0, depth + 1) if newAgentIndex >= gameState.getNumAgents() else (newAgentIndex, depth)
        #print "curNode:(agentIndex: {0}, depth: {1})\nnextNode:(agentIndex: {2}, depth: {3})".format(agentIndex, depth, nextNode[0], nextNode[1])
        return nextNode

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '1'):
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
        """
        bestScore = -sys.maxint - 1
        bestAction = None
        for action in gameState.getLegalActions(0):
            if self.getActionValue(gameState.generateSuccessor(0, action), 0, 1) > bestScore:
                bestAction = action
        return bestAction

    def getActionValue(self, gameState, depth, agentIndex):
        # returns minimax score of this agent with this gameState
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if depth == self.depth and agentIndex == gameState.getNumAgents() - 1:
            # Base Case
            evalFnStr = "endgame"
            def evalFn(action):
                return self.evaluationFunction(gameState.generateSuccessor(agentIndex, action))
        else:
            evalFnStr = "continue"
            def evalFn(action):
                newAgentIndex, newDepth = self.nextNode(gameState, agentIndex, depth)
                #print newAgentIndex, newDepth
                return self.getActionValue(gameState.generateSuccessor(agentIndex, action), newDepth, newAgentIndex)
        orderingFn = max if agentIndex == 0 else min
        print "depth: {0}, agentIndex: {1}, orderingFn: {2}, evalFn: {3}, numAgents: {4}".format(depth, agentIndex, str(orderingFn), evalFnStr, gameState.getNumAgents())
        return orderingFn(map(evalFn, gameState.getLegalActions(agentIndex)))

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        bestAction = None
        alpha = -sys.maxint - 1
        beta = sys.maxint
        for action in gameState.getLegalActions(0):
            value = self.getActionValue(gameState.generateSuccessor(0, action), 0, 1, alpha, beta)
            if value > alpha:
                alpha = value
                bestAction = action
            if beta <= alpha:
                break
        return bestAction

    def getActionValue(self, gameState, depth, agentIndex, alpha, beta):
        if depth == self.depth and agentIndex == gameState.getNumAgents() - 1:
            # Base Case
            def evalFn(action, alpha, beta):
                return self.evaluationFunction(gameState.generateSuccessor(action))
        else:
            def evalFn(action, alpha, beta):
                newDepth, newAgentIndex = self.nextNode(gameState, depth, agentIndex)
                return self.getActionValue(gameState.generateSuccessor(action), newDepth, newAgentIndex, alpha, beta)
        if agentIndex == 0:
            for action in gameState.getLegalActions(agentIndex):
                value = evalFn(action, alpha, beta)
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    break
            return alpha
        else:
            for action in gameState.getLegalActions(agentIndex):
                value = evalFn(action, alpha, beta)
                if value < beta:
                    beta = value
                if alpha >= beta:
                    break
            return beta


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
        bestAction = None
        maxValue = -sys.maxint - 1
        for action in gameState.getLegalActions(0):
            value = self.getActionValue(gameState.generateSuccessor(0, action), 0, 1)
            if value > maxValue:
                bestAction = action
                maxValue = maxValue
        return bestAction

    def getActionValue(self, gameState, depth, agentIndex):
        if depth == self.depth and agentIndex == gameState.getNumAgents() - 1:
            # base case
            def evalFn(action):
                return self.evaluationFunction(gameState.generateSuccessor(agentIndex, action))
        else:
            def evalFn(action):
                newDepth, newAgentIndex = self.nextNode(gameState, depth, agentIndex)
                return self.getActionValue(gameState.generateSuccessor(agentIndex), newDepth, newAgentIndex)
        if agentIndex == 0:
            return max(map(evalFn, gameState.getLegalActions()))
        else:
            return float(sum(map(evalFn, gameState.getLegalActions()))) / len(gameState.getLegalActions())

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

