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
        dis = 1e6
        for food in newFood.asList():
            if manhattanDistance(food, newPos) < dis:
                dis = manhattanDistance(food, newPos)

        for ghost in newGhostStates:  # the impact of ghost surges as distance get close
            if ghost.scaredTimer == 0 and manhattanDistance(ghost.getPosition() , newPos) == 0:
                return -1e6
        return successorGameState.getScore() + 1.0/dis

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
        score = -1e6
        best_move = Directions.STOP
        num = gameState.getNumAgents() - 1

        def maxPlayer(gameState, depth):
            current_depth = depth + 1
            if gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions(0)) == 0 or current_depth == self.depth:
                return self.evaluationFunction(gameState)

            max_score = -1e6
            all_moves = gameState.getLegalActions(0)

            for move in all_moves:
                suc = gameState.generateSuccessor(0,move)
                max_score = max(max_score, minPlayer(suc, current_depth, 1))
            return max_score

        def minPlayer(gameState, depth, index):
            if gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions(0)) == 0:
                return self.evaluationFunction(gameState)

            min_score = 1e6
            all_moves = gameState.getLegalActions(index)

            for move in all_moves:
                suc = gameState.generateSuccessor(index, move)
                if index == num:
                    min_score = min(min_score, maxPlayer(suc, depth))
                else:
                    min_score = min(min_score, minPlayer(suc, depth, index + 1))
            return min_score


        for move in gameState.getLegalActions(0):
            next_state =gameState.generateSuccessor(0,move)
            current_score = minPlayer(next_state,0,1)
            if(current_score > score):
                score =current_score
                best_move = move
        return best_move





class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -1e6
        beta = 1e6
        best_move = self.maxPlayer(gameState,0,alpha,beta)[1]
        return best_move

    def maxPlayer(self,gameState,depth,alpha,beta):
        if gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions(0)) == 0 or depth >= self.depth:
            return self.evaluationFunction(gameState),Directions.STOP

        max_score = -1e6
        best_move = Directions.STOP
        all_moves = gameState.getLegalActions(0)
        for move in all_moves:
            suc = gameState.generateSuccessor(0,move)
            current_score = self.minPlayer(suc,depth,1,alpha,beta)

            if(current_score > max_score):
                max_score = current_score
                best_move = move

            if(max_score > beta):
                return max_score,best_move

            alpha = max(alpha,max_score)

        return max_score,best_move

    def minPlayer(self,gameState,depth,index,alpha,beta):
        if gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions(index)) == 0 or depth >= self.depth:
            return self.evaluationFunction(gameState)

        min_score = 1e6
        all_moves = gameState.getLegalActions(index)
        num = gameState.getNumAgents() - 1

        for move in all_moves:
            suc = gameState.generateSuccessor(index,move)

            if (index == num) :
                current_score = self.maxPlayer(suc,depth+1,alpha,beta)[0]
            else:
                current_score = self.minPlayer(suc,depth,index+1,alpha,beta)

            if(current_score < min_score):
                min_score = current_score

            if(min_score < alpha):
                return min_score

            beta = min(beta,min_score)

        return min_score



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
        best_move = self.maxPlayer(gameState, 0)[1]
        return best_move

    def maxPlayer(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions(0)) == 0 or depth >= self.depth:
            return self.evaluationFunction(gameState), Directions.STOP

        max_score = -1e6
        best_move = Directions.STOP
        all_moves = gameState.getLegalActions(0)
        for move in all_moves:
            suc = gameState.generateSuccessor(0, move)
            current_score = self.minPlayer(suc, depth, 1)

            if (current_score > max_score):
                max_score = current_score
                best_move = move

        return max_score, best_move

    def minPlayer(self, gameState, depth, index):
        if gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions(index)) == 0 or depth >= self.depth:
            return self.evaluationFunction(gameState)

        all_moves = gameState.getLegalActions(index)
        num = gameState.getNumAgents() - 1
        all_suc = [ gameState.generateSuccessor(index,move) for move in all_moves ]

        if(index == num):
            scores = [self.maxPlayer(suc, depth + 1)[0] for suc in all_suc]
        else:
            scores = [self.minPlayer(suc, depth, index + 1) for suc in all_suc ]

        return sum(scores)/len(scores)


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
