# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem : SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    from game import Directions
    stack = util.Stack()
    stack.push(problem.getStartState())
    parents = {}
    visited = {}
    result = []
    while not(stack.isEmpty()):
        current_state = stack.pop()
        if(current_state in visited):
            continue
        if(problem.isGoalState(current_state)):
            while(current_state != problem.getStartState()):
                next_state, side = parents[current_state]
                current_state = next_state
                result.append(side)
            return result[::-1]
        visited[current_state] = True
        for next_state, side, _ in problem.getSuccessors(current_state):
            if(next_state in visited):
                continue
            parents[next_state] = (current_state, side)
            stack.push(next_state)
    util.raiseNotDefined()

def breadthFirstSearch(problem : SearchProblem):
    queue = util.Queue()
    queue.push(problem.getStartState())
    visited = {}
    parents = {}
    visited[problem.getStartState()] = True
    while not(queue.isEmpty()):
        current_state = queue.pop()
        if(problem.isGoalState(current_state)):
            path = []
            while(current_state != problem.getStartState()):
                next_state, side = parents[current_state]
                path.append(side)
                current_state = next_state
            return path[::-1]
        for next_state, side, _ in problem.getSuccessors(current_state):
            if(next_state in visited):
                continue
            visited[next_state] = True
            queue.push(next_state)
            parents[next_state] = (current_state, side)    
    util.raiseNotDefined()

def uniformCostSearch(problem : SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()
    distances = {}
    parents = {}
    distances[problem.getStartState()] = 0
    fringe.push(problem.getStartState(), 0)
    while not(fringe.isEmpty()):
        current_state = fringe.pop()
        if(problem.isGoalState(current_state)):
            path = []
            while(current_state != problem.getStartState()):
                next_state, side = parents[current_state]
                current_state = next_state
                path.append(side)
            return path[::-1]
        for next_state, side, cost in problem.getSuccessors(current_state):
            if not(next_state in distances):
                distances[next_state] = distances[current_state] + cost
                fringe.push(next_state, distances[next_state])
                parents[next_state] = (current_state, side)
            elif(distances[next_state] > distances[current_state]  + cost):
                fringe.update(next_state, distances[current_state] + cost)
                parents[next_state] = (current_state, side)
                distances[next_state] = distances[current_state] + cost

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem : SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    fringe = util.PriorityQueue()
    distance = {}
    parents = {}
    fringe.push(problem.getStartState(), heuristic(problem.getStartState(), problem))
    distance[problem.getStartState()] = 0
    while not(fringe.isEmpty()):
        current_state = fringe.pop()
        if(problem.isGoalState(current_state)):
            path = []
            while(current_state != problem.getStartState()):
                next_state, side = parents[current_state]
                path.append(side)
                current_state = next_state
            return path[::-1]
        for next_state, side, cost in problem.getSuccessors(current_state):
            if(next_state not in distance):
                distance[next_state] = distance[current_state] + cost
                fringe.push(next_state, distance[next_state] + heuristic(next_state, problem))
                parents[next_state] = (current_state, side)
            elif(distance[next_state]  > distance[current_state]  + cost):
                distance[next_state] = distance[current_state] + cost
                fringe.update(next_state, distance[next_state] + heuristic(next_state, problem))
                parents[next_state] = (current_state, side)
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
