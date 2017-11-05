# pacmanAgents.py
# ---------------
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


from pacman import Directions
from game import Agent
from heuristics import *
import random
import math

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];
        tempState = state;
        for i in range(0,len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
            else:
                break;
        # returns random action from all the valide actions
        return self.actionList[0];

class GreedyAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(scoreEvaluation(state), action) for state, action in successors]
        # get best choice
        bestScore = max(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)

class Win: pass
class Lose:pass
class UseUp:pass

def randomSeq(state):
    ret = []
    possible = state.getAllPossibleActions();
    for i in range(5):
        ret.append( possible[random.randint(0,len(possible)-1)]);
    return ret;

class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame

    def getAction(self, state):
        def move(state,action):
            if state.isWin():
                raise Win()
            if state.isLose():
                raise Lose()
            state = state.generatePacmanSuccessor(action)
            if state is None :
                raise UseUp()
            else:
                return state
        def evaluateSeq(state, seq):
            try:
                for action in seq:
                    state = move(state,action)
            except (Win,Lose):
                return scoreEvaluation(state)
            return scoreEvaluation(state)
        possible = state.getAllPossibleActions();

        randomAction=lambda: possible[random.randint(0,len(possible)-1)]
        seq=randomSeq(state)
        oldScore = evaluateSeq(state,seq)
        try :
            while True:
                tmp = seq[:]
                for i in range(5):
                    if random.uniform(0.0,1.0) > 0.5:
                        continue
                    tmp[i]=randomAction()
                #oldScore = scoreEvaluation(state)
                newScore = evaluateSeq(state,tmp)
                if newScore > oldScore:
                    tmp,seq=seq,tmp
                    oldScore = newScore
        except UseUp:
            return seq[0]
        return Directions.STOP


class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        #constant value table for selection algorithm
        self.SEQ_LEN=5;
        self.POPULATION=8;
        self.RANKING=[ 1/36.0,3/36.0, 6/36.0, 10/36.0, 15/36.0, 21/36.0, 28/36.0, 1.001 ]
        return;

    def select(self, rand):
        for i in range(len(self.RANKING)):
            if rand < self.RANKING[i]:
                return i
        assert (False)

        
    def evaluateSeq(self,initState,seq):
        #naive algorithm, maybe we can do optimization
        state = initState
        for move in seq:
            if state.isWin():
                return 100000.0;
            elif state.isLose():
                return -100000.0;
            state = state.generatePacmanSuccessor(move)
            if state is None:
                raise Exception("generatePacmanSuccessor() returns None")
        return scoreEvaluation(state);

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Genetic Algorithm instead of returning Directions.STOP
        population = []
        possible = state.getAllPossibleActions();
        randf = lambda : random.uniform(0.0,1.0)
        randint = random.randint
        for i in range(self.POPULATION):
            population.append( randomSeq(state) )
        #start algorithm
        while True:
            tmp=population[:]
            try:
                #sort on a copy, exception-safe
                tmp.sort (key=lambda x : self.evaluateSeq(state,x) )
                population,tmp = tmp,population
            except Exception:
                return population[-1][0]
            #selection
            for i in range(len(tmp)):
                tmp[i] = population[self.select(randf())]
            population,tmp = tmp,population
            #crossover
            for i in range(0,len(tmp),2):
                if randf() > 0.7:
                    #keep the pair
                    tmp[i],tmp[i+1]=population[i],population[i+1]
                    continue
                #otherwise crossover
                for j in range(self.SEQ_LEN):
                    g1,g2 = population[i][j], population[i+1][j]
                    if randf()>0.5:
                        g1,g2=g2,g1
                    tmp[i][j],tmp[i+1][j] = g1,g2
            population, tmp= tmp,population
            #mutation
            for i in population:
                if randf()>0.1:
                    continue
                i[randint(0,self.SEQ_LEN-1)] = possible[randint(0,len(possible)-1)]

        return Directions.STOP

class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;
    # GetAction Function: Called with every frame
    def getAction(self, state):
        # node is a tuple (under the hood python list) 
        # [ tot_score, tot_visit, parent, action, [list of children] ]
        def createNode(parent,action):
            return [0,0,parent,action, 
                    {Directions.SOUTH:None,
                        Directions.NORTH:None,
                        Directions.EAST:None,
                        Directions.WEST:None} ]

        root = createNode(None,Directions.STOP)

        def UCB (node):
            assert(not node is None)
            assert (root[1]!=0)
            return node[0]/node[1] + 1.0 * math.sqrt(math.log(root[1]) / node[1])


        def move(state,action):
            if state.isWin():
                raise Win()
            if state.isLose():
                raise Lose()
            state = state.generatePacmanSuccessor(action)
            if state is None :
                raise UseUp()
            else:
                return state

        def rollout(rootState, node, state):
            #do rollout and back propagate here
            assert( not state is None)
            seq=randomSeq(state)
            rethrow = False
            try:
                for action in seq:
                    old=state
                state = move(state,action)
            except (Lose, Win):
                pass
            except UseUp:
                rethrow = True
            #score = scoreEvaluation(state)
            score = normalizedScoreEvaluation(rootState, state)
            backPropagate(node,score)
            if rethrow:
                raise UseUp
            #let UseUp() go up, don't handle it

        def backPropagate(node, score):
            while True:
                if node is None:
                    break;
                node[0]+=score
                node[1]+=1
                node = node[2]

        def span(root,state):
            cur = root;
            rootState = state
            #keep invariant: state correspond to node
            try:
                while True:
                    assert (not cur is None)
                    children = cur[4]
                    legal = state.getLegalPacmanActions();
                    for action in legal:
                        if children[action] is None:
                            state = move( state,action)
                            children[action] = createNode(cur,action)
                            cur = children[action]
                            rollout(rootState, cur, state)
                            return
                    choice=Directions.STOP
                    score = -100000.0
                    for action in legal:
                        ucb = UCB( children[action] )
                        if  ucb> score:
                            score,choice = ucb, action
                    state = move(state,choice) #can't swap
                    cur = children[choice]              #these two lines
            except (Win,Lose):
                rollout(rootState, cur,state)

        while True:
            try:
                span(root,state)
            except UseUp:
                break

        children = root[4]
        idx=Directions.STOP
        visit_cnt = -1
        for k in children :
            if not children[k] is None and children[k][1]>visit_cnt:
                idx,visit_cnt = k, children[k][1]
        return children[idx][3]
