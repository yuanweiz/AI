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

class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Hill Climber Algorithm instead of returning Directions.STOP
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];
        return Directions.STOP

def randomSeq(state):
    ret = []
    possible = state.getAllPossibleActions();
    for i in range(5):
        ret.append( possible[random.randint(0,len(possible)-1)]);
    return ret;

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
                #print i
                i[randint(0,self.SEQ_LEN-1)] = possible[randint(0,len(possible)-1)]

        return Directions.STOP

class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;


    # GetAction Function: Called with every frame
    def getAction(self, state):
        WIN=0
        LOSE=1
        MORE=2
        USE_UP=3
        # node is a tuple (under the hood python list) 
        # [ tot_score, tot_visit, parent, action, [list of children] ]

        def createNode(parent,action):
            return [0,0,parent,action, [None,None,None,None] ]

        root = createNode(None,Directions.STOP)
        def UCB (node):
            assert(not node is None)
            assert (root[1]!=0)
            return node[0]/node[1] + 2 * math.sqrt(math.log(root[1]) / node[1])

        def rollout( node, state):
            #do rollout and back propagate here
            assert( not state is None)
            seq=randomSeq(state)
            status = MORE
            for action in seq:
                old=state
                state = state.generatePacmanSuccessor(action)
                if state is None:
                    state = old
                    status = USE_UP
                    break
                elif state.isWin():
                    status = WIN
                    break
                elif state.isLose():
                    status=LOSE
                    break
            score = scoreEvaluation(state)
            backPropagate(node,score)
            return status

        def backPropagate(node, score):
            while True:
                if node is None:
                    break;
                node[0]+=score
                node[1]+=1
                node = node[2]

        def span(root,state):
            print "span"
            cur = root;
            status = MORE
            def move(state,action):
                oldstate = state
                state = state.generatePacmanSuccessor(possible[i])
                if state is None:
                    return USE_UP,oldstate
                elif state.isWin():
                    return WIN,state
                elif state.isLose():
                    return LOSE,state
                else:
                    return MORE,state
            while True:
                print "while loop"
                assert (not cur is None)
                children = cur[4]
                oldstate = state
                for i in range (4):
                    if children[i] is None:
                        children[i] = createNode(cur,possible[i])
                        assert(not possible[i] is None)
                        state = state.generatePacmanSuccessor(possible[i])
                        status,state = getStatus(state)
                        return rollout(children[i], state)
                choice=-1
                score = -100000.0
                print "check terminal"
                if not terminal:
                    for i in range (4):
                        ucb = UCB( children[i] )
                        if  ucb> score:
                            score,choice = ucb, i
                    state = state.generatePacmanSuccessor(possible[choice])
                    if state is None:
                        terminal = True
                        state = oldstate
                    elif state.isWin() or state.isLose() :
                        terminal = True
                    cur = children[choice]
                else :
                    print "terminal"
                    score = scoreEvaluation(state)
                    if state.isWin():
                        backPropagate(cur,score)
                        return WIN
                    elif state is None:
                        backPropagate(cur,score)
                        return USE_UP
                    elif state.isLose():
                        backPropagate(cur,score)
                        return LOSE
                    assert (False)

        possible = state.getAllPossibleActions();

        #while true:
        cur = root;
        while span(root,state)==MORE:
            pass
        children = root[4]
        assert(not children is None)
        idx=-1
        visit_cnt = -1
        for i in range(len(children)):
            if not children[i] is None and children[i][1]>visit_cnt:
                print children[i][3]
                idx,visit_cnt = i, children[i][1]
        assert(idx!=-1)
        return children[idx][3]