# -*- coding: utf-8 -*-
"""
Agent-based model(ABM) for the COVID-19

This file includes,

class Agent

functions:
simulation_basic()
socialNetwork()
simulation_social_network()
simulation_social_network_quarantine()
simulation_basic_reg_after7days()
simulation_basic_reg()

"""
import numpy as np
import networkx as nx
import math


class Agent:
    
    ### Create Constants - transition matrix(bioTransition), maximum and minimum duration of each state(bioMin, bioMax)
    ###                     and the number of states(__states)
    
    # total 7 states
    # * 0. Unexposed
    # * 1. Asymptomatic but infected/contagious
    # * 2. Symptomatic and contagious
    # * 3. Symptomatic and not contagious
    # * 4. Post-COVID Immune
    # * 5. Naturally immune (will not contract)
    # * 6. Death
    
    # private variable
    __states = 7
    
    # To keep it simple, we will make all timing distributions uniform 
    # with a min and max parameter for time in each state. 

    # initialization
    # bioTransition matrix describes the posibility to transfer from the state(row) to the state(column)
    bioTransition = np.zeros((__states,__states)) # row: the number of states, column: the number of states
    bioMin = np.ones(__states, dtype=int) # state time minimum
    bioMax = np.ones(__states, dtype=int) # state time maximum

    # 1. Asymptomatic but infected/contagious for 3 to 10 days
    bioMin[1] = 3 
    bioMax[1] = 10
    # bio Transition 1 -> 2(Symptomatic and contagious)
    bioTransition[1,2] = 0.5
    #bio Transition 1 -> 4(Post-COVID Immune)
    bioTransition[1,4] = 0.5
    #print(bioMin)

    # 2. Symptomatic and contagious for 3 to 8 days
    bioMin[2] = 3
    bioMax[2] = 8
    # bio Transition 2 -> 3(Symptomatic and not contagious)
    bioTransition[2,3] = 0.95
    # bio Transition 2 -> 6(Death)
    bioTransition[2,6] = 0.05

    # 3. Symptomatic and not contagious for 1 to 7 days
    bioMin[3] = 1
    bioMax[3] = 7
    # bio Transition 3 -> 4(Post-COVID Immune)
    bioTransition[3,4] = 1
    
    # constructor
    def __init__(self, biostate, age=30, nextbiostate=np.nan, biostatecountdown=np.nan): # nextbiostate, biostatecountdown initial values are set to np.nan not -1
        self.biostate = biostate
        self.age = age
        self.nextbiostate = nextbiostate
        self.biostatecountdown = biostatecountdown
    
    # set the state of the agent
    def setAgentState(self, biostate):
        self.biostate = biostate
        if np.sum(self.bioTransition[biostate,] > 0): # this state transitions to something else
            # which state do we go to?
            self.biostatecountdown = np.random.choice(a = np.arange(self.bioMin[biostate],self.bioMax[biostate]+1), size = 1)[0]
            
            self.nextbiostate = np.random.choice(a = np.arange(0,self.__states), size = 1, p= self.bioTransition[self.biostate,])[0]
        else:
            self.biostatecountdown = np.nan
            self.nextbiostate = np.nan # so that we can tell if the agent is finished or not
    
    # transfer the current state to the 'nextbiostate'
    def transitionAgent(self): 
        self.setAgentState(self.nextbiostate)
    
    # update the agent
    def updateagent(self):
        if(not(np.isnan(self.biostatecountdown))):
            self.biostatecountdown = self.biostatecountdown - 1
            if self.biostatecountdown <= 0: # new state
                self.transitionAgent() 
                
    # getter for the number of states
    @classmethod
    def numberStates(cls):
        return cls.__states
    
    # getter for the trnasition matrix
    @classmethod
    def transitionMatrix(cls):
        return cls.bioTransition
    
    # getter for the bioMin, the array of minimal time of each state
    @classmethod
    def minimalTime(cls):
        return cls.bioMin
    
    # getter for the bioMax, the array of maximal time of each state
    @classmethod
    def maximalTime(cls):
        return cls.bioMax
    
    # for print()    
    def __str__(self):
        return f"(biostate = {self.biostate}, age = {self.age}, nextbiostate = {self.nextbiostate}, biostatecountdown = {self.biostatecountdown})"
    



# We will assume a flat organization where everyone has an equal chance of 
# interacting with everyone else.

# simulation with a basic social model using the class 'Agent'
def simulation_basic(numAgents, naturalImmunity, numInteractions, numDays, contagionProb, numInfected):
    """ 
    This function simulates the model(using ABM) with the basic social model(Anyone can interact with anyone)
    according to the parameters.
    
    Parameters
    ----------
    numAgents : int, the number of agents
    naturalImmunity : float, the proportion of naturally immuned people
    numInteractions : int, how many interactions per day per agent on average
    numDays : int, the number of days
    contagionProb : float,  normal contagion probability
    numInfected : int, the number of infected at the starting point

    Returns
    -------
    disthistory : numpy matrix, row = each day, col = each state, element = the number of agents
    """
    index_agents = np.arange(numAgents)
    disthistory = np.empty([numDays, Agent.numberStates()]) # nrow=numDays, ncol=states
    disthistory[:] = np.nan

    # List of agents' objects
    pool = []
    index_natural_immuned = []
    for i in index_agents:
        pool.append(Agent(biostate=np.random.choice(a = np.array([0,5]), size = 1, p = np.array([1-naturalImmunity, naturalImmunity]))[0]))
        if pool[i].biostate == 5:
            index_natural_immuned.append(i)

    # infect patients 0
    # eliminate the natural immuned agents from the entire agents to choose the infected agents
    index_not_natural_immuned = np.setdiff1d(index_agents, index_natural_immuned)

    index_infected_agents = np.random.choice(a = index_not_natural_immuned, size = 3, replace=False)
    for i in index_infected_agents:
        pool[i].setAgentState(1) # infect this person


    for day in range(numDays):
        # Who sneezes
        who_sneezes = np.repeat(np.arange(numAgents), numInteractions)
        # the people who are net to sneezer
        person_next_to_sneezer = np.random.choice(index_agents, replace=True, size=numAgents*numInteractions)

        # for every situation
        for i in range(np.size(who_sneezes)): 
            agent1 = pool[who_sneezes[i]] # who sneezes
            agent2 = pool[person_next_to_sneezer[i]] # who interacts with agent1(got sneeze form agent1)

            if((agent1.biostate==1 or agent1.biostate==2 ) and agent2.biostate==0 and np.random.uniform(low=0, high=1, size=1)[0]<contagionProb):
                agent2.setAgentState(1) ## infected!

        ## update Agents each day
        for i in index_agents:
            pool[i].updateagent()

        # update the matrix of disthistory
        biostates_of_agents = np.zeros(numAgents)
        for i in index_agents:
            biostates_of_agents[i] = pool[i].biostate
        unique, frequency = np.unique(biostates_of_agents, return_counts=True)

        distrib = np.zeros(Agent.numberStates())
        for i in range (Agent.numberStates()):
            for j in range(np.size(unique)):
                if i == unique[j]:
                    distrib[i] = frequency[j]

        disthistory[day,:] = distrib
        
    return disthistory



def socialNetwork(nNodes, probability_for_edges):
    """ 
    Implementation of Erdos-Renyi Model on a Social Network
    
    Parameters
    ----------
    nNodes : int, the number of Nodes
    probability_for_edges : float, the probability value for edges

    Returns
    -------
    g : graph object of the social network
    
    source: https://www.geeksforgeeks.org/implementation-of-erdos-renyi-model-on-social-networks/
    """
    # Create an empty graph object
    g = nx.Graph()
    
    # Adding nodes
    g.add_nodes_from(range(1, nNodes + 1))
    
    # Add edges to the graph randomly
    for i in g.nodes():
        for j in g.nodes():
            if(i < j):
                # Take ranom number R.
                R = np.random.random()
                
                # Check if R < probability_for_edges add the edge to the graph else ignore
                if (R < probability_for_edges):
                    g.add_edge(i,j)
    return g
    
    
def simulation_social_network(numAgents, naturalImmunity, numInteractions, numDays, contagionProb, numInfected, socialNetwork, sampleFromNetwork):
    """ 
    This function simulates the model(using ABM) with the social network, 
    which we create using the function 'socialNetwork()',
    according to the parameters.
    
    Parameters
    ----------
    numAgents : int, the number of agents
    naturalImmunity : float, the proportinn of naturally immuned people
    numInteractions : int, how many interactions per day per agent on average
    numDays : int, the number of days
    contagionProb : float,  normal contagion probability
    numInfected : int, the number of infected at the starting point
    socialNetwork : graph, the social network, which we created using the function 'socialNetwork()'
    sampleFromNetwork : float, how much this social network effects on whom an agent meet

    Returns
    -------
    disthistory : numpy matrix, row = each day, col = each state, element = the number of agents
    """
    
    index_agents = np.arange(numAgents)
    disthistory = np.empty([numDays, Agent.numberStates()]) # nrow=numDays, ncol=states
    disthistory[:] = np.nan
    # get the adjacency matrix of the graph of social network
    adj_matrix = nx.adjacency_matrix(socialNetwork)

    # List of agents' objects
    pool = []
    index_natural_immuned = []
    for i in index_agents:
        pool.append(Agent(biostate=np.random.choice(a = np.array([0,5]), size = 1, p = np.array([1-naturalImmunity, naturalImmunity]))[0]))
        if pool[i].biostate == 5:
            index_natural_immuned.append(i)

    # infect patients 0
    # eliminate the natural immuned agents from the entire agents to choose the infected agents
    index_not_natural_immuned = np.setdiff1d(index_agents, index_natural_immuned)


    index_infected_agents = np.random.choice(a = index_not_natural_immuned, size = 3, replace=False)
    for i in index_infected_agents:
        pool[i].setAgentState(1) # infect this person


    for day in range(numDays):
        # Who sneezes
        who_sneezes = np.repeat(np.arange(numAgents), numInteractions)
        # the people who are net to sneezer
        person_next_to_sneezer = np.zeros(who_sneezes.size, dtype = int)

        # sneezers meet whom?
        for i in range(np.size(who_sneezes)): 
            
            # when we get the probability of lower than (1-sampleFromNetwork) -> the sneezer meets a random person
            if(np.random.uniform(low=0, high=1, size=1)[0]<(1-sampleFromNetwork)):
                person_next_to_sneezer[i] = np.random.choice(index_agents, size=1)[0]
                
            # when we get the probability of greater(or as same as) than (1-sampleFromNetwork) -> the sneezer meets a person, who is in his/her social connections
            else:
                # find the index of Agent who sneezes
                j = math.floor(i/numInteractions)
                adj_matrix_to_array = adj_matrix[j,].toarray()[0]
                
                # if sneezer has no social network, he/she meets a random person
                if(adj_matrix_to_array.sum() == 0):
                    person_next_to_sneezer[i] = np.random.choice(index_agents, size=1)[0]
                    
                # if sneezer has a social network...    
                else:
                    # probability of meeting each agent. Assumption: If some has connection with n people, the probability of meeting each n person is same
                    probability = adj_matrix_to_array / adj_matrix_to_array.sum() #normalize
                    person_next_to_sneezer[i] = np.random.choice(a = np.arange(numAgents), size = 1, p= probability)[0]

        # for every situation
        for i in range(np.size(who_sneezes)): 
            agent1 = pool[who_sneezes[i]] # who sneezes!!!
            agent2 = pool[person_next_to_sneezer[i]] # who interacts with agent1(got sneeze form agent1)
            

            if((agent1.biostate==1 or agent1.biostate==2 ) and agent2.biostate==0 and np.random.uniform(low=0, high=1, size=1)[0]<contagionProb):
                agent2.setAgentState(1) ## infected!

        ## update Agents each day
        for i in index_agents:
            pool[i].updateagent()

        # update the matrix of disthistory
        biostates_of_agents = np.zeros(numAgents)
        for i in index_agents:
            biostates_of_agents[i] = pool[i].biostate
        unique, frequency = np.unique(biostates_of_agents, return_counts=True)
       
        distrib = np.zeros(Agent.numberStates())
        for i in range (Agent.numberStates()):
            for j in range(np.size(unique)):
                if i == unique[j]:
                    distrib[i] = frequency[j]

        disthistory[day,:] = distrib
        
    return disthistory


def simulation_social_network_quarantine(numAgents, naturalImmunity, numInteractions, numDays, contagionProb, numInfected, socialNetwork, sampleFromNetwork):
    """ 
    This function simulates the model(using ABM) with the social network, 
    which we create using the function 'socialNetwork()',
    according to the parameters.
    
    In this model, the people who have symptoms cannot meet other people
    
    Parameters
    ----------
    numAgents : int, the number of agents
    naturalImmunity : float, the proportinn of naturally immuned people
    numInteractions : int, how many interactions per day per agent on average
    numDays : int, the number of days
    contagionProb : float,  normal contagion probability
    numInfected : int, the number of infected at the starting point
    socialNetwork : graph, the social network, which we created using the function 'socialNetwork()'
    sampleFromNetwork : float, how much this social network effects on whom an agent meet

    Returns
    -------
    disthistory : numpy matrix, row = each day, col = each state, element = the number of agents
    """
    
    index_agents = np.arange(numAgents)
    disthistory = np.empty([numDays, Agent.numberStates()]) # nrow=numDays, ncol=states
    disthistory[:] = np.nan
    # get the adjacency matrix of the graph of social network
    adj_matrix = nx.adjacency_matrix(socialNetwork)

    # List of agents' objects
    pool = []
    index_natural_immuned = []
    for i in index_agents:
        pool.append(Agent(biostate=np.random.choice(a = np.array([0,5]), size = 1, p = np.array([1-naturalImmunity, naturalImmunity]))[0]))
        if pool[i].biostate == 5:
            index_natural_immuned.append(i)


    # infect patients 0
    # eliminate the natural immuned agents from the entire agents to choose the infected agents
    index_not_natural_immuned = np.setdiff1d(index_agents, index_natural_immuned)

    index_infected_agents = np.random.choice(a = index_not_natural_immuned, size = 3, replace=False)
    for i in index_infected_agents:
        pool[i].setAgentState(1) # infect this person

    for day in range(numDays):
        # Who sneezes
        who_sneezes = np.repeat(np.arange(numAgents), numInteractions)
        # the people who are net to sneezer
        person_next_to_sneezer = np.zeros(who_sneezes.size, dtype = int)

        # sneezers meet whom?
        for i in range(np.size(who_sneezes)): 
            
            # when we get the probability of lower than (1-sampleFromNetwork) -> the sneezer meets a random person
            if(np.random.uniform(low=0, high=1, size=1)[0]<(1-sampleFromNetwork)):
                person_next_to_sneezer[i] = np.random.choice(index_agents, size=1)[0]
                
            # when we get the probability of greater(or as same as) than (1-sampleFromNetwork) -> the sneezer meets a person, who is in his/her social connections
            else:
                # find the index of Agent who sneezes
                j = math.floor(i/numInteractions)
                adj_matrix_to_array = adj_matrix[j,].toarray()[0]
                
                # if sneezer has no social network, he/she meets a random person
                if(adj_matrix_to_array.sum() == 0):
                    person_next_to_sneezer[i] = np.random.choice(index_agents, size=1)[0]
                    
                # if sneezer has a social network...    
                else:
                    # probability of meeting each agent. Assumption: If some has connection with n people, the probability of meeting each n person is same
                    probability = adj_matrix_to_array / adj_matrix_to_array.sum() #normalize
                    person_next_to_sneezer[i] = np.random.choice(a = np.arange(numAgents), size = 1, p= probability)[0]
                    
            
        # for every situation
        for i in range(np.size(who_sneezes)): 
            agent1 = pool[who_sneezes[i]] # who sneezes!!!
            agent2 = pool[person_next_to_sneezer[i]] # who interacts with agent1(got sneeze form agent1)
            

            if((agent1.biostate==1) and agent2.biostate==0 and np.random.uniform(low=0, high=1, size=1)[0]<contagionProb):
                agent2.setAgentState(1) ## infected!

        ## update Agents each day
        for i in index_agents:
            pool[i].updateagent()

        # update the matrix of disthistory
        biostates_of_agents = np.zeros(numAgents)
        for i in index_agents:
            biostates_of_agents[i] = pool[i].biostate
        unique, frequency = np.unique(biostates_of_agents, return_counts=True)
        
        distrib = np.zeros(Agent.numberStates())
        for i in range (Agent.numberStates()):
            for j in range(np.size(unique)):
                if i == unique[j]:
                    distrib[i] = frequency[j]

        disthistory[day,:] = distrib
        
    return disthistory

def simulation_basic_reg_after7days(numAgents, naturalImmunity, numInteractions_before, numInteractions_after, numDays, contagionProb, numInfected):
    """ 
    This function simulates the model(using ABM) with the basic social model(Anyone can interact with anyone)
    according to the parameters.
    
    Parameters
    ----------
    numAgents : int, the number of agents
    naturalImmunity : float, the proportinn of naturally immuned people
    numInteractions_before : int, how many interactions per day per agent on average before the regulation by government
    numInteractions_after : int, how many interactions per day per agent on average after the regulation by government
    numDays : int, the number of days
    contagionProb : float,  normal contagion probability
    numInfected : int, the number of infected at the starting point

    Returns
    -------
    disthistory : numpy matrix, row = each day, col = each state, element = the number of agents
    """
    
    # the number of interactions at the starting point
    numInteractions = numInteractions_before
    index_agents = np.arange(numAgents)
    disthistory = np.empty([numDays, Agent.numberStates()]) # nrow=numDays, ncol=states
    disthistory[:] = np.nan

    # List of agents' objects
    pool = []
    index_natural_immuned = []
    for i in index_agents:
        pool.append(Agent(biostate=np.random.choice(a = np.array([0,5]), size = 1, p = np.array([1-naturalImmunity, naturalImmunity]))[0]))
        if pool[i].biostate == 5:
            index_natural_immuned.append(i)

    # infect patients 0
    # eliminate the natural immuned agents from the entire agents to choose the infected agents
    index_not_natural_immuned = np.setdiff1d(index_agents, index_natural_immuned)

    index_infected_agents = np.random.choice(a = index_not_natural_immuned, size = 3, replace=False)
    for i in index_infected_agents:
        pool[i].setAgentState(1) # infect this person

    for day in range(numDays):
        
        # after 14 days of the start date
        if (day >= 7):
            # the number of interactions changes
            numInteractions = numInteractions_after
            
        # Who sneezes
        who_sneezes = np.repeat(np.arange(numAgents), numInteractions)
        # the people who are net to sneezer
        person_next_to_sneezer = np.random.choice(index_agents, replace=True, size=numAgents*numInteractions)

        # for every situation
        for i in range(np.size(who_sneezes)): 
            agent1 = pool[who_sneezes[i]] # who sneezes
            agent2 = pool[person_next_to_sneezer[i]] # who interacts with agent1(got sneeze form agent1)

            if((agent1.biostate==1) and agent2.biostate==0 and np.random.uniform(low=0, high=1, size=1)[0]<contagionProb):
                agent2.setAgentState(1) ## infected!

        ## update Agents each day
        for i in index_agents:
            pool[i].updateagent()

        # update the matrix of disthistory
        biostates_of_agents = np.zeros(numAgents)
        for i in index_agents:
            biostates_of_agents[i] = pool[i].biostate
        unique, frequency = np.unique(biostates_of_agents, return_counts=True)
        
        distrib = np.zeros(Agent.numberStates())
        for i in range (Agent.numberStates()):
            for j in range(np.size(unique)):
                if i == unique[j]:
                    distrib[i] = frequency[j]

        disthistory[day,:] = distrib
    return disthistory

def simulation_basic_reg(numAgents, naturalImmunity, regulationStartday, numInteractions_before, numInteractions_after, numDays, contagionProb, numInfected):
    """ 
    This function simulates the model(using ABM) with the basic social model(Anyone can interact with anyone)
    according to the parameters.
    
    Parameters
    ----------
    numAgents : int, the number of agents
    naturalImmunity : float, the proportinn of naturally immuned people
    regulationStartday : int, when starts the regulation, after how many days?
    numInteractions_before : int, how many interactions per day per agent on average before the regulation by government
    numInteractions_after : int, how many interactions per day per agent on average after the regulation by government
    numDays : int, the number of days
    contagionProb : float,  normal contagion probability
    numInfected : int, the number of infected at the starting point

    Returns
    -------
    disthistory : numpy matrix, row = each day, col = each state, element = the number of agents
    """
    
    # the number of interactions at the starting point
    numInteractions = numInteractions_before
    index_agents = np.arange(numAgents)
    disthistory = np.empty([numDays, Agent.numberStates()]) # nrow=numDays, ncol=states
    disthistory[:] = np.nan

    # List of agents' objects
    pool = []
    index_natural_immuned = []
    for i in index_agents:
        pool.append(Agent(biostate=np.random.choice(a = np.array([0,5]), size = 1, p = np.array([1-naturalImmunity, naturalImmunity]))[0]))
        if pool[i].biostate == 5:
            index_natural_immuned.append(i)

    # infect patients 0
    # eliminate the natural immuned agents from the entire agents to choose the infected agents
    index_not_natural_immuned = np.setdiff1d(index_agents, index_natural_immuned)

    index_infected_agents = np.random.choice(a = index_not_natural_immuned, size = 3, replace=False)
    for i in index_infected_agents:
        pool[i].setAgentState(1) # infect this person

    for day in range(numDays):
        
        # after 14 days of the start date
        if (day >= regulationStartday):
            # the number of interactions changes
            numInteractions = numInteractions_after
            
        # Who sneezes
        who_sneezes = np.repeat(np.arange(numAgents), numInteractions)
        # the people who are net to sneezer
        person_next_to_sneezer = np.random.choice(index_agents, replace=True, size=numAgents*numInteractions)

        # for every situation
        for i in range(np.size(who_sneezes)): 
            agent1 = pool[who_sneezes[i]] # who sneezes
            agent2 = pool[person_next_to_sneezer[i]] # who interacts with agent1(got sneeze form agent1)

            if((agent1.biostate==1) and agent2.biostate==0 and np.random.uniform(low=0, high=1, size=1)[0]<contagionProb):
                agent2.setAgentState(1) ## infected!

        ## update Agents each day
        for i in index_agents:
            pool[i].updateagent()

        # update the matrix of disthistory
        biostates_of_agents = np.zeros(numAgents)
        for i in index_agents:
            biostates_of_agents[i] = pool[i].biostate
        unique, frequency = np.unique(biostates_of_agents, return_counts=True)
        
        distrib = np.zeros(Agent.numberStates())
        for i in range (Agent.numberStates()):
            for j in range(np.size(unique)):
                if i == unique[j]:
                    distrib[i] = frequency[j]

        disthistory[day,:] = distrib
    return disthistory