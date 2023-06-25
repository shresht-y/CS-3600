import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
#  pgmpy
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy
#You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

def make_security_system_net():
    """Create a Bayes Net representation of the above security system problem. 
    Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
    "D"'. (for the tests to work.)
    """
    BayesNet = BayesianModel()
    # TODO: finish this function    
    
    #add the nodes
    BayesNet.add_node("H")
    BayesNet.add_node("C")
    BayesNet.add_node("M")
    BayesNet.add_node("Q")
    BayesNet.add_node("B")
    BayesNet.add_node("K")
    BayesNet.add_node("D")
    #add edges 
    BayesNet.add_edge("H","Q")
    BayesNet.add_edge("C","Q")
    BayesNet.add_edge("M","K")
    BayesNet.add_edge("B","K")
    BayesNet.add_edge("Q","D")
    BayesNet.add_edge("K","D")
    
    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the security system.
    Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
    "D"'. (for the tests to work.)
    """
    # TODO: set the probability distribution for each node
    cpd_h = TabularCPD('H', 2, values=[[0.5], [0.5]])
    cpd_c = TabularCPD('C', 2, values=[[0.7], [0.3]])
    cpd_m = TabularCPD('M', 2, values=[[0.2], [0.8]])
    cpd_b = TabularCPD('B', 2, values=[[0.5], [0.5]])
    
    cpd_q = TabularCPD('Q', 2, values=[[0.95, 0.75, 0.45, 0.1], \
                    [0.05, 0.25, 0.55, 0.9]], evidence=['H', 'C'], evidence_card=[2, 2])
    
    cpd_k = TabularCPD('K', 2, values=[[0.25, 0.99, 0.05, 0.85], \
                    [0.75, 0.01, 0.95, 0.15]], evidence=['M', 'B'], evidence_card=[2, 2])
    
    cpd_d = TabularCPD('D', 2, values=[[0.98, 0.65, 0.4, 0.01], \
                    [0.02, 0.35, 0.6, 0.99]], evidence=['Q', 'K'], evidence_card=[2, 2])
    
    bayes_net.add_cpds(cpd_h, cpd_c, cpd_m, cpd_b, cpd_q, cpd_k, cpd_d)
    
    return bayes_net


def get_marginal_double0(bayes_net):
    """Calculate the marginal probability that Double-0 gets compromised.
    """
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['D'], joint=False)
    prob = marginal_prob['D'].values
    double0_prob = prob[1]
    return double0_prob


def get_conditional_double0_given_no_contra(bayes_net):
    """Calculate the conditional probability that Double-0 gets compromised
    given Contra is shut down.
    """
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['D'],evidence={'C':0}, joint=False)
    prob = conditional_prob['D'].values
    double0_prob = prob[1]
    return double0_prob


def get_conditional_double0_given_no_contra_and_bond_guarding(bayes_net):
    """Calculate the conditional probability that Double-0 gets compromised
    given Contra is shut down and Bond is reassigned to protect M.
    """
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['D'],evidence={'B':1,'C':0}, joint=False)
    prob = conditional_prob['D'].values
    double0_prob = prob[1]
    return double0_prob


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel()
    # TODO: fill this out
    BayesNet.add_node("A")
    BayesNet.add_node("B")
    BayesNet.add_node("C")
    BayesNet.add_node("AvB")
    BayesNet.add_node("CvA")
    BayesNet.add_node("BvC")
    
    BayesNet.add_edge("A", "AvB")
    BayesNet.add_edge("A", "CvA")
    BayesNet.add_edge("B", "AvB")
    BayesNet.add_edge("B", "BvC")
    BayesNet.add_edge("C", "CvA")
    BayesNet.add_edge("C", "BvC")
    
    #set table values
    
    cpd_a = TabularCPD('A', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_b = TabularCPD('B', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_c = TabularCPD('C', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    
    cpd_avb = TabularCPD('AvB', 3, values=[[0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1], \
                    [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.10], \
                        [0.8, 0.2, 0.10, 0.05, 0.2, 0.80, 0.2, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.1, 0.20, 0.80]], evidence=['A', 'B'], evidence_card=[4, 4])
    
    cpd_cva = TabularCPD('CvA', 3, values=[[0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1], \
                    [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.10], \
                        [0.8, 0.2, 0.10, 0.05, 0.2, 0.80, 0.2, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.1, 0.20, 0.80]], evidence=['A', 'C'], evidence_card=[4, 4])
    
    cpd_bvc = TabularCPD('BvC', 3, values=[[0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1], \
                    [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.10], \
                        [0.8, 0.2, 0.10, 0.05, 0.2, 0.80, 0.2, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.1, 0.20, 0.80]], evidence=['B', 'C'], evidence_card=[4, 4])
    
    BayesNet.add_cpds(cpd_a, cpd_b, cpd_c, cpd_avb, cpd_cva, cpd_bvc)
    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # TODO: finish this function    
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['BvC'],evidence={'AvB':0,'CvA':2}, joint=False)
    posterior = conditional_prob['BvC'].values
    
    return posterior # list 


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """    
    # TODO: finish this function
    if initial_state is None or len(initial_state) == 0:
        sample = [0,0,0,0,0,0]
        values = numpy.random.randint(4, size=3)
        sample[0] = values[0]
        sample[1] = values[1]
        sample[2] = values[2]
        
        matches = numpy.random.randint(3, size=3)
        sample[3] = matches[0]
        sample[4] = matches[1]
        sample[5] = matches[2]
        #print(sample)
        #return tuple(sample)
        sample = tuple(sample)
    else:
        sample = tuple(initial_state)  
    
    #pick node to change
    index = numpy.random.randint(6)
    node = sample[index]
    #print(node)
    
     
    return sample


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values
    sample = tuple(initial_state)    
    # TODO: finish this function
        
    
    return sample


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # TODO: finish this function
    raise NotImplementedError        
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    raise NotImplementedError
    choice = 2
    options = ['Gibbs','Metropolis-Hastings']
    factor = 0
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Shreshta Yadav"
