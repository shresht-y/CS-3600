import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
#  pgmpy͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy
#You are not allowed to use following set of modules from 'pgmpy' Library.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
#͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
# pgmpy.sampling.*͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
# pgmpy.factors.*͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
# pgmpy.estimators.*͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃

def make_security_system_net():
    """Create a Bayes Net representation of the above security system problem. 
    Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
    "D"'. (for the tests to work.)
    """
    BayesNet = BayesianModel()
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃     
    
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
    # TODO: set the probability distribution for each node͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
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
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['D'], joint=False)
    prob = marginal_prob['D'].values
    double0_prob = prob[1]
    return double0_prob


def get_conditional_double0_given_no_contra(bayes_net):
    """Calculate the conditional probability that Double-0 gets compromised
    given Contra is shut down.
    """
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['D'],evidence={'C':0}, joint=False)
    prob = conditional_prob['D'].values
    double0_prob = prob[1]    
    return double0_prob


def get_conditional_double0_given_no_contra_and_bond_guarding(bayes_net):
    """Calculate the conditional probability that Double-0 gets compromised
    given Contra is shut down and Bond is reassigned to protect M.
    """
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['D'],evidence={'B':1,'C':0}, joint=False)
    prob = conditional_prob['D'].values
    double0_prob = prob[1]
    return double0_prob


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel()
    # TODO: fill this out͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
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


def getA(bayes_net, evidence):
    """
    ***DO NOT POST OR SHARE THESE FUNCTIONS WITH ANYONE***
    Returns a distribution of probability of skill levels for team "A" given an evidence vector.
    Parameter: 
    : bayes net: Baysian Model Object
    : evidence: array of length 6 containing the skill levels for teams A, B, C in indices 0, 1, 2
    : and match outcome for AvB, BvC and CvA in indices 3, 4 and 5
    """
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values
    a = []
    normalizer = 0
    for i in range(4):
        normalizer += (team_table[i] * match_table[evidence[3]][i][evidence[1]] * 
                       match_table[evidence[5]][i][evidence[2]])
    for i in range(4):
        unnorm_prob = (team_table[i] * match_table[evidence[3]][i][evidence[1]] * 
                       match_table[evidence[5]][i][evidence[2]])
        a.append(unnorm_prob)
    return numpy.array(a)/normalizer


def getBvC(bayes_net, B, C):
    """
    ***DO NOT POST OR SHARE THESE FUNCTIONS WITH ANYONE***
    Returns a distribution of probability of match outcomes for "BvC" given the skill levels of B and C as evidence
    Parameter: 
    : bayes net: Baysian Model Object
    : B: int representing team B's skill level
    : C: int representing team C's skill level
    """
    BvC_cpd = bayes_net.get_cpds('BvC')
    match_table = BvC_cpd.values
    bvc = []
    for i in range(0, 3):
        bvc.append(match_table[i][B][C])
    return bvc   


def getB(bayes_net, evidence):
    """
    ***DO NOT POST OR SHARE THESE FUNCTIONS WITH ANYONE***
    Returns a distribution of probability of skill levels for team "B" given an evidence vector.
    Parameter: 
    : bayes net: Baysian Model Object
    : evidence: array of length 6 containing the skill levels for teams A, B, C in indices 0, 1, 2
    : and match outcome for AvB, BvC and CvA in indices 3, 4 and 5
    """
    B_cpd = bayes_net.get_cpds("B")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = B_cpd.values
    b = []
    normalizer = 0
    for i in range(4):
        normalizer += (team_table[i] * match_table[evidence[3]][evidence[0]][i] * 
                       match_table[evidence[4]][i][evidence[2]])
    for i in range(4):
        unnorm_prob = (team_table[i] * match_table[evidence[3]][evidence[0]][i] * 
                       match_table[evidence[4]][i][evidence[2]])
        b.append(unnorm_prob)
    return numpy.array(b)/normalizer


def getC(bayes_net, evidence):
    """
    ***DO NOT POST OR SHARE THESE FUNCTIONS WITH ANYONE***
    Returns a distribution of probability of skill levels for team "C" given an evidence vector.
    Parameter: 
    : bayes net: Baysian Model Object
    : evidence: array of length 6 containing the skill levels for teams A, B, C in indices 0, 1, 2
    : and match outcome for AvB, BvC and CvA in indices 3, 4 and 5
    """
    C_cpd = bayes_net.get_cpds("C")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = C_cpd.values
    c = []
    normalizer = 0
    for i in range(4):
        normalizer += (team_table[i] * match_table[evidence[5]][i][evidence[0]] * 
                       match_table[evidence[4]][evidence[1]][i])
    for i in range(4):
        unnorm_prob = (team_table[i] * match_table[evidence[5]][i][evidence[0]] * 
                       match_table[evidence[4]][evidence[1]][i])
        c.append(unnorm_prob)
    return numpy.array(c)/normalizer

def calculateMH(bayes_net, evidence):
    """
    ***DO NOT POST OR SHARE THESE FUNCTIONS WITH ANYONE***
    Returns the probability of a state.
    Parameter: 
    : bayes net: Baysian Model Object
    : evidence: array of length 6 containing the skill levels for teams A, B, C in indices 0, 1, 2
    : and match outcome for AvB, BvC and CvA in indices 3, 4 and 5
    """
    AvB_cpd = bayes_net.get_cpds('AvB').values
    BvC_cpd = bayes_net.get_cpds('BvC').values
    CvA_cpd = bayes_net.get_cpds('CvA').values
    skill_dist = [0.15, 0.45, 0.30, 0.10]
    A_skill_prob = skill_dist[evidence[0]]
    B_skill_prob = skill_dist[evidence[1]]
    C_skill_prob = skill_dist[evidence[2]]
    AvB_outcome_prob = AvB_cpd[evidence[3]][evidence[0]][evidence[1]]
    BvC_outcome_prob = BvC_cpd[evidence[4]][evidence[1]][evidence[2]]
    CvA_outcome_prob = CvA_cpd[evidence[5]][evidence[2]][evidence[0]]
    
    
    return (A_skill_prob * B_skill_prob * C_skill_prob * AvB_outcome_prob * 
            BvC_outcome_prob * CvA_outcome_prob)


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    sample = initial_state
    
    #print(sample)
    #pick a node to change 
    node = numpy.random.randint(4) 
    skillValues = [0,1,2,3]
    resultValues = [0,1,2]
    #node = 5
    if node == 0:
        #print("0")
        #case for A
        output = getA(bayes_net, [0, sample[1], sample[2], sample[3], sample[4], sample[5]])
        #print(output)
        newValue = numpy.random.choice(skillValues, p=output)
        #print(output[getprob])
        sample[0] = newValue
        #print(sample)
    elif node == 1:
        #print("1")
        #case for B
        output = getB(bayes_net, [sample[0], 0, sample[2], sample[3], sample[4], sample[5]])
        newValue = numpy.random.choice(skillValues, p=output)
        #print(output[getprob])
        sample[1] = newValue
        #print(sample)
    elif node == 2:
        #rint("2")
        #case for C
        output = getC(bayes_net, [sample[0], sample[1], 0, sample[3], sample[4], sample[5]])
        newValue = numpy.random.choice(skillValues, p=output)
        #print(output[getprob])
        sample[2] = newValue
        #print(sample)
    elif node == 3:
        #print("3")
        output = getBvC(bayes_net, sample[1], sample[2])
        newValue = numpy.random.choice(resultValues, p=output)
        sample[4] = newValue
    
    #print(sample)
    return tuple(sample)


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    #A_cpd = bayes_net.get_cpds("A")      
    #AvB_cpd = bayes_net.get_cpds("AvB")
    #match_table = AvB_cpd.values
    #team_table = A_cpd.values
    sample = [0,0,0,initial_state[3],0,initial_state[5]]   
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
    #print(sample)
    skillValues = [0,1,2,3]
    resultValues = [0,1,2]
    
    #Variable A change
    output = getA(bayes_net, [0, sample[1], sample[2], sample[3], sample[4], sample[5]])
    newValue = numpy.random.choice(skillValues, p=output)
    sample[0] = newValue
    
    #Variable B change
    output = getB(bayes_net, initial_state)
    newValue = numpy.random.choice(skillValues, p=output)
    sample[1] = newValue
    
    #Variable C change 
    output = getC(bayes_net, initial_state)
    newValue = numpy.random.choice(skillValues, p=output)
    sample[2] = newValue
    
    #variable BvC change 
    output = getBvC(bayes_net, sample[1], sample[2])
    newValue = numpy.random.choice(resultValues, p=output)
    sample[4] = newValue
    
    newProb = calculateMH(bayes_net, sample)
    oldProb = calculateMH(bayes_net, initial_state)
    #print("New value: ", calculateMH(bayes_net, sample))
    #print("old Value: ", calculateMH(bayes_net, initial_state))
    
    if (newProb > oldProb):
        #print("new value")
        
        return tuple(sample)
    else:
        #print("Old values")
        return tuple(initial_state)
    
    return "ERROR"


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
    N = 100
    delta = 0.000001
    
    #for Gibbs
    
    previous = initial_state
    i = 1
    while i < N:
        output = Gibbs_sampler(bayes_net, previous)
        probValue = calculateMH(bayes_net, output)
        previousProb = calculateMH(bayes_net, previous)
        if abs(probValue - previousProb) < delta:
            Gibbs_convergence = getBvC(bayes_net, output[1], output[2])
            Gibbs_count = i
            break
        previous = output
        i+=1
    
    #for MH
    previous = initial_state
    i = 1
    while i < N:
        output = MH_sampler(bayes_net, initial_state)
        probValue = calculateMH(bayes_net, output)
        previousProb = calculateMH(bayes_net, previous)
        if abs(probValue - previousProb) < delta:
            MH_convergence = getBvC(bayes_net, output[1], output[2])
            MH_count = i
            break
        if previous == output:
            MH_rejection_count +=1 
        previous = output
        i+=1
    
    
    
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
    
    choice = 0
    options = ['Gibbs','Metropolis-Hastings']
    factor = 1
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
    return "Shreshta Yadav"
