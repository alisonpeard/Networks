

import numpy as np
import pathpy as pp
import scipy as sp
from scipy.sparse import linalg as spl
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict # think these two are used to iterate through dicts
from collections.abc import Iterable

#---------------------------------------------------------------------------------------------------------
# useful things

def matprint(mat, fmt="g"):
    # just some useful code to displays matrices nicer for looking at this
    # https://gist.github.com/braingineer/d801735dac07ff3ac4d746e1f218ab75
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
  

def colorbar(mappable,label=None):
    '''adapted from https://joseph-long.com/writing/colorbars/'''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    cbar = fig.colorbar(mappable, cax=cax,orientation="horizontal",label=label)
    plt.sca(last_axes)
    return cbar
    
    
def every_nth_label(x,xticks,N=2):
    '''from a Stack Exchange : https://stackoverflow.com/questions/6682784/reducing-number-of-plot-ticks'''
    x = [j for i,j in enumerate(x) if not i%N]  # index of selected ticks
    xticks = [label for i,label in enumerate(xticks) if not i%N]
    return x,xticks
#---------------------------------------------------------------------------------------------------------
# directed network processing

def make_connected(net):
    '''not using: Adds one out-link to a randomly selected node to all hanging nodes'''
    nodes_list = list(net.nodes.keys())
    out_degrees = np.array(net.node_properties('outdegree'))
    ind = np.where(out_degrees == 0)[0]
    hanging_nodes = []
    for i in ind: hanging_nodes.append(list(net.nodes.keys())[i])
    for node in hanging_nodes : net.add_edge(node,np.random.choice(nodes_list))
    return net

def transition_mat(net,alpha=0.9,method="smart"):
    '''Modified from pathpy pagerank code, (not-transposed) transition matrix with teleportation
    this will slow down computation as matrices will no longer be sparse'''
    
    N = net.ncount()
    I = sp.sparse.identity(N)
    A = net.adjacency_matrix()

    # row sums are out-degrees for adjacency matrix
    row_sums = np.array(A.sum(axis=1)).flatten()

    # replace non-zero out-degree entries x by 1/x
    row_sums[row_sums != 0] = 1.0 / row_sums[row_sums != 0]

    # indices of zero entries in row_sums
    b = list(np.where(row_sums != 0)[0])
    d = list(np.where(row_sums == 0)[0])

    # create sparse matrix with inverted row_sums as diagonal elements
    Dinv = sp.sparse.spdiags(row_sums.T, 0, A.shape[0], A.shape[1],
                               format='csr')

    # with this, we have divided elements in non-zero rows in A by intverted row sums
    T = Dinv * A
    
    if method == "smart":
        # calculate preference vector using node in-strengths (Lambiotte & Rosvall, 2012)
        w_in = np.array(net.node_properties("inweight"))
        W = sum(w_in)
        v = w_in/W 
    elif method == "standard":
        v = np.ones(N)/N
    else:
        raise ValueError("Specify method as smart or standard teleportation")
    
    # replace nonzero rows with alpha*T + (1-alpha)*v
    for ib in b: T[ib,:] = alpha*T[ib,:] + (1-alpha)*v
        
    # replace all fully zero rows with v 
    for id in d: T[id,:] = v
    
    return T.todense()

def get_stationary(T):
    '''not using: will get the left eigenvector corresponding to eigenvalue 1 and return it normalised'''
    # np.linalg.eig finds right eigenvectors - adapted from code from a very helpful Stack Exchange
    # https://stackoverflow.com/questions/31791728/python-code-explanation-for-stationary-distribution-of-a-markov-chain?fbclid=IwAR0YnlQ7iwr1Ve1kRn-b6CDT0rbq7lDdBM1oD_KwiaEODWPv_-GMwiGBcVw
    evals, evecs = np.linalg.eig(T.T)
    evec1 = evecs[:,np.isclose(evals, 1)]
    evec1 = evec1[:,0]
    stationary = evec1 / evec1.sum()
    stationary = stationary.real
    stationary = np.squeeze(np.asarray(stationary))
    return stationary


def Laplacian_mat(net,alpha=0.9,method="smart"):
    '''my code for random walk-normalised Laplacian with smart recorded teleportation'''
    N = net.ncount()
    I = np.eye(N)
    T = transition_mat(net,method=method)
    
    Lrw = I - T 
    return Lrw

def my_pagerank(net,alpha=0.9,method = "smart"):
    '''Needs to be imporoved to be more efficient but does the job right now
    slightly modified.'''
    
    pr = defaultdict(lambda: 0)
    
    # adjacency matrix where Aij s.t. i->j
    I = sp.sparse.identity(net.ncount())
    A = net.adjacency_matrix()
    T = transition_mat(net,method=method)
    
    pr = get_stationary(T)
    
    pr = dict(zip(net.nodes, map(float, pr)))
    
    return pr

#---------------------------------------------------------------------------------------------------------
# functions similarity matrices

def row_normalise(mat,p=2):
    '''normalise each row of matrix matrix rows by its p-norm (yi/||yi||_2), default: p=2'''
    norms = np.linalg.norm(mat,p,axis=1) # axis=1 is norm along the rows when I do it manually and check
    norms_inv = [1/x if x!=0 else 0 for x in norms ]
    norm_mat = np.diag(norms_inv)
    return norm_mat @ mat

def col_normalise(mat,p=2):
    '''normalise each row of matrix matrix rows by its p-norm (yi/||yi||_2)'''
    norms = np.linalg.norm(mat,p,axis=0) # axis=1 is norm along the rows when I do it manually and check
    norms_inv = [1/x if x!=0 else 0 for x in norms ]
    norm_mat = np.diag(norms_inv)
    return mat @ norm_mat


def walktrap_similarity(net, n=None, alpha=0.85,option=2):
    '''Calculate Walktrap similarity using PageRank as the null-model for non-ergodic dynamics.
    option: whether to use transition matrix with teleportation (2) or just PageRank (1). '''
    if n==None:
        # choose a number of steps based on network diameter / number of vertices
        if pp.algorithms.shortest_paths.diameter(net) == np.inf:
            n = net.ncount()
        else:
            n = np.int(np.ceil((pp.algorithms.shortest_paths.diameter(net))))
            
    N = net.ncount()
    
    if option == 1:
        #option 1: using standard transition matrix where dangling nodes represented by rows of zeros (std. pathpy function)
        T = net.transition_matrix().todense().T # (T)ij : i->j
        pi_t = list(pp.algorithms.centralities.pagerank(net).values())
    elif option == 2:
        # option 2: transiton matrix with full teleportation - my code
        T = transition_mat(net,alpha=alpha)
        pi_t = np.array(list(my_pagerank(net,alpha=alpha).values()))
    else:
        raise ValueError("Please enter option=1 (full teleport) or option=2 (PageRank only)")


    # construct (not centered) null model for Walktrap using PageRank
    Pi = sp.sparse.diags(pi_t).todense()
  
    W = Pi
    Tn = T**n
    Tn = row_normalise(Tn @ np.sqrt(W),p=2)
    Psi_Wt = Tn @ Tn.T
    #Psi_Wt = (Tn @ W @ Tn.T)
    
    return Psi_Wt

def dynamical_similarity(net,t,weighted=True,option=2,alpha=None):
    '''CHECK TRANSPOSES: returns similarity matrix for dynamical similarity using weighting matrix W=Pi-pipi^T'''

    N = net.ncount()
    
    if option == 1:
        #option 1: using standard transition matrix where dangling nodes represented by rows of zeros (std. pathpy function)
        T = net.transition_matrix().todense().T # (T)ij : i->j
        pi_t = list(pp.algorithms.centralities.pagerank(net).values())
    elif option == 2:
        # option 2: transiton matrix with full teleportation 
        T = transition_mat(net,alpha=alpha)
        pi_t = np.array(list(my_pagerank(net,alpha=alpha).values()))
    else:
        raise ValueError("Please enter option=1 (full teleport) or option=2 (PageRank only)")
 
    I = np.eye(N,N)
    Lrw = I - T
    
    if weighted:
        Pi = sp.sparse.diags(pi_t).todense()
        Sigma_0 = np.abs(Pi - sp.sparse.csr_matrix(np.outer(pi_t,pi_t)).todense()) # justify taking abs
    else:
        print('unweighted!')
        Sigma_0 = np.eye(N,N)
    
    Y = spl.expm(-Lrw.T * t) #note
    Y = col_normalise(np.sqrt(Sigma_0) @ Y,p=2)
    Psi = Y.T @ Y
    #Psi = Y.T @ Sigma_0 @ Y
    
    return Psi

def spec_clust(net,algorithm="walktrap",k=2,time=10,option=2,alpha=None):
    '''k=number of clusters to identify'''
    from sklearn.cluster import SpectralClustering
    
    # define a similarity matrix to optimise over
    if algorithm == "walktrap":
        Psi = walktrap_similarity(net,n=time,option=option,alpha=alpha)
    elif algorithm == "dynamical similarity":
        Psi = dynamical_similarity(net,t=time,option=option,alpha=alpha)
    elif algorithm == "unweighted discrete":
        if option == 1:
            #option 1: using standard transition matrix where dangling nodes represented by rows of zeros (std. pathpy function)
            T = net.transition_matrix().todense().T # (T)ij : i->j
            pi_t = list(pp.algorithms.centralities.pagerank(net).values())
        elif option == 2:
            # option 2: transiton matrix with full teleportation - my code
            T = transition_mat(net,alpha=alpha)
            pi_t = np.array(list(my_pagerank(net,alpha=alpha).values()))
        else:
            raise ValueError("Please enter option=1 (full teleport) or option=2 (PageRank only)")
        Tn = T**time
        Tn = row_normalise(Tn,p=2) # optional
        Psi = (Tn) @ (Tn.T)
    else:
        raise ValueError("no algorithm specified!")

    clustering = SpectralClustering(n_clusters=k,assign_labels='discretize',random_state=0,affinity='precomputed').fit(Psi)
    
    # create clusters dictionary and updated
    group_dict1={}
    i=0
    for group in clustering.labels_:
        group_dict1.update({i:group})
        i = i+1
        
    group_dict2 = {}
    for key,ind in net.node_to_name_map().items():
        group_dict2.update({key: group_dict1[ind]})
        
    for node,group in group_dict2.items():
        net.nodes[node].update({'group':group})
        
    return net, group_dict2


def group_viz(net,group_dict,cols = "GnBu_r",magnify=5,label_size=8):
    
    norm =matplotlib.colors.Normalize(vmin=min(group_dict.values())-0.5, vmax=max(group_dict.values())+0.5)
    cmap =matplotlib.cm.get_cmap(cols)
        
    style={}
    style['node_color']={v:matplotlib.colors.rgb2hex(cmap(norm(u))) for v,u in group_dict.items()}
    style['node_size']={v:magnify for v,u in group_dict.items()}
    style['label_size'] ={v:label_size for v,u in group_dict.items()}
    style['edge_color']="gainsboro"
        
    pp.visualisation.plot(net, **style)
    

#---------------------------------------------------------------------------------------------------------
# core-periphery stuff

def get_node_strengths(net):
    strengths = np.array(net.node_properties('inweight')) + np.array(net.node_properties('outweight'))
    return strengths

def remove_hanging(net,method="cut"):
    '''Remove hanging nodes and returns network'''
    if method == "cut":
        out_degrees = np.array(net.node_properties('outweight'))
        
        if (out_degrees == 0).any():
            ind = np.where(out_degrees == 0)[0] 
            hanging_nodes = []
            for i in ind: hanging_nodes.append(list(net.nodes.keys())[i])
            for node in hanging_nodes : net.remove_node(node)
            
        return net
    
    else:
        return "no method defined for this"

    
def alpha_S(ind, T_t, ps_t):
    '''Calculate the persistence probability of a set of nodes S 
    ind : list of indices
    T_t : sparse transposed transition matrix
    ps_t = stationary probability column vector'''
    T_t = T_t[ind,:][:,ind]
    ps_t = ps_t[ind]
    return T_t.dot(ps_t).sum()/ps_t.sum()
    
    
def get_coreness(net,option=1):
    '''Calculates coreness of each node returns network with new 'coreness' node attributes.'''
    
    # check 
    A = net.adjacency_matrix(weighted=True)
    
    if option == 1:
        # pathpy gives you the tranpose of the transition matrix
        T_t = net.transition_matrix()
        ps = np.array(list(pp.algorithms.centralities.pagerank(net).values()))
        T = T_t.T
    elif option ==2:
        T = transition_mat(net) 
        ps = np.array(list(my_pagerank(net).values()))
    else : raise ValueError('Enter option for transition matrix: option=1 (no teleport) / option=1 (teleport)')
    
    N = net.ncount()
    
    node_strengths = get_node_strengths(net)
    
    # get indices for node names
    nodes_ind = np.arange(0,N)
    node_dict = {}
    coreness = {}
    i=0
    for node in net.nodes.keys():
        node_dict.update({i:node})
        coreness.update({node:0.0}) # initialise to zero
        i=i+1
    
    # calculate node strengths
    i_min = np.where(node_strengths == node_strengths.min())[0] 
    # randomly choose one if there's multiple mins
    i_min = np.random.choice(i_min,1)[0] 
    coreness[node_dict[i_min]] = coreness[node_dict[i_min]] + 0.0
    # node to initalise CP algorithm
    s0 = list(net.nodes.keys())[i_min] 

    # (greedy) algorithm to test which node will create smallest increase in alpha_S
    S = [i_min]
    alpha = alpha_S(S,T,ps) # alpha_S is zero for any single node
    nodes = np.delete(nodes_ind,i_min)

    while (len(nodes)>0):
        alpha_test = np.empty(N)
        alpha_test[:] = np.nan

        # calculate alpha increase for each node
        for node in nodes:
            S_test = S + [node]
            alpha_test[node] = alpha_S(S_test,T,ps) 

        # choose minimum alpha for this step
        node_min = np.where(alpha_test == np.nanmin(alpha_test))[0] 
        node_min = np.random.choice(node_min,1)[0]
        alpha = alpha + alpha_test[node_min]
        coreness[node_dict[node_min]] = coreness[node_dict[node_min]] + alpha_test[node_min]

        S = S + [node_min]
        nodes = nodes[nodes!=node_min]
    
    for node, val in coreness.items():
        net.nodes[node]['coreness']=val
    
    return net,coreness

    
def print_coreness(net,coreness,alpha_c=1e-3,core_col='darkorange',periph_col='lightskyblue',edge_col='gainsboro',magnify=1):
    '''https://github.com/IngoScholtes/csh2018-tutorial/blob/master/solutions/2_pathpy.ipynb'''
    # colour edges according to coreness of nodes they emanate from
    edge_coreness = {} 
    for edge in net.edges.keys():
        edge_coreness.update({edge:coreness[edge[0]]})
    
    style={}
    style['edge_color']={v:edge_col for v,u in edge_coreness.items()}
    style['node_color']={v:periph_col if u < alpha_c else core_col for v,u in coreness.items()}
    style['node_size']={v:5+magnify*u for v,u in coreness.items()}
    style['force_charge']={v: -100 if u<alpha_c else -20 for v,u in coreness.items()}
    style['node_text'] = {v:u for v,u in coreness.items()}
    pp.visualisation.plot(net, **style)
    
def html_coreness(net,coreness,alpha_c=1e-3,core_col='darkorange',periph_col='lightskyblue',edge_col='gainsboro',magnify=1):
    '''https://github.com/IngoScholtes/csh2018-tutorial/blob/master/solutions/2_pathpy.ipynb'''
    # colour edges according to coreness of nodes they emanate from
    edge_coreness = {} 
    for edge in net.edges.keys():
        edge_coreness.update({edge:coreness[edge[0]]})
    
    style={}
    style['edge_color']={v:edge_col for v,u in edge_coreness.items()}
    style['node_color']={v:periph_col if u < alpha_c else core_col for v,u in coreness.items()}
    style['node_size']={v:5+magnify*u for v,u in coreness.items()}
    style['force_charge']={v: -100 if u<alpha_c else -20 for v,u in coreness.items()}
    style['node_text'] = {v:u for v,u in coreness.items()}
    return pp.visualisation.html.generate_html(net, **style)


#---------------------------------------------------------------------------------------------------------
# make some model nets

def toy_net_1():
    #node_list = [1,2,3,4,5,6,7,8,9,10,11]
    #toy_net = pp.Network()
    #for node in node_list : toy_net.add_node(node)
    toy_net=pp.Network(directed=True)
    toy_net.add_edge("a","b",weight=2)
    toy_net.add_edge("b","c",weight=2)
    toy_net.add_edge("c","a",weight=2)

    toy_net.add_edge("c","d")
    toy_net.add_edge("d","e")
    toy_net.add_edge("e","c")

    toy_net.add_edge("b","h")
    toy_net.add_edge("h","i")
    toy_net.add_edge("i","b")

    toy_net.add_edge("a","f")
    toy_net.add_edge("f","g")
    toy_net.add_edge("g","a")

    return toy_net

def toy_net_2():
    node_list = [1,2,3,4,5,6,7,8,9,10,11,12]
    toy_net = pp.Network()
    for node in node_list : toy_net.add_node(node)
    toy_net=pp.Network(directed=True)
    toy_net.add_edge(1,4,weight=1)
    toy_net.add_edge(4,1,weight=5)
    toy_net.add_edge(1,3,weight=5)
    toy_net.add_edge(1,2,weight=1)
    toy_net.add_edge(2,4,weight=5)
    toy_net.add_edge(4,2,weight=1)
    toy_net.add_edge(2,3,weight=1)
    toy_net.add_edge(3,2,weight=5)
    toy_net.add_edge(3,1,weight=1)

    toy_net.add_edge(1,5,weight=0.5) #
    toy_net.add_edge(3,7,weight=0.5) #
    toy_net.add_edge(2,8,weight=0.5) #
    
    toy_net.add_edge(4,6,weight=0.5) #

    toy_net.add_edge(8,11,weight=1)
    toy_net.add_edge(11,8,weight=1)
    #toy_net.add_edge(8,10)
    #toy_net.add_edge(10,8)
    toy_net.add_edge(8,9,weight=1)
    toy_net.add_edge(9,8,weight=1)

    toy_net.add_edge(7,11,weight=1)
    toy_net.add_edge(11,7,weight=1)
    #toy_net.add_edge(7,9)
    #toy_net.add_edge(9,7)
    toy_net.add_edge(7,10,weight=1)
    toy_net.add_edge(10,7,weight=1)
    

    toy_net.add_edge(6,10,weight=1)
    toy_net.add_edge(10,6,weight=1)
    toy_net.add_edge(6,9,weight=1)
    toy_net.add_edge(9,6,weight=1)

    toy_net.add_edge(5,9,weight=1)
    toy_net.add_edge(9,5,weight=1)
    toy_net.add_edge(5,10,weight=1)
    toy_net.add_edge(10,5,weight=1)
    
    # hanging node
    toy_net.add_edge(11,12,weight=1)
    # fix order of nodes
    toy_net.nodes = {k: toy_net.nodes[k] for k in node_list}


    return toy_net


#---------------------------------------------------------------------------------------------------------
# unused functions
def quality_func(Psi,H):
    R = np.trace(H.T @ Psi @ H)
    return R


def Louvain(net,algorithm,time):
    '''Phase 1 of Louvain clustering, returns optimal Walktrap perturbation matrix for given number of timesteps'''
    
    # define similarity matrix to optimise over
    if algorithm == "walktrap":
        Psi = walktrap_similarity(net,n=time)
    elif algorithm == "dynamical similarity":
        Psi = dynamical_similarity(net,t=time)
    elif algorithm == "unweighted discrete":
        T = net.transition_matrix().todense().T
        Tn = T**time
        Tn = row_normalise(Tn,order=2)
        Psi = (Tn) @ (Tn.T)
    else:
        print("no algorithm specified!")
    
    #initial random module assignments
    N = net.ncount()
    H = np.eye(N, N) # identity will do - there just has to be one per row/col
    
    Hopt = np.eye(N,N)

    # until nodes stop swapping communities
    k = 1
    while k > 0:
        k = 0
        for i in range(0,N):        
                # locate current community assignment and get quality func for node i
                Hj = np.copy(Hopt)
                jopt = np.where(Hj[i,:]==1.0)[0][0]
                R = quality_func(Psi,Hj)

                # see if any other assignments give better quality function value
                for j in range(0,N):
                    # move node i to community j and calculate change in quality  function
                    ej = np.zeros(N)
                    ej[j] = 1.0

                    Hj[i,:] = ej 
                    Rj = quality_func(Psi,Hj)

                    if Rj > R:
                        #print('{}->{}; {} is bigger than {}'.format(jopt,j,Rj,R))
                        R = Rj
                        jopt = j
                        k = k+1 # record that a node has moved community
                        # now the ith row of H has changed
                        Hopt[i,:] = ej
                        
    return Hopt # permutation matrix

def group_network(net,timescale,algorithm):
    '''For Louvain only'''
    
    H = Louvain(net,algorithm=algorithm,time=timescale)
    
    group_dict1 = {} # record which nodes are in which group
    group_dict2 = {} # transform to index:group format
    group_dict3 = {} # change indices to node names


    i=0
    for col in H.T:
        if (col==1.0).any():
            group_dict1.update({i:list(np.where(col==1.0)[0])})
            i = i +1

    for group, nodes in group_dict1.items():
        group_dict2.update({node:group for node in nodes})

    for key,ind in net.node_to_name_map().items():
        group_dict3.update({key: group_dict2[ind]})

    for node,group in group_dict3.items():
                net.nodes[node].update({'group':group})
        
    return net, group_dict3
