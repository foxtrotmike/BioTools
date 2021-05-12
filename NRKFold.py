import random
import numpy as np
def NRKFold(E,pc,K = 5, shuffle=True):
    """
    Generate non-redundant K-folds so that no two folds contain proteins
    belonging to the same cluster and the number of examples are (approx)
    equal in all folds. Implements the greedy number partitioning method
    https://en.wikipedia.org/wiki/Greedy_number_partitioning
    >>> NRKFold(['p1','p2','p3','p4','p5','p6','p1'],{'p1':1,'p2':2,'p3':1,'p4':2,'p5':3,'p6':3},K=2, shuffle = False)
    Here we have 7 examples involving 6 proteins p1-p6 such that proteins
    p1,p3,p5 form one cluster whereas p2,p4,p6 form another cluster
    This results in division into two folds as [[0, 2, 6], [1, 3, 4, 5]]
    Note that examples 0,2,6 comprising fold-1 with proteins p1 and p2 are 
    from the first cluster whereas the remaining examples are from other clusters
    Parameters
    ----------
    E : TYPE List (length equal to number of examples)
        DESCRIPTION. Protein id of protein involved in each example
    pc : TYPE dictionary 
        DESCRIPTION. Cluster assignment of each protein
    K : TYPE, optional Integer
        DESCRIPTION. Number of folds The default is 5.
    shuffle: TYPE, Boolean
        cluster to fold assignments are different across different runs
    Returns
    -------
    F : TYPE list of lists
        DESCRIPTION. Indices of examples in E in each fold

    """
    e = [pc[str(x)] for x in E] #cluster indices of all proteins in the examples
    c2idx={} #indices of examples of each cluster in e
    for i,x in enumerate(e):
        try: 
            c2idx[x].append(i)
        except:
            c2idx[x]=[i]    
    ce = dict([(c,len(c2idx[c])) for c in c2idx]) #counts of examples of different clusters    
    cF = [0]*K; #counts of examples in each fold
    CF = [[] for _ in range(K)]; #clusters in each fold
    F = [[] for _ in range(K)];#indices of examples in each fold
    keys = list(ce.keys())
    if shuffle:
        random.shuffle(keys)
    for k in keys:
        v = ce[k]
        idx = np.argmin(cF)
        cF[idx]+=v
        CF[idx].append(k) #add cluster to fold
        F[idx].extend(c2idx[k])
    return F
