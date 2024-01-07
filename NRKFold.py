import random
import numpy as np
def NRKFold(E,pc,K = 5, shuffle=True):
    """
    Generate non-redundant K-folds for a dataset where each example involves an object 
    that belongs to a certain cluster. This function ensures that no two folds contain 
    objects from the same cluster and aims to distribute the number of examples 
    approximately equally across all folds.

    This is particularly useful in scenarios where data points can naturally group into 
    clusters (e.g., proteins in bioinformatics), and it is important to avoid having 
    similar examples in both the training and validation sets of a particular fold.

    Parameters
    ----------
    E : List
        A list containing identifiers of objects involved in each example.
    pc : Dictionary
        A dictionary mapping each object to its cluster assignment.
    K : Integer, optional
        The number of folds to create. Default is 5.
    shuffle : Boolean, optional
        Determines whether to shuffle the cluster to fold assignments in different runs.
        Default is True.

    Returns
    -------
    List of lists
        A list where each sublist contains the indices of examples in `objects` that belong to a particular fold.

    Example
    -------
    >>> objects = ['obj1', 'obj2', 'obj3', 'obj4', 'obj5', 'obj6', 'obj1']
    >>> clusters = {'obj1': 1, 'obj2': 2, 'obj3': 1, 'obj4': 2, 'obj5': 3, 'obj6': 3}
    >>> folds = NRKFold(objects, clusters, K=2, shuffle=False)
    >>> print(folds)
    Output might be: [[0, 2, 6], [1, 3, 4, 5]]
    Here, objects 'obj1', 'obj3', and 'obj1' (indices 0, 2, 6) are in one fold, 
    and the rest are in another fold, ensuring no fold has objects from the same cluster.
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
