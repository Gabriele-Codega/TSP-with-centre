import numpy as np
from scipy.spatial import distance

# distances
def d2(a,b):
    return np.sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]))


# edge distances
def dm(a,b,**kwargs):
    """
    Midpoint distance.
    Center is assumed at (0.5,0.5)


    Parameters
    ----------
    a : 1D array
        coordinates of the first node
    b : 1D array
        coordinates of the second node

    Returns
    -------
    scalar
        midpoint distance from the centre
    """
    if (a == b).all():
        return np.sqrt((a[0]-0.5)*(a[0]-0.5) + (a[1]-0.5)*(a[1]-0.5))
    return 0.5*np.sqrt((a[0]+b[0]-1)*(a[0]+b[0]-1) + (a[1]+b[1]-1)*(a[1]+b[1]-1))

def dn(a,b,**kwargs):
    """
    Nearest point distance.
    Center is assumed at (0.5,0.5)


    Parameters
    ----------
    a : 1D array
        coordinates of the first node
    b : 1D array
        coordinates of the second node

    Returns
    -------
    scalar
        nearest distance from the centre
    """
    if (a == b).all():
        return np.sqrt((a[0]-0.5)*(a[0]-0.5) + (a[1]-0.5)*(a[1]-0.5))
    c = np.array([0.5,0.5])
    if ((c-a).dot(b-a) > 0) and ((c-b).dot(a-b) > 0):
        if a[1] != b[1]:
            m = (b[1]-a[1])/(b[0]-a[0])
            x = (0.5+0.5/m - a[1] + m*a[0])/(m+1/m)
            y = a[1] + m * (x-a[0])
        else:
            x = (a[0]+b[0])/2
            y = a[1]
        return np.sqrt((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5))
    return np.min([np.linalg.norm(c-a),np.linalg.norm(c-b)])

def da(a,b,p = 51,**kwargs):
    """
    Average distance.
    Center is assumed at (0.5,0.5)

    Parameters
    ----------
    a : 1D array
        coordinates of the first node
    b : 1D array
        coordinates of the second node
    p : int, optional
        number of sample points

    Returns
    -------
    scalar
        average distance from the centre
    """
    if (a == b).all():
        return np.sqrt((a[0]-0.5)*(a[0]-0.5) + (a[1]-0.5)*(a[1]-0.5))
    x = np.array([k/p * a[0] + (1-k/p)* b[0] for k in range(1,p)])
    y = np.array([k/p * a[1] + (1-k/p)* b[1] for k in range(1,p)])

    return np.mean(np.sqrt((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5)))



def t_path_dist(a,b,t):
    """
    Distance of a triangular path from the center.

    Parameters
    ----------
    a : 1D array
        coordinates of the first node
    b : 1D array
        coordinates of the second node
    t : 1D array
        coordinates of the triangular node
    Returns
    -------
    float
        distance of tpath from the centre
    """
    if np.allclose(a,t) or np.allclose(b,t):
        return da(a,b,p = 81)
    
    la = d2(a,t)
    lb = d2(b,t)
    
    ## weighted average
    # return (da(a,t,p = 81)*la + da(b,t,p = 81)*lb)/(la+lb)

    ## arithmetic average
    return (da(a,t,p=81)+da(b,t,p=81))*0.5



# energies
def E(L,C,r = 1,**kwargs):
    """
    E energy function:

    Parameters
    ----------
    L : scalar | array
        length
    C : scalar | array
        distance from center
    r : float, optional
        multiplicative factor for C, by default 1

    Returns
    -------
    scalar | array
        energy
    """
    return L+r*C

def E_star(L,C,omega = 0.5,**kwargs):
    """
    E* energy function:

    Parameters
    ----------
    L : scalar | array
        length
    C : scalar | array
        distance from center
    omega : float, optional
        must be in [0,1], by default 0.5. Coefficient of C

    Returns
    -------
    scalar | array
        energy
    """
    return (1-omega) * L + omega*C

def E_p(L,C,r=1,**kwargs):
    """
    E' energy function:

    Parameters
    ----------
    L : scalar | array
        length
    C : scalar | array
        distance from center
    r : float, optional
        multiplicative factor for LC, by default 1

    Returns
    -------
    scalar | array
        energy
    """
    return L+r*L*C

def E_pp(L,C,r=1,x=2,**kwargs):
    """
    E'' energy function:

    Parameters
    ----------
    L : scalar | array
        length
    C : scalar | array
        distance from center
    r : float, optional
        multiplicative factor for LC, by default 1
    x : float, optional
        additive factor for L in log(L+x), by default 2

    Returns
    -------
    scalar | array
        energy
    """
    return L + r*np.log(L + x) * C


# utility 
def find_t_nodes(a,b,K,m = 2500):
    """
    Find K triangular nodes for the pair (a,b).

    Parameters
    ----------
    a : 1D array
        coordinates of the first node
    b : 1D array
        coordinates of the second node
    K : int
        number of triangular nodes to be found
    m : int, optional
        number of samples on the circumference, by default 2500

    Returns
    -------
    t_nodes: array of shape (K,3)
        array with (x,y,l) for each triangular node
    """
    t_nodes = np.zeros((K,3))
    theta = np.linspace(0,2*np.pi,m)
    d = dn(a,b)
    for k in range(K):
        R = (k+1)*d/K
        if np.abs(R-d)<= 1e-8:
            t_nodes[k] = [a[0],a[1],d2(a,b)]
        else:
            pts = np.array([np.cos(theta),np.sin(theta)]).T * R + np.array([0.5,0.5])
            #dst = np.array([d2(a,p) + d2(b,p) for p in pts])
            dsts = distance.cdist(np.array([a,b]),pts,metric = 'euclidean')
            dst = np.sum(dsts,axis=0)
            #dst = distance.cdist(a.reshape((1,2)),pts,metric = 'euclidean') + distance.cdist(b.reshape((1,2)),pts,metric = 'euclidean')
            closest = np.argmin(dst)
            t_nodes[k,:] = [pts[closest,0],pts[closest,1], dst[closest]]
    return t_nodes