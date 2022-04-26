import numpy as np
import h5py
from skimage.measure import block_reduce

def directed_adj():
    acc = [-1,36,49,61,79,105,116,151,162,200]
    
    base = np.identity(201,dtype=bool)
    
    for i in range(0,201):
        if i not in acc:
            base[i][i+1]=True
            
    base[36][37]=True
    base[36][80]=True
    base[49][50]=True
    base[116][50]=True
    base[105][106]=True
    base[105][117]=True
    base[61][62]=True
    base[61][152]=True
    base[162][163]=True
    base[151][163]=True
    base[200][0]=True
    base[79][0]=True
    
    both = np.logical_or(base, base.transpose())
    
    return base.astype(int), base.transpose().astype(int), both.astype(int)

def normalized_laplacian(adj):
    D = 1/adj.sum(axis=1)
    D = np.diag(D)
    return adj+D

def calculate_adjacency_k(L, k):
    dim = L.shape[0]
    if k > 0:
        output = L
        for i in range(k-1):
            output = np.matmul(output,L)
        out = np.sign(output)
    elif k == 0:
        out = np.eye(dim)
    return out

def z_score(x, mean, std):
    return (x - mean) / std

def z_inverse(x, mean, std):
    return x * std + mean

def DLdata(V, Q, obs, pred, mode='singlestep'):
    
    V = np.transpose(V, (0,2,1))
    Q = np.transpose(Q, (0,2,1))
    D, T, N = V.shape
    #print(D, T, N)
    V[V>130.] = 130.
    V = V/130.
    Q[Q>3000.] = 3000.
    Q = Q/3000.
    
    X = []
    Y = []
    
    for i in range(D):
        for j in range(obs+1, T-15-1):
            inp1 = V[i][j-obs:j]
            inp2 = Q[i][j-obs:j]
            inp = np.stack([inp1, inp2], axis=-1)
            if mode == 'singlestep':
                out = V[i][j+pred-1]
            else:
                out = V[i][j:j+pred]
            
            if np.amin(V[i][j-5:j+5]) < 1/3:
                X.append(inp)
                Y.append(out)
    X0 = np.array(X)
    Y0 = np.array(Y)
    
    print(X0.shape, Y0.shape)
    
    return X0, Y0

def data_generator(obs=15, pred=5, mode='singlestep'):
    
    Data = h5py.File('AMSnetM.h5', 'r')
    v = np.array(Data['V'])
    q = np.array(Data['Q'])
    v = block_reduce(v,block_size=(1,1,2),func=np.mean)
    q = block_reduce(q,block_size=(1,1,2),func=np.mean)
    
    X1, Y1 = DLdata(v,q, obs, pred, mode=mode)
    
    Data = h5py.File('AMSnetE.h5', 'r')
    v = np.array(Data['V'])
    q = np.array(Data['Q'])
    v = block_reduce(v,block_size=(1,1,2),func=np.mean)
    q = block_reduce(q,block_size=(1,1,2),func=np.mean)
    
    X2, Y2 = DLdata(v, q, obs, pred, mode=mode)
    
    X = np.concatenate([X1, X2], axis=0)
    Y = np.concatenate([Y1, Y2], axis=0)
    
    mean_v = np.mean(X[...,0])
    mean_q = np.mean(X[...,1])
    std_v = np.std(X[...,0])
    std_q = np.std(X[...,1])
    
    X[...,0] = z_score(X[...,0], mean_v, std_v)
    X[...,1] = z_score(X[...,1], mean_q, std_q)
    
    split = round(len(X)*0.8)
    
    X_train = X[:split]
    X_test = X[split:]
    Y_train = Y[:split]
    Y_test = Y[split:]
    
    return X_train, X_test, Y_train, Y_test, mean_v, std_v

def load_data(self):
    X, Xt, Y, Yt = data_generator(self.obs, self.pred, mode=self.step)
    print(Y.shape, np.amin(Y), np.amax(Y))
    print(Yt.shape, np.amin(Yt), np.amax(Yt))

    return X, Xt, Y, Yt