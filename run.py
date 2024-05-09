#!/usr/bin/env python3

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys

data=None
log_returns=None
countries=None
use_synthetic_data=False

def parse_data():
    global data, log_returns, countries

    if use_synthetic_data:
        df=pd.read_csv('synthetic_time_series.csv')
        data=df.values # (1000, 6)
        countries=list(df.columns)
        #print("Original data shape:", data.shape)
        log_returns = np.diff(np.log(data), axis=0) # (999, 6)

    else:
        df=pd.read_csv('stock_price_globe.csv', usecols=["Symbol", "Adj Close"])
        data=df["Adj Close"].values.reshape(814, 18, order='F') # (814, 18)
        countries=list(df.iloc[::814, 0].values)
        #print("Original data shape:", data.shape)
        log_returns = np.diff(np.log(data), axis=0) # (813, 18)


def q1():
    print("Log returns of the last 5 days for Taiwan:")
    print(log_returns[-5:,countries.index("Taiwan")])



def OMP(A, y, sparsity=np.infty):
    '''
    Parameters
    ----------
    A: matrix of shape (n, m)
    y: vector of shape (n,)

    Returns
    -------
    x: coefficient vector of shape (m,)
    _lambda: indices of the non-zero entries in x

    Reference
    ---------
    https://github.com/Adamdad/Orthogonal-Matching-Pursuit/blob/master/OMP.py
    '''
    assert A.ndim==2 and y.ndim==1
    r=y
    n,m=A.shape
    assert y.shape==(n,)
    x = np.zeros(m)
    _lambda = set()
    i=0
    while i<sparsity:
        index = np.argmax(abs(A.T.dot(r)))
        if index in _lambda:
            raise Exception("OMP failed")
        _lambda.add(index)

        basis = A[:,sorted(_lambda)]

        # least square solution for y=Ax: x'=(B^t B)^{-1} B^t y, where B=basis is the matrix whose columns are the chosen basis vectors
        # x = np.zeros(N) # unnecssary
        x[sorted(_lambda)] = np.linalg.inv(np.dot(basis.T, basis)).dot(basis.T).dot(y)

        # residual r = y - Bx' = y - Proj_{im(B)}(y)
        r = y - A.dot(x)
        i+=1
    return x, sorted(_lambda)


def draw_graph(vertices, edges, filename):
    G = nx.DiGraph()
    G.add_nodes_from(range(len(vertices)))
    G.add_edges_from(edges)

    plt.figure(figsize=(20, 20))
    ax = plt.gca()
    ax.set_aspect('equal')

    nx.draw_networkx(G, pos=nx.circular_layout(G), with_labels=True, labels={i: vertices[i] for i in range(len(vertices))}, node_color='skyblue', node_size=2000, edge_color='black', linewidths=1, font_size=20)

    plt.title('Granger Causality Graph', fontsize=20)
    plt.axis("off")
    plt.savefig(filename, format='png')



def find_coeff_by_OMP(lag, sparsity, filename):
    L=lag
    n,m=log_returns.shape
    assert m==len(countries)
    lagged_data = np.empty((n - L, m * L))

    for j in range(m):
        for i in range(L):
            lagged_data[:, L*j+i] = log_returns[i:-L+i, j]

    edges = []

    for i in range(m):
        y= log_returns[L:, i]
        x, coeff_idx = OMP(lagged_data, y, sparsity=sparsity)
        assert x.shape==(m*L,) and len(coeff_idx)<=m*L
        print(coeff_idx)

        for j in coeff_idx:
            assert x[j]!=0

        related = set([j//L for j in coeff_idx])

        edges.extend([(j,i) for j in related])
    draw_graph(countries, edges, filename)

def q2():
    find_coeff_by_OMP(lag=5, sparsity=1, filename="img/graph-Q2.png")

def q3():
    find_coeff_by_OMP(lag=20, sparsity=1, filename="img/graph-Q3.png")

def q4():
    for i in [1,2,3]:
        find_coeff_by_OMP(lag=20, sparsity=i, filename=f"img/graph-Q4-{i}.png")


def print_help():
    print("Usage: python run.py <question_number>")
    sys.exit(1)


if __name__=='__main__':
    if len(sys.argv) == 3 and sys.argv[2]=="test":
        use_synthetic_data = True
    elif len(sys.argv) == 2:
        pass
    else:
        print_help()

    try:
        arg = int(sys.argv[1])
    except ValueError:
        print("Error: Question number must be an integer.")
        print_help()

    parse_data()

    match arg:
        case 1:
            q1()
        case 2:
            q2()
        case 3:
            q3()
        case 4:
            q4()
