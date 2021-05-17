#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import operator

from scipy.optimize import curve_fit
import matplotlib.colors as mcolors


from tqdm import tqdm


class Nodes(object):
    """Nodes object.

        An object that stores the information about a network. The basic is only the pandas.DataFrame
        but in this way, it is more manageable. Access directly: Nodes.df. Whole work is based on 
        Papadopoulos' (et al.) article: https://www.nature.com/articles/nature11459.

        Call signature: 

            n           int, number of nodes
            m           int, number of connections when new node added
            T           float, temperature
            beta        float, [0,1], drift parameter (0: full drift, 1: no drift)

        
        example:
            
            import sys
            sys.path.append('../')
            from pvs import network_creation as nc

            Nodes = nc.Nodes()  

    """

    def __init__(self, n: int = 50, m: int = 5, T: float = 1., beta: float = 1.):
        super(Nodes, self).__init__()
        self.df = pd.DataFrame(columns = ['id', 'edges', 'r', 'theta'])
        self.n = n
        self.m = m
        self.T = T
        self.beta = beta

        for _ in tqdm(range(n)):
            self.add_node(m = self.m, T = self.T, beta = self.beta)

        self.update_df()

    def add_node(self, m: int = None, T: int = None, beta: float = None) -> pd.DataFrame:

        """
        Adding a node to existing network. Any given parameter will override the netowork's
        default during node creation, but not permamently. Default parameters are the 
        network's inner parameters. 

            m           int, number of connections when new node added
            T           float, temperature
            beta        float, [0,1], drift parameter (0: full drift, 1: no drift)

        Returns nodes in a pandas.DataFrame (and refreshes self.df, too). 
        """
        
        nodes_df = self.df
        m = self.m if m is None else m
        T = self.T if T is None else T
        beta = self.beta if beta is None else beta


        def create_connection(id1, id2):
            """Creating a connection between nodes with id1 and id2. Returns nothing.

            TODO: Check if connection exists.
            """
            node = nodes_df[nodes_df.id == id1]
            elist = node.edges.values[0]
            idx = node.index.values[0]
            elist.append(id2)
            nodes_df.at[idx, "edges"] = elist

            node = nodes_df[nodes_df.id == id2]
            elist = node.edges.values[0]
            idx = node.index.values[0]
            elist.append(id1)
            nodes_df.at[idx, "edges"] = elist

        new_id = len(nodes_df) + 1
        fst_params = [new_id, [], np.log(new_id), np.random.rand()*360]
        nodes_df = nodes_df.append({k:v for k, v in zip(nodes_df.columns, fst_params)}, ignore_index=True)
        
        if len(nodes_df)-1 <= m:
            for id_ in nodes_df.id[:-1]:
                create_connection(id_, new_id)
            self.df = nodes_df
            return nodes_df

        dists = {}
        
        newnode = nodes_df.loc[nodes_df.id == new_id]
        
        for index, node in nodes_df.iterrows():
            if node['id'] != new_id:
                dist = Nodes.calculate_dist(nodes_df, node["id"], new_id, beta = beta)
                dists[node["id"]] = dist
        sorted_dists = sorted(dists.items(), key=operator.itemgetter(1))
        
        m_put = 0
        put = []
        while m_put < m:
            for nodeid, dist in sorted_dists:
                r = np.random.rand()
                prob = 1/(1 + np.exp((dist-np.log(new_id))/T ))
                if r<= prob and nodeid not in put:
                    create_connection(nodeid, new_id)
                    m_put += 1
                    put.append(nodeid)
                else:
                    continue
                if m_put>=m:
                    break
        self.df = nodes_df
        return nodes_df

    def current_radius(nodeid: int, t: int, beta: float):
        """
        Calculating the current (drifted) radius of nodeid (so original radius is log(nodeid))
        at given t time having beta drifting parameter. Returns current radius as float.
        """
        return beta*np.log(nodeid) + (1-beta)*np.log(t)

    def calculate_dist(nodes_df: pd.DataFrame, nodeid: int, newNodeid: int, beta: float = 0) -> float:
        """
        Calculating logarithmic distances between different nodes. 
            
            nodes_df    pandas.DataFrame, usually self.df is okay
            nodeid      int, id of one node (id = timestep that it was attached to network)
            newNodeid   int, id of new node (id = current timestep)
            beta        float, [0,1], drift parameter (0: full drift, 1: no drift)
        
        Returns logarithmic distance as float.
        """

        node1, node2 = nodes_df.loc[nodes_df.id == nodeid], nodes_df.loc[nodes_df.id == newNodeid]
        rad1, rad2 = Nodes.current_radius(nodeid, newNodeid, beta = beta), np.log(newNodeid)
        return rad1 + rad2 + np.log(abs(float(node1.theta)-float(node2.theta))/2)

    def update_df(self) -> pd.DataFrame:
        """
        Updating DataFrame: calculating drifted distances and counting current edges for every node.
        Updates self.df and returns pandas.DataFrame.
        """
        nodes_df = self.df
        beta = self.beta
        t = len(nodes_df)
        current_rad = [ Nodes.current_radius(nodeid_, t, beta = beta) for nodeid_ in nodes_df.id]
        num_of_edges = [len(elist) for elist in nodes_df.edges]
        nodes_df["current_rad"] = current_rad
        nodes_df["num_of_edges"] = num_of_edges
        self.df = nodes_df
        return nodes_df
