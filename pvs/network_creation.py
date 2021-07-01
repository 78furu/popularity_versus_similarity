#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import operator

from scipy.optimize import curve_fit
import matplotlib.colors as mcolors


from tqdm import tqdm
import time

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

    def __init__(self, n: int = 50, m: int = 5, T: float = 1., beta: float = 1., verbose: bool = True):
        super(Nodes, self).__init__()
        self.df = pd.DataFrame(columns = ['id', 'edges', 'r', 'theta'])
        self.n = n
        self.m = m
        self.T = T
        self.beta = beta

        self.nodes = self.creategraph(n = self.n, m = self.m,  T = self.T, beta = self.beta, verbose=verbose)
        self.nodes_df = pd.DataFrame().from_dict(self.nodes, orient='index')
        self.nodes_df['num_of_edges'] = [len(r["conns"]) for i, r in self.nodes_df.iterrows()]


    def creategraph(self, n: int = None, m: int = None, T: float = None, beta: float = None, verbose: bool = True) -> dict:
        """
            Creating the graph itself by adding nodes based on the algorithm specified by the parameters.

            n           int, number of nodes
            m           int, number of connections when new node added
            T           float, temperature
            beta        float, [0,1], drift parameter (0: full drift, 1: no drift)
        
            Returns a dict with nodes in it.
        """
        n = self.n if n is None else n
        m = self.m if m is None else m
        T = self.T if T is None else T
        beta = self.beta if beta is None else beta

        nodes = {}
        for i in tqdm(range(n), leave = False, desc = 'Adding nodes'):
            nodes = Nodes.add_node(nodes, i+1,m, T)
            nodes = Nodes.step_nodes(nodes, beta,i+1)
        self.nodes = nodes.copy()
        return nodes

    def add_node(nodes: dict, n: int, m: int, T: float) -> dict:
        """
            Adding a node to an existing dict of nodes.

            nodes       dict, containing the nodes
            n           int, number of id of node to be added
            m           int, number of connections when new node added
            T           float, temperature

            Returns the modified nodes dict.
        """
        m_put = 0
        nodeid = 0
        nodes[n] = {'conns' : [], 'theta' : np.random.rand()*360, 'pos': n } 
        dists = {}
        for key in nodes.keys():
            if key!=n:
                dists[key] = np.log(nodes[n]['pos']*nodes[key]['pos']
                                 *np.abs(np.deg2rad(nodes[n]['theta']-nodes[key]['theta']))*0.5)
        sorted_dists = sorted(dists.items(), key=operator.itemgetter(1))
        #print(sorted_dists)

        if n< m:
            for key in nodes.keys():
                if key != n:
                    nodes[n]['conns'].append(key)
            m_put = m

        while m_put < m:
            try:
                v = sorted_dists[nodeid][0]
                if v not in nodes[n]['conns']:
                    prob = 1/(1 + np.exp((sorted_dists[nodeid][1]-np.log(n))/T ))
                    r = np.random.rand()
                    if r<=prob:
                        nodes[n]['conns'].append(v)
                        m_put +=1
                        nodeid +=1
                    else:
                        #print("Unable to attach:", prob, r, ' nodeid:', nodeid)
                        nodeid +=1
                #print('added:', v)
            except IndexError:
                #print('not enough nodes yet')
                m_put = m

        for v in nodes[n]['conns']:
            if n not in nodes[v]['conns']:
                nodes[v]['conns'].append(n)
        return nodes

    def step_nodes(nodes: dict, beta: float, T: float) -> dict:
        """Updating the _nodes_ dict's radii based on the _beta_ drifting parameter and _T_ temperature. Return mod. dict."""
        for key in nodes.keys():
            nodes[key]['pos'] = beta*np.log(key) + (1-beta)*np.log(T)
        return nodes


    def add_node_depr(self, m: int = None, T: int = None, beta: float = None) -> pd.DataFrame:
        """
        DEPRECATED: SLOW
        Adding a node to existing network. Any given parameter will override the netowork's
        default during node creation, but not permamently. Default parameters are the 
        network's inner parameters. 

            m           int, number of connections when new node added
            T           float, temperature
            beta        float, [0,1], drift parameter (0: full drift, 1: no drift)

        Returns nodes in a pandas.DataFrame (and refreshes self.df, too). 
        """
        t = time.time()
        
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

                
        newnode = nodes_df.loc[nodes_df.id == new_id]
        dists = {node["id"] : Nodes.calculate_dist(nodes_df, node["id"], new_id, beta = beta) for index, node in nodes_df.iterrows() if node['id'] != new_id}
        dt = time.time()-t
        t = time.time()
        print(f"\t calced dist, time: {dt}")
        sorted_dists = {k: v for k, v in sorted(dists.items(), key=lambda item: item[1])}
        dt = time.time()-t
        t = time.time()
        print(f"\t sorted dist, time: {dt}")
        
        m_put = 0
        put = []
        while m_put < m:
            nodeid = np.random.choice(list(sorted_dists.keys()))
            dist = sorted_dists[nodeid]
            r = np.random.rand()
            prob = 1/(1 + np.exp((dist-np.log(new_id))/T ))
            if r<= prob and nodeid not in put:
                create_connection(nodeid, new_id)
                m_put += 1
                put.append(nodeid)
            else:
                continue
        self.df = nodes_df
        dt = time.time()-t
        t = time.time()
        print(f"\t put nodes, time: {dt}")
        print(f"done node: {new_id}")
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

    def get_node(self, nodeid: int, nodes_df: pd.DataFrame = None) -> pd.DataFrame :
        """
        Get the parameters of a node with nodeid. Returns pandas.DataFrame
        """
        nodes_df = self.df if nodes_df is None else nodes_df
        return nodes_df[nodes_df.id == nodeid]

        
    def set_node(self, nodeid: int, param: str, value, nodes_df: pd.DataFrame = None):
        """
        Set the parameter param of a node with nodeid with value. Returns nothing.
        """
        nodes_df = self.df if nodes_df is None else nodes_df
        node = nodes_df[nodes_df.id == nodeid]
        idx = node.index.values[0]
        nodes_df.at[idx, param] = value
        self.df = nodes_df

    def to_edgelist(self, save_as = 'edgelist.txt'):
        """Saving as an edgelist under _save_as_."""
        import copy

        nodesc = copy.deepcopy(self.nodes)
        f = open(save_as, "w")
        for key in nodesc.keys():
            for v in nodesc[key]['conns']:
                f.write(str(key) + ' ' + str(v) + '\n')
                nodesc[v]['conns'].remove(key)
        f.close()


    def to_networkx(self) -> nx.Graph:
        """Returns a Networkx Graph object."""
        import networkx as nx
        import copy
        G = nx.Graph()
        edges = []
        nodesc = copy.deepcopy(self.nodes)
        for key in nodesc.keys():
            for v in nodesc[key]['conns']:
                edges.append((key, v))
                nodesc[v]['conns'].remove(key)
        G.add_edges_from(edges)

        return G

    def plotnodes(nodes, delta = .15, save_as = None, ax = None ) -> plt.ax:
        """
        Creating a hyperbolic plot.

            nodes:      dict, containing the nodes
            delta:      float (def: .15), length of arrows indicating the direction of nodes
            save_as:    str (def: None), save as figure
            ax:         plt.ax (def: None), plotting on given axis

        Returns the modified plt.ax object.
        """
        ax = plt.subplot(111, projection='polar') if ax is None else ax

        ax.plot([np.deg2rad(nodes[key]['theta']) for key in nodes.keys()],
                [np.log(nodes[key]['pos']) for key in nodes.keys()], 'o')

        for key in nodes.keys():
            for v in nodes[key]['conns']:
                ax.plot([np.deg2rad(nodes[key]['theta']), np.deg2rad(nodes[v]['theta'])], 
                        [np.log(nodes[key]['pos']), np.log(nodes[v]['pos'])], color = 'red')

        for key in nodes.keys():
            ax.text(np.deg2rad(nodes[key]['theta']) , np.log(nodes[key]['pos']) , key)
            ax.arrow(np.deg2rad(nodes[key]['theta']) , np.log(nodes[key]['pos']), 0, delta)


        ax.grid(True)
        if save_as is not None:
            plt.tight_layout()
            plt.savefig(save_as, dpi  =200)
        return ax