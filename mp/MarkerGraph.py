import numpy as np

class MarkerGraph:
    """construct the marker graph"""
    def __init__(self, subgraphs_idx, subgraphs_mtx, num_markers):
        """
        subgraphs_idx: list of lists, each list contains the node indices in a subgraph
        subgraphs_mtx: list of matrices, each matrix is the adjacency matrix of a subgraph
        num_markers: number of markers
        """
        self.subgraphs_idx = subgraphs_idx 
        self.subgraphs_mtx = subgraphs_mtx
        for i in range(len(subgraphs_idx)):
            assert len(subgraphs_idx[i]) == subgraphs_mtx[i].shape[0]
        self.num_markers = num_markers
        self.overlap_nodes = self.get_overlap_nodes()
        self.overlap_sets = self.get_overlap_sets(self.subgraphs_idx)
        self.sg_nodes, self.sg_mtx = self.get_subgraph_graph() # get a subgraph graph whose nodes are subgraphs
        self.sg_graph = get_adjacency_list(self.sg_nodes, self.sg_mtx)
        self.node_weights = np.zeros(num_markers)
        self.node_activation = np.ones(num_markers)            # 0: active, 1: inactive
        self.subgraph_inout = self.initialize_subgraph_inout()
        self.subgraph_out_last = [None for i in range(len(self.subgraph_inout))]
        self.mtx = self.initialize_adjacency_matrix()
        self.mtx0 = self.mtx.copy()                            # adjacency matrix without bridging subpanels
        self.mask0 = np.where(self.mtx0 > 0, 1, 0)             # mask for the original mtx without bridging subpanels
        self.bridge_subpanels()

    def locate_node(self, nidx):
        """
        locate the node in which subgraph
        nidx: node index
        return: list of subgraph indices that contain the node
        """
        gidx_list = []
        for gidx, g in enumerate(self.subgraphs_idx):
            if nidx in g:
                gidx_list.append(gidx)
        return gidx_list
    
    def get_overlap_nodes(self):
        """get a list of nodes that are at least in two subgraphs"""
        overlap_list = []
        for g1 in self.subgraphs_idx:
            for g2 in self.subgraphs_idx:
                if g1 != g2:
                    ol = list(set(g1).intersection(g2))
                    overlap_list = list(set(overlap_list + ol))
        return overlap_list
    
    def get_overlap_nodes_connecting2sg(self, gs, gt):
        """get a list of nodes that are on the paths between two subgraphs"""
        paths = find_all_paths(gs, gt, self.sg_graph)  # find all paths from gs to gt in subgraph graph
        overlap_nodes = []
        for p in paths:
            for i in range(len(p)):
                for j in range(i+1, len(p)):
                    gi = self.subgraphs_idx[p[i]]
                    gj = self.subgraphs_idx[p[j]]
                    ol = list(set(gi).intersection(set(gj)))
                    overlap_nodes = list(set(overlap_nodes + ol))
        return overlap_nodes
    
    def get_overlap_sets(self, subgraph_list):
        """get a list of overlap node sets between each 2 subgraphs in subgraph list subgraph_list"""
        overlap_sets = []
        for g1 in subgraph_list:
            for g2 in subgraph_list:
                if g1 != g2:
                    ol = list(set(g1).intersection(g2))
                    if len(ol) > 0:
                        if ol not in overlap_sets:
                            overlap_sets.append(ol)
        return overlap_sets
    
    def get_subgraph_graph(self):
        """get subgraph graph whose nodes are subgraphs"""
        sg_nodes = [i for i in range(len(self.subgraphs_idx))]
        sg_mtx = np.zeros((len(self.subgraphs_idx), len(self.subgraphs_idx)))
        for i in range(len(self.subgraphs_idx)):
            for j in range(i+1, len(self.subgraphs_idx)):
                if len(set(self.subgraphs_idx[i]).intersection(set(self.subgraphs_idx[j]))) > 0:
                    sg_mtx[i, j] = sg_mtx[j, i] = 1
        return sg_nodes, sg_mtx
    
    def break_edges_in_overlap_sets(self, overlap_sets_between2subgraphs):
        """break edges between overlap nodes"""
        out_mtx = self.mtx0.copy()
        # for s in self.overlap_sets:
        for s in overlap_sets_between2subgraphs:
            for i in range(len(s)):
                for j in range(i+1, len(s)):
                    out_mtx[s[i], s[j]] = out_mtx[s[j], s[i]] = 0
        return out_mtx

    def get_shortest_path(self, gs, gt, s, t):
        """
        get the shortest path between two subgraphs by simplifying the graph by only keeping the overlap nodes, source node s and target node t
        gs: source subgraph index
        gt: target subgraph index
        return: simplified nodes and adjacency matrix
        """
        overlap_nodes = self.get_overlap_nodes_connecting2sg(gs, gt)
        simplified_nodes = list(set(overlap_nodes + [s, t]))
        simplified_nodes.sort()
        
        min_dist = np.inf
        subgraphs_paths = find_all_paths(gs, gt, self.sg_graph)                               # find all paths from gs to gt in subgraph graph
        for subgraphs_graphidx_between2sgs in subgraphs_paths:
            subgraphs_between2sgs = [self.subgraphs_idx[i] for i in subgraphs_graphidx_between2sgs]
            overlap_nodes_between2subgraphs = self.get_overlap_sets(subgraphs_between2sgs)
            simplified_mtx = self.break_edges_in_overlap_sets(overlap_nodes_between2subgraphs) # break edges between overlap nodes
            simplified_mtx = simplified_mtx[simplified_nodes, :][:, simplified_nodes] 

            dist, _ = self.dijkstra(s, t, simplified_nodes, simplified_mtx)                    # get the shortest distance between s and t
            if dist[t] < min_dist:
                min_dist = dist[t]
        return min_dist
    
    def dijkstra(self, s, t, nodes, mtx):
        """dijkstra algorithm to find the shortest path"""
        dist = {}
        prev = {}
        Q = []
        for v in nodes:
            dist[v] = np.inf
            prev[v] = None
            Q.append(v)
        dist[s] = 0

        while len(Q) > 0:
            u = min(Q, key=lambda x: dist[x])
            if u == t:
                break
            Q.remove(u)
            for v in Q:
                i = nodes.index(u)
                j = nodes.index(v)
                if mtx[i, j] > 0:
                    alt = dist[u] + mtx[i, j]
                    if alt < dist[v]:
                        dist[v] = alt
                        prev[v] = u
        return dist, prev

    def calc_edge_upperbound(self, i, j):
        """calculate the upperbound of the edge weight between node i and node j"""
        min_dist = np.inf
        gs_list = self.locate_node(i)
        gt_list = self.locate_node(j)
        for gs in gs_list:
            for gt in gt_list: 
                assert gs != gt                               # make sure gs and gt are not the same subgraph
                dist = self.get_shortest_path(gs, gt, i, j)   # get the shortest distance between i and j by simplifying the graph
                if dist < min_dist:
                    min_dist = dist
        return min_dist
    
    def bridge_subpanels(self):
        """bridge the subpanels connected by the overlap nodes"""
        for i in range(self.num_markers):
            for j in range(i+1, self.num_markers):
                if self.mtx0[i, j] == 0:
                    ub = self.calc_edge_upperbound(i, j) # one entry
                    self.mtx[i, j] = ub
                    self.mtx[j, i] = ub 

    def update_input_output(self, input_idx):
        """update the input and output"""
        self.input.append(input_idx)
        self.output.remove(input_idx)
        print("input:", self.input)
        print("output:", self.output)
        print("--------------------------------")
        self.node_activation[input_idx] = 0  # activate input node
        self.update_subgraph_inout()
    
    def select_first_input(self):
        """select the first input marker"""
        node_degree = np.sum(self.mtx, axis=0)
        min_idx = np.argmin(node_degree)
        input_idx = min_idx

        # update input and output
        self.input = [input_idx]
        self.output = np.delete(np.arange(self.num_markers), input_idx).tolist()
        self.node_activation[input_idx] = 0 
        print("input:", self.input)
        print("output:", self.output)
        print("--------------------------------") 

        self.update_subgraph_inout()

        return input_idx
    
    def select_next_input(self):
        """select the next input marker from updated graph by maximizing the expected improvement(expected change on node weight)"""
        max_improvement = 0
        min_exp_sumOfNodeWeights = np.inf
        next_input = []
        exp_improvements = []
        for exp_in in self.output:
            inputs_exp = self.input.copy()
            node_activation_exp = self.node_activation.copy()
            inputs_exp.append(exp_in)
            node_activation_exp[exp_in] = 0
            expected_node_weights = np.min(self.mtx[inputs_exp], axis=0)
            expected_node_weights *= node_activation_exp
            exp_improvement = np.sum(self.node_weights - expected_node_weights)
            exp_improvements.append(exp_improvement)
            if exp_improvement > max_improvement:
                max_improvement = exp_improvement                          # record the maximum expected improvement
                min_exp_sumOfNodeWeights = np.sum(expected_node_weights)   # record the minimum expected sum of node weights
       
        max_improvement_idx = np.where(np.array(exp_improvements) == max_improvement)[0]
        if len(max_improvement_idx) > 1:
            print('tie')

            print('candidates: ', end=' ')
            for i in max_improvement_idx:
                c = self.output[i]
                print(str(c), end=' ')

        # TO_DO: break the tie
        next_input = self.output[max_improvement_idx[0]]

        self.max_improvement = max_improvement
        self.min_exp_sumOfNodeWeights = min_exp_sumOfNodeWeights
 
        self.update_input_output(next_input)

        return next_input
    
    def initialize_subgraph_inout(self):
        """initialize the input and output in each subgraph"""
        subgraph_inout = []
        for g in self.subgraphs_idx:
            # length = len(g)
            # input = []
            # output = list(range(length))
            # subgraph_inout.append((input, output))
            input = []
            output = g.copy()
            subgraph_inout.append((input, output))
        return subgraph_inout
    
    def initialize_adjacency_matrix(self):
        """initialize the adjacency matrix from multiple matrices in subgraphs""" 
        mtx = np.zeros((self.num_markers, self.num_markers))
        mask = mtx.copy()
        for g_idx, g in enumerate(self.subgraphs_idx):
            for idx, i in enumerate(g):
                for idx2, j in enumerate(g):
                    mtx[i, j] += self.subgraphs_mtx[g_idx][idx, idx2]
                    mask[i, j] += 1
        return np.divide(mtx, mask, out=np.zeros_like(mtx), where=mask!=0)
    
    def update_subgraph_inout(self):
        """update the input and output in the subgraph"""
        input_idx = self.input[-1]
        training_subgraph_idx = []
        for i, g in enumerate(self.subgraphs_idx):
            if input_idx in g:
                training_subgraph_idx.append(i)
                input = self.subgraph_inout[i][0]
                output = self.subgraph_inout[i][1]
                input.append(input_idx)
                output.remove(input_idx)
                self.subgraph_inout[i] = (input, output)
        self.training_subgraph_idx = training_subgraph_idx   # record the subgraph idx for training

        # print training info
        for i in training_subgraph_idx:
            print('training subgraph: ', i, 'input: ', self.subgraph_inout[i][0], 'output: ', self.subgraph_inout[i][1])
            if len(self.subgraph_inout[i][0]) > 1:
                input_idx = self.input[-1]
                output_last = self.subgraph_out_last[i]
                out_drop_index = output_last.index(input_idx)
                print('--out_drop_index: ', out_drop_index)
            self.subgraph_out_last[i] = self.subgraph_inout[i][1].copy()
    
    def update_graph_edge(self):
        """update the edge weight in the graph"""
        self.mtx = self.initialize_adjacency_matrix()
        self.mtx0 = self.mtx.copy()            

        self.bridge_subpanels()         # update all the bridge edges in the graph

    def update_node_weight(self):
        """update the node weight"""
        self.node_weights = np.min(self.mtx[self.input], axis=0)    # to calculate each node's weight, we sum up the edge weights(activated) of the input nodes
        self.node_weights *= self.node_activation                   # let the input nodes have zero weight

    def update_subgraph(self, sg_idx, subgraph_prediction):
        """update the subgraph with the predicted loss from the subpanel"""
        assert len(subgraph_prediction) == len(self.subgraph_inout[sg_idx][1])
        sg = self.subgraphs_mtx[sg_idx].copy()
        sg_markers = self.subgraphs_idx[sg_idx].copy()
        in_ = self.subgraph_inout[sg_idx][0][-1]
        out_list = self.subgraph_inout[sg_idx][1]
        in_idx = sg_markers.index(in_)
        for i, out_ in enumerate(out_list):
            out_idx = sg_markers.index(out_)
            sg[in_idx, out_idx] = sg[out_idx, in_idx] = subgraph_prediction[i]
        self.subgraphs_mtx[sg_idx] = sg


    def update(self):
        """update the graph"""
        self.update_graph_edge()
        self.update_node_weight()
    
    

def get_adjacency_list(nodes, mtx):
    """get adjacency list from adjacency matrix"""
    adj_list = {}
    for i in range(len(nodes)):
        adj_list[i] = []
        for j in range(len(nodes)):
            if mtx[i, j] == 1:
                adj_list[i].append(j)   
    return  adj_list

def find_all_paths(s, t, adj_list):
    """find all paths from s to t"""
    visitedList = []
    def depthFirst(graph, currentVertex, visited):
        visited.append(currentVertex)
        for vertex in graph[currentVertex]:
            if vertex not in visited:
                depthFirst(graph, vertex, visited.copy())
        visitedList.append(visited)
    depthFirst(adj_list, s, [])
    paths = []
    for p in visitedList:
        if p[-1] == t:
            paths.append(p)
    return paths

