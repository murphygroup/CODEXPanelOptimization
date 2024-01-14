import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Node:
    def __init__(self, idx, num_biomarkers):
        self.idx = idx
        self.value = 100
        self.value2 = 100  # value2：when finding the next input，range all the neighbors of the input node. Assume add a neighbor into input, update value2 when value < value2
        self.edges = []
        self.init_degree = None


class Edge:
    def __init__(self, value=None):
        self.value = value
        self.init_value = value.copy()
        self.nodes = []
        self.activate = False
        self.activate2 = False


class Graph:
    def __init__(self, nodes, edges, num_biomarkers):
        self.nodes = nodes
        self.edges = edges
        self.input_ = []
        self.output = [i for i in range(num_biomarkers)]
        self.val_scores = []
        self.objectives = []
        self.train_scores = []
        self.val_corrs = []
        self.sum_nv_list = [] # store the sum of node weights

    def getEdge(self):
        """ Return the edge with format: (node1, node2, edge value)"""
        edge_list = []
        for e in self.edges:
            temp = []
            for n in e.nodes:
                temp.append(n)
            temp.append(e.value)
            edge_list.append(temp)
        return edge_list

    def updateNodeValue(self):
        # maxChange = (0, 0)
        for n in self.nodes:
            temp = 100
            for e in n.edges:
                if e.activate:
                    if e.value < temp:
                        temp = e.value
            if temp < n.value:  # minimize mse
                n.value = temp
        #     degreeChange = n.init_degree - n.value
        #     print(n.idx, degreeChange, "=", n.init_degree, "-", n.value)
        #     if (degreeChange > maxChange[1]) & (n.value != 0):
        #         maxChange = (n.idx, degreeChange)
        # print("node with maximum decreasing node degree: ", maxChange)

    def measureChange(self):
        # change = 0
        sum_node_value = 0
        for n in self.nodes:
            # change += (n.value2 - n.value)
            sum_node_value += n.value2
        return sum_node_value

    def updateNodeValue2(self):
        """update each node's value2 when pretending putting a specific marker into input """
        for n in self.nodes:
            # print(n.idx, n.value2)
            if n.value2 != 0:  # if not the chosen node itself
                if n.value == 0:  # if node is an input
                    n.value2 = 0
                else:             # if node is from output
                    temp = 100
                    for e in n.edges:  # update all the nodes' value
                        # print(e.nodes, e.value, e.activate, e.activate2, e.activate | e.activate2)
                        if e.activate | e.activate2:
                            # print(e.nodes, e.value)
                            if e.value < temp:  # minimize
                                temp = e.value
                    n.value2 = temp
                    # print(n.idx, n.value2)

    def clearValue2AndActivate2(self):
        for n in self.nodes:
            n.value2 = 100
            for e in n.edges:
                e.activate2 = False

    def plotGraph(self):
        fig = plt.figure(1, figsize=(20, 10), dpi=60)

        G = nx.Graph()

        for (i, j, v) in self.getEdge():
            v = float("{:.2f}".format(v))
            G.add_edge(str(i), str(j), weight=v)

        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

        pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=700)

        # edges
        nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
        nx.draw_networkx_edges(
            G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
        )

        # node labels
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
        # edge weight labels
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

        ax = plt.gca()
        ax.margins(0.08)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def recordTrainLoss(self, train_loss_path):
        corr = np.load(train_loss_path)
        new_edge_weights = np.mean(corr, axis=0)
        train_loss = np.mean(new_edge_weights)
        self.train_scores.append(train_loss)
        print("Train Loss in each target channel: ", new_edge_weights)
        print("Train Loss: ", train_loss)

    def updateAndFindNext(self, idx, val_loss_path):
        """update the graph edges"""
        self.nodes[idx].value = 0

        for e in self.nodes[idx].edges:
            e.activate = True

        corr = np.load(val_loss_path, allow_pickle=True)
        if len(corr.shape) == 2:
            new_edge_weights = np.mean(corr, axis=0)
        else:
            new_edge_weights = corr
        val_loss = np.mean(new_edge_weights)
        self.val_scores.append(val_loss)

        # print("validation loss in each target channel: ", new_edge_weights)
        # print("validation loss: ", val_loss)

        assert len(self.nodes[idx].edges) - (len(self.input_) - 1) == len(new_edge_weights)

        i = 0
        # maxDecrease = (0, 0)
        for e in self.nodes[idx].edges:
            if (e.nodes[0] in self.output) | (e.nodes[1] in self.output):
                # assert e.value <= new_edge_weights[i]
                e.value = new_edge_weights[i]
                # edgeChange = e.init_value - e.value
                # print(e.nodes, edgeChange, "=", e.init_value,"-", e.value)
                # if edgeChange > maxDecrease[1]:
                #     maxDecrease = (e.nodes, edgeChange)
                i += 1
            if (e.nodes[0] in self.input_) & (e.nodes[1] in self.input_):
                e.value = 0
            # or if/else logic
        # print("edge with maximum decreasing edge weight:", maxDecrease) "check edge changes of edge weights#
        # update node value
        self.updateNodeValue()

        """find the next input marker"""
        # changes = []
        node_values = []
        for n in self.nodes:
            # print("*** assume add", n.idx, "into input", n.value, n.value2)
            if n.value != 0:  # choose node from output
                n.value2 = 0
                for e in n.edges:
                    e.activate2 = True
                self.updateNodeValue2()
                sum_node_value = self.measureChange()
                # changes.append((n.idx, change))
                node_values.append((n.idx, sum_node_value))
                self.clearValue2AndActivate2()

        # print(node_values)

        # changes = np.array(changes)
        node_values = np.array(node_values)
        # print(node_values)
        # max_nv = np.max(node_values[:, 1])
        # self.objectives.append(max_nv)
        min_nv = np.min(node_values[:, 1])
        self.objectives.append(min_nv)
        # print("total change = ", np.min(changes[:, 1]))
        # print("sum of node value = ", np.min(node_values[:, 1]))

        # x_idx = np.argmax(changes[:, 1])
        # max_change = changes[x_idx, 1]
        # x = int(changes[:, 0][x_idx])

        x_idx = np.argmin(node_values[:, 1])
        x = int(node_values[:, 0][x_idx])  # selected marker should be the one cause the least risk when added into input

        # print(node_values)

        # print(x, x3) # verify if the selected node with max change is the same with the node with the min loss
        # assert x == x3

        # Ted: What if there are tie between the top 2?
        # print(changes)
        node_values = np.delete(node_values, x_idx, 0)
        x_idx2 = np.argmin(node_values[:, 1])
        min_nv2 = node_values[x_idx2, 1]
        x2 = int(node_values[:, 0][x_idx2])
        # print(min_nv, min_nv2)
        # if min_nv == min_nv2:
        #     print("Alert! There is a tie between", x, "and", x2, "!")

        self.input_.append(x)
        # idx = int(self.input_[-1])
        idx = x

        print("add", idx, "into input")
        print("input:", self.input_)

        for i in self.output:
            if i == idx:
                self.output.remove(i)
        print("output: ", self.output)

        sum_nv = 0
        for n in self.nodes:
            sum_nv += n.value
        self.sum_nv_list.append(sum_nv)

        return idx

    def stop_sign(self, val_corr_path, stop_threshold):
        corr = np.load(val_corr_path)
        val_score_each_chan = np.mean(corr, axis=0)
        val_s = np.mean(val_score_each_chan)
        self.val_corrs.append(val_s)

        print("validation correlation of each channel: ", val_score_each_chan)
        print("validation score: ", val_s)
        if all(pred >= stop_threshold for pred in val_score_each_chan):
            print("stop selecting training biomarkers")
        else:
            print("continue selecting training biomarkers")

    def update(self, train_loss_path, val_loss_path, val_corr_path, stop_threshold):
        self.recordTrainLoss(train_loss_path)
        self.updateAndFindNext(val_loss_path)
        self.stop_sign(val_corr_path, stop_threshold)


