import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Node:
    def __init__(self, idx, marker_id):
        self.idx = idx
        self.value = 100
        self.value2 = 100  # value2：when finding the next input，range all the neighbors of the input node. Assume add a neighbor into input, update value2 when value < value2
        self.edges = []
        self.marker_id = marker_id


class Edge:
    def __init__(self, value=None):
        self.value = value
        self.nodes = []
        self.activate = False
        self.activate2 = False


class Graph:
    def __init__(self, nodes, edges, full_panel, p1_img, p2_img, overlap):
        self.nodes = nodes
        self.edges = edges
        self.input_ = []
        # self.output = full_panel.copy()
        self.output = [x for x in range(46)]
        self.val_scores = []
        self.objectives = []
        self.train_scores = []
        self.val_corrs = []
        self.input_A = []
        self.input_B = []
        self.output_A = None
        self.output_B = None
        self.full_panel = full_panel
        self.p1_img = p1_img
        self.p2_img = p2_img
        self.overlap = overlap

    def getEdge(self):
        """ Return the edge with format: [node1, node2, edge value]"""
        edge_list = []
        for e in self.edges:
            temp = []
            for n in e.nodes:
                temp.append(n)
            temp.append(e.value)
            edge_list.append(temp)
        return edge_list

    def updateNodeValue(self):

        for n in self.nodes:
            temp = 2
            for e in n.edges:
                if e.activate:
                    if e.value < temp:
                        temp = e.value
            if temp < n.value:  # get the minimal edge weight
                n.value = temp

    def measureChange(self):
        # change = 0
        sum_node_value = 0
        for n in self.nodes:
            # change += (n.value2 - n.value)
            sum_node_value += n.value2
        return sum_node_value

    def updateNodeValue2(self):
        """ update each node's value2 when pretending adding a specific node into input """
        for n in self.nodes:
            # print(n.idx, n.value2)
            if n.value2 != 0:  # if not the chosen node itself
                if n.value == 0:  # if node is an input
                    n.value2 = 0
                else:  # if node is from output
                    temp = 100
                    for e in n.edges:  # update all the nodes' value
                        # print(e.nodes, e.value, e.activate, e.activate2, e.activate | e.activate2)
                        if e.activate | e.activate2:
                            # print(e.nodes, e.value)
                            if e.value < temp:  # minimize
                                temp = e.value.copy()
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

    def updateEdges(self, in_A=None, out_A=None, val_loss_path_A=None, in_B=None, out_B=None, val_loss_path_B=None):
        x = None
        for n in self.nodes:
            if n.marker_id == in_A[-1]:
                x = n.idx

        # set input's node value into 0.
        self.nodes[x].value = 0
        # activate all the x's edges
        for e in self.nodes[x].edges:
            e.activate = True

        new_edge_weights_fromA = np.mean(np.load(val_loss_path_A), axis=0)
        new_edge_weights_fromB = np.mean(np.load(val_loss_path_B), axis=0)

        # update edges
        sum_e = 0
        for i, e in enumerate(self.nodes[x].edges):
            if self.nodes[e.nodes[0]].idx != x:
                node_idx = e.nodes[0]
            else:
                node_idx = e.nodes[1]
            marker_id = self.nodes[node_idx].marker_id  # find the node having the edge connected to input
            if marker_id in np.intersect1d(out_A, out_B):  # overlap
                index1 = np.where(out_A == marker_id)[0]
                index2 = np.where(out_B == marker_id)[0]
                pred_loss = (new_edge_weights_fromA[index1] + new_edge_weights_fromB[index2]) / 2
                e.value = pred_loss
                self.nodes[node_idx].pred_loss = pred_loss
                sum_e += pred_loss
            elif marker_id in out_A:  # if marker in panelA output
                index = np.where(out_A == marker_id)[0]
                pred_loss = new_edge_weights_fromA[index]
                e.value = pred_loss
                self.nodes[node_idx].pred_loss = pred_loss
                sum_e += pred_loss
            elif marker_id in out_B:  # if marker in panelB output
                index = np.where(out_B == marker_id)[0]
                pred_loss = new_edge_weights_fromB[index]
                e.value = pred_loss
                self.nodes[node_idx].pred_loss = pred_loss
                sum_e += pred_loss
            else:  # if this marker also in input, it means the edge are connected by 2 input marker, then turn the edge weight to 0
                e.value = 0
                self.nodes[node_idx].pred_loss = 0
        assert len(self.output) == len(np.union1d(out_A, out_B))
        # print(sum_e[0], len(np.union1d(out_A, out_B)))
        # print("average loss: ", sum_e[0]/len(np.union1d(out_A, out_B)))

    def updateEdges2(self, in_, out_, val_loss_path):
        # m = 0
        x = None
        for n in self.nodes:
            if n.marker_id == in_[-1]:
                x = n.idx

        # set input's node value into 0.
        self.nodes[x].value = 0
        # activate all the x's edges
        for e in self.nodes[x].edges:
            e.activate = True

        new_edge_weights = np.mean(np.load(val_loss_path), axis=0)  # load loss from panelA

        # update edges from training model
        sum_e = 0
        for i, e in enumerate(self.nodes[x].edges):
            # find the node connected to input
            if self.nodes[e.nodes[0]].idx != x:
                node_idx = e.nodes[0]
            else:
                node_idx = e.nodes[1]
            marker_id = self.nodes[node_idx].marker_id
            if marker_id in out_:
                if marker_id in self.overlap:
                    index = np.where(out_ == marker_id)[0]
                    e.value = new_edge_weights[index]
                    # e.value = (e.value + new_edge_weights[index])/2
                    pred_loss = (self.nodes[node_idx].pred_loss + new_edge_weights[index])/2
                    self.nodes[node_idx].pred_loss = pred_loss
                    sum_e += pred_loss
                    # print(pred_loss, sum_e)
                    # m += 1
                else:
                    index = np.where(out_ == marker_id)[0]
                    pred_loss = new_edge_weights[index]
                    e.value = pred_loss
                    self.nodes[node_idx].pred_loss = pred_loss
                    sum_e += pred_loss
                    # print(pred_loss, sum_e)
                    # m += 1
            elif marker_id in self.input_:
                e.value = 0
                self.nodes[node_idx].pred_loss = 0

        # update edges that bridge 2 panels
        for i, e in enumerate(self.nodes[x].edges):
            node_idx = e.nodes[0]
            if node_idx == x:
                node_idx = e.nodes[1]
            marker_id = self.nodes[node_idx].marker_id

            if marker_id not in np.union1d(self.input_, out_):  # if marker is only in another panel (exclude overlapping part)
                ii = e.nodes[0]
                jj = e.nodes[1]

                e1 = []
                for ee in self.nodes[ii].edges:
                    n = ee.nodes[0]
                    if n == ii:
                        n = ee.nodes[1]
                    if n in range(20, 26):
                        e1.append(ee)
                        # print("e1", n)

                e2 = []
                for ee in self.nodes[jj].edges:
                    n = ee.nodes[0]
                    if n == jj:
                        n = ee.nodes[1]
                    if n in range(20, 26):
                        e2.append(ee)
                        # print("e2", n)

                sum_list = []
                assert len(e1) == 6
                for k in range(len(e1)):
                    sum = e1[k].value + e2[k].value
                    sum_list.append(sum)
                e.value = min(sum_list)
                sum_e += self.nodes[node_idx].pred_loss
                # print(self.nodes[node_idx].pred_loss, sum_e)
                # m += 1

        # assert m == len(self.output)
        # print(sum_e[0], len(self.output))
        # print("average loss: ", sum_e[0] / len(self.output))

    def updateAndFindNext(self, in_A=None, out_A=None, val_loss_path_A=None, in_B=None, out_B=None,
                          val_loss_path_B=None):
        """use validation loss to update the graph edges"""
        if (in_A is not None) & (in_B is not None):
            self.updateEdges(in_A, out_A, val_loss_path_A, in_B, out_B, val_loss_path_B)
        elif in_A is not None:
            self.updateEdges2(in_A, out_A, val_loss_path_A)
        elif in_B is not None:
            self.updateEdges2(in_B, out_B, val_loss_path_B)

        # update node value
        self.updateNodeValue()

        """find the next input"""
        node_values = []
        for n in self.nodes:
            if n.value != 0:
                n.value2 = 0
                for e in n.edges:
                    e.activate2 = True
                self.updateNodeValue2()
                sum_node_value = self.measureChange()
                node_values.append((n.idx, sum_node_value))
                self.clearValue2AndActivate2()

        node_values = np.array(node_values, dtype=object)
        min_nv = np.min(node_values[:, 1])
        self.objectives.append(min_nv)
        # print("sum of node value = ", np.min(node_values[:, 1]))

        x_idx = np.argmin(node_values[:, 1])
        x = int(node_values[:, 0][x_idx])

        # Ted: What if there are tie between the top 2?
        node_values = np.delete(node_values, x_idx, 0)
        x_idx2 = np.argmin(node_values[:, 1])
        min_nv2 = node_values[x_idx2, 1]
        x2 = int(node_values[:, 0][x_idx2])
        if min_nv == min_nv2:
            print("Alert! There is a tie between", x, "and", x2, "!")

        """print information"""
        self.print_info(x)

        return x

    # def stop_sign(self, val_corr_path, stop_threshold):
    #     corr = np.load(val_corr_path)
    #     val_score_each_chan = np.mean(corr, axis=0)
    #     val_s = np.mean(val_score_each_chan)
    #     self.val_corrs.append(val_s)
    #
    #     print("validation correlation of each channel: ", val_score_each_chan)
    #     print("validation score: ", val_s)
    #     if all(pred >= stop_threshold for pred in val_score_each_chan):
    #         print("stop selecting training biomarkers")
    #     else:
    #         print("continue selecting training biomarkers")

    def update(self, train_loss_path, val_loss_path, val_corr_path, stop_threshold):
        self.recordTrainLoss(train_loss_path)
        self.updateAndFindNext(val_loss_path)
        self.stop_sign(val_corr_path, stop_threshold)

    def train_which_model(self, idx):
        s1 = "A"
        s2 = "B"
        marker_name = self.full_panel[idx]
        if idx < 20:
            return s1, self.p1_img.index(marker_name)
        elif idx >= 26:
            return s2, self.p2_img.index(marker_name)
        else:
            return [s1, s2], [self.p1_img.index(marker_name), self.p2_img.index(marker_name)]

    def initialization(self):
        node_degrees = []
        for n in self.nodes:
            degree = sum(e.value for e in n.edges)
            node_degrees.append(degree)

        x = np.argmin(node_degrees)  # index for next round
        print("minimal node degree", np.min(node_degrees))

        self.print_info(x)
        return x

    def print_info(self, x):
        marker_name = self.full_panel[x]
        self.input_.append(marker_name)

        # print(f"input: panel index is {x}, whose marker name is {marker_name}")

        print(f"add marker {marker_name} into input")
        print("input:", self.input_)

        for i in self.output:
            # if i == self.full_panel[marker_name]:
            if i == marker_name:
                self.output.remove(i)
        print("output: ", self.output)

        panel, in_idx = self.train_which_model(x)
        print(f"training model in panel {panel}, input (image) channel index is {in_idx}")
        # print("***marker", marker_name, "is the order", in_idx, "channel (0,26) in panel", panel, "image ")
        # print("training model in panel", panel)

        if len(panel) == 1:
            if panel == 'A':
                p = self.p1_img
            if panel == 'B':
                p = self.p2_img
            assert self.full_panel[x] == p[in_idx]
        else:
            assert self.full_panel[x] == self.p1_img[in_idx[0]]
            assert self.full_panel[x] == self.p2_img[in_idx[1]]
