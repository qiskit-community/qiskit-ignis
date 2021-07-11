# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""
Visualization tools for graphs genrated for and by graph theoretical decoders.
"""


from mayavi import mlab
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
def draw_3d_error_graph( graph):
    """
        Args:
           graph (Retworkx Graph) : Error Graph to be visualised.
        Returns:
            A interactive graph window.
    """
    edges = graph.edge_list()
    nodes = graph.nodes()
    xyz = np.array(nodes)
    if max(xyz[:, 2]) == 0:
        resize_factor = 1
    else:
        resize_factor = max(xyz[:, 2])
    figure = mlab.figure(1, bgcolor=(1, 1, 1))
    figure.scene.disable_render = True
    pts = mlab.points3d(xyz[:, 0],xyz[:, 1],xyz[:, 2]/resize_factor,
                        scale_factor=0.02,color = (1,0,0),
                        resolution=100)
    for x in graph.nodes():
        mlab.text3d(x[0], x[1], x[2]/resize_factor, text = str(tuple(x)),
                    color = (0,0,0), scale=(0.01,0.01,0.01))
    for edge in graph.weighted_edge_list():
        mlab.text3d((nodes[edge[0]][0]+nodes[edge[1]][0])/2,
                    (nodes[edge[0]][1]+nodes[edge[1]][1])/2,
                    (nodes[edge[0]][2]+nodes[edge[1]][2])/(2*resize_factor),
                    text = str(abs(edge[2])),color = (0,0,0),
                    scale=(0.01,0.01,0.01))
    pts.mlab_source.dataset.lines = np.array(edges)
    tube = mlab.pipeline.tube(pts, tube_radius=0.0005)
    mlab.pipeline.surface(tube, color = (0,0,0))
    figure.scene.disable_render = False
def draw_2d_error_graph(graph):
    """
        Args:
           graph (Retworkx Graph) : Error Graph to be visualised.
        Returns:
            A 2-d graph.
    """
    G = nx.Graph()
    pos = {}
    i = 0
    for x,y,z in graph.nodes():
        if x == 0:
            pos[i] = (y,z)
            G.add_node(i,pos = (y,z))
            i += 1
        else:
            pos[i] = (y,z+2)
            G.add_node(i,pos = (y,z+2))
            i += 1
    plt.figure(figsize = (10,10))
    edge_labels = {}
    for _ in graph.edge_list():
        G.add_edge(_[0],_[1])
        edge_labels[(_[0],_[1])] = abs(graph.get_edge_data(_[0], _[1]))
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    return nx.draw(G, pos, with_labels = True, node_color = 'red', font_size = 8)
def draw_3d_decoded_graph( graph, Edgelist, nodelist):
    """
        Args:
           Graph (Networkx Graph) : Decoded Graph to be visualised.
        Returns:
            A interactive graph window.
    """
    pos = {}
    for x,i in enumerate(nodelist):
        pos[i] = x
    Edges = []
    for edge in Edgelist:
        Edges.append((pos[edge[0]],pos[edge[1]]))
    figure = mlab.figure(1, bgcolor=(1, 1, 1))
    xyz = np.array(list(pos.keys()))
    figure.scene.disable_render = True
    if max(xyz[:, 2]) == 0:
        resize_factor = 1
    else:
        resize_factor = max(xyz[:, 2])
    pts = mlab.points3d(xyz[:, 0],xyz[:, 1],xyz[:, 2]/resize_factor,
                        scale_factor=0.01,color = (0,0,1),
                        resolution=100)
    for x in nodelist:
        mlab.text3d(x[0]-(1/max(xyz[:, 2]*4)), x[1], x[2]/resize_factor,
                    text = str(tuple(x)),color = (0,0,0),
                    scale=(0.01,0.01,0.01))
    for x in Edgelist:
        mlab.text3d((x[0][0]+x[1][0])/2,
                    (x[0][1]+x[1][1])/2,
                    (x[0][2]+x[1][2])/(2*resize_factor),
                    text = str(abs(graph.get_edge_data(
                    graph.nodes().index(x[0]), graph.nodes().index(x[1])))),
                    color = (0,0,0), scale=(0.01,0.01,0.01))
    pts.mlab_source.dataset.lines = np.array(Edges)
    tube = mlab.pipeline.tube(pts, tube_radius=0.0005)
    mlab.pipeline.surface(tube, color = (0,0,0))
    figure.scene.disable_render = False
    return mlab.show()

def draw_2d_decoded_graph(graph, Edgelist, neutral_nodelist):
    """
        Args:
           Graph (Networkx Graph) : Decoded Graph to be visualised.
        Returns:
            A 2-d graph.
    """
    G = nx.Graph()
    pos = {}
    i = 0
    for x,y,z in graph.nodes():
        if (x,y,z) in neutral_nodelist:
            if x == 0:
                pos[i] = (y,z)
                G.add_node(i,pos = (y,z),)
                i += 1
            else:
                pos[i] = (y,z+2)
                G.add_node(i,pos = (y,z+2))
                i += 1
    plt.figure(figsize = (10,10))
    edge_labels = {}
    for _ in Edgelist:
        G.add_edge(neutral_nodelist.index(_[0]),neutral_nodelist.index(_[1]))
        edge_labels[neutral_nodelist.index(_[0]),
                    neutral_nodelist.index(_[1])]= abs(graph.get_edge_data(
                        graph.nodes().index(_[0]),graph.nodes().index(_[1])))
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    return nx.draw(G, pos, with_labels = True,
            node_color = 'b', font_size = 8)
