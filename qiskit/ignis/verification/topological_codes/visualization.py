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
Visualtization Tools for Graph Theoritical Decoders. 3D and 2D both, it requires
some extra libraries
"""
import numpy as np

from qiskit.exceptions import QiskitError

try:
    from retworkx.visualization import mpl_draw
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


class GraphVisualization():
    """Class to draw 3D and 2D graphs needed for graph theoritical decoders."""

    def __init__(self):
        """Blank init function."""
        pass

    def draw_3d_error_graph(self, graph, notebook=False):
        """Draws a 3d Error Graph.

        Args:
            graph (retworkx.PyGraph) : Error Graph to be visualised.
            notebook (bool) : Set True if using Jupyter.

        Returns:
            list: List of camera position, focal point, and view up.
            numpy.ndarray: Array containing pixel RGB and optionally
            alpha values.

        Raises:
            QiskitError: If pyvista is not installed
        """
        if not HAS_PYVISTA:
            raise QiskitError('please install pyvista')

        nodes = np.array(graph.nodes(), dtype='f')
        edges = []
        edge_label = []
        for edge in graph.weighted_edge_list():
            edges.append(graph[edge[0]])
            edges.append(graph[edge[1]])
            edge_label.append(str(abs(edge[2])))
        edges = np.array(edges, dtype='f')
        if max(np.array(graph.nodes())[:, 2]) == 0:
            resize = 1
        else:
            resize = float(max(np.array(graph.nodes())[:, 2]))
        nodes[:, 2] = nodes[:, 2]/resize
        edges[:, 2] = edges[:, 2]/resize
        edge_centers = [(edges[i]+edges[i+1])/2
                        for i in range(0, len(graph.edges())*2, 2)]
        labels = [str(i)for i in graph.nodes()]
        # Plotting
        p = pv.Plotter(notebook=notebook)
        p.set_background("white")
        pdata = pv.PolyData(nodes)
        edata = pv.PolyData(edge_centers)
        p.add_point_labels(pdata, labels, point_size=10, font_size=10,
                           text_color='black', point_color='red',
                           render_points_as_spheres=True,
                           shape_opacity=0, always_visible=True)
        p.add_point_labels(edata, edge_label, show_points=False, font_size=10,
                           text_color='black', shape_opacity=0, always_visible=True)
        for i in range(0, len(edges), 2):
            p.add_lines(np.array([edges[i], edges[i+1]]),
                        width=1, color='black')
        return p.show()

    def draw_2d_error_graph(self, graph):
        """Draws a 2d Error Graph.

        Args:
            graph (retworkx.PyGraph) : Error Graph to be visualised.

        Returns:
            retworkx: A 2-d graph.

        Raises:
            QiskitError: If matplotlib is not installed.
        """
        if not HAS_MATPLOTLIB:
            raise QiskitError('please install matplotlib')
        pos = {}
        for i in graph.node_indexes():
            if graph[i][0] == 0:
                pos[i] = (graph[i][1], graph[i][2])
            else:
                pos[i] = (graph[i][1], graph[i][2]+2)
        return mpl_draw(graph, pos=pos, with_labels=True, node_color='red',
                        labels=lambda node: str(node),  # pylint: disable=W0108
                        edge_labels=lambda edge: str(abs(edge)))

    def draw_3d_decoded_graph(self, graph, edgelist, nodelist, notebook=False):
        """Draws a 3d Decoded Graph.

        Args:
            graph (retworkx.PyGraph) : Decoded Graph to be visualised.

            edgelist (list) : List of matched edges.

            nodelist (list) : List of matched nodes.

            notebook (bool) : Set True if using Jupyter.

        Returns:
            list: List of camera position, focal point, and view up.
            numpy.ndarray: Array containing pixel RGB and optionally
            alpha values.

        Raises:
            QiskitError: If pyvista is not installed.
        """
        if not HAS_PYVISTA:
            raise QiskitError('please install pyvista')

        nodes = np.array(nodelist, dtype='f')
        labels = [str(i)for i in nodes]
        edges = []
        edge_label = []
        for edge in edgelist:
            edges.append(edge[0])
            edges.append(edge[1])
            edge_label.append(abs(graph.get_edge_data(
                    graph.nodes().index(edge[0]),
                    graph.nodes().index(edge[1]))))
        edges = np.array(edges, dtype='f')
        if max(np.array(graph.nodes())[:, 2]) == 0:
            resize = 1
        else:
            resize = float(max(np.array(graph.nodes())[:, 2]))
        nodes[:, 2] = nodes[:, 2]/resize
        edges[:, 2] = edges[:, 2]/resize
        edge_centers = [(edges[i]+edges[i+1])/2
                        for i in range(0, len(edgelist)*2, 2)]
        # Plotting
        p = pv.Plotter(notebook=notebook)
        p.set_background("white")
        pdata = pv.PolyData(nodes)
        edata = pv.PolyData(edge_centers)
        p.add_point_labels(pdata, labels, point_size=10, font_size=10,
                           text_color='black', point_color='blue',
                           render_points_as_spheres=True,
                           shape_opacity=0, always_visible=True)
        p.add_point_labels(edata, edge_label, show_points=False, font_size=10,
                           text_color='black', shape_opacity=0, always_visible=True)
        for i in range(0, len(edges), 2):
            p.add_lines(np.array([edges[i], edges[i+1]]),
                        width=1, color='blue')
        return p.show()

    def draw_2d_decoded_graph(self, graph, edgelist, neutral_nodelist):
        """Draws a 3d Decoded Graph.

        Args:
            graph (retworkx.PyGraph) : Decoded Graph to be visualised

            edgelist (list) : List of matched edges.

            neutral_nodelist (list) : List of matched nodes.

        Returns:
            networkx: A 2-d graph.

        Raises:
            QiskitError: If matplotlib is not installed.
        """
        if not HAS_MATPLOTLIB:
            raise QiskitError('please install matplotlib')
        graph_c = graph.copy()
        pos = {}
        for i in graph_c.node_indexes():
            if graph[i][0] == 0:
                pos[i] = (graph[i][1], graph[i][2])
            else:
                pos[i] = (graph[i][1], graph[i][2]+2)
        edges = graph.edge_list()
        for edge in edges:
            if (graph[edge[0]],
                graph[edge[1]]) in edgelist or (graph[edge[1]],
                                                graph[edge[0]]) in edgelist:
                pass
            else:
                graph_c.remove_edge(edge[0], edge[1])
        nodes = graph.nodes()
        for node in nodes:
            if node not in neutral_nodelist:
                graph_c.remove_node(graph.nodes().index(node))
        return mpl_draw(graph_c, pos=pos, with_labels=True, node_color='blue',
                        labels=lambda node: str(node),  # pylint: disable=W0108
                        edge_labels=lambda edge: str(abs(edge)))
