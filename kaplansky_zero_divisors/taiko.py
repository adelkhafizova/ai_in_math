import networkx as nx


class Taiko(nx.DiGraph):
    default_color = dict(color=0)

    def single_edge_dict(self):
        """
        Creates the default color dictionary for a new edge.

        :return: the default color dictionary.
        """
        return self.default_color

    edge_attr_dict_factory = single_edge_dict

    def __init__(self, two_cell_list=[]):
        """
        Creates a taiko with the given two cells and upper bounds for vertices.

        2-cells are given as 4-tuples of positive integers (i1, j1, i2, j2).
        If i1 < i2, the orientation is positive, and the orientation is negative otherwise.

        :param two_cell_list: the list of 4-tuples giving the vertices and orientations of the 2-cells
        """
        nx.DiGraph.__init__(self)

        self.next_color = 1  # Next available color for cell addition if a new one is needed

        # Compute number of nodes needed
        lambda_a, lambda_b = 0, 0
        for cell in two_cell_list:
            if cell[0] > lambda_a:
                lambda_a = cell[0]
            if cell[1] > lambda_a:
                lambda_a = cell[1]
            if cell[2] > lambda_b:
                lambda_b = cell[2]
            if cell[3] > lambda_b:
                lambda_b = cell[3]
        self.M, self.N = lambda_a, lambda_b
        a_vertices = [i for i in range(1, self.M + 1)]
        b_vertices = [j for j in range(-1, -self.N - 1, -1)]
        self.add_nodes_from(a_vertices)
        self.add_nodes_from(b_vertices)  # Note that B-vertices are denoted with negative integers.

        # Add vertical edges
        for i in range(self.M + 1):
            for j in range(-1, -(self.N + 1), -1):
                self.add_edge(i, j)

        # Initialize middle link graph L_1
        self.middle_link_graph = nx.Graph()
        self.middle_link_graph.add_nodes_from([str(i) for i in range(1, self.M + 1)])
        self.middle_link_graph.add_nodes_from([str(j) for j in range(-1, -self.N - 1, -1)])
        self.middle_link_graph.add_nodes_from([str(color) + "_in" for color in range(1, self.next_color)])
        self.middle_link_graph.add_nodes_from([str(color) + "_out" for color in range(1, self.next_color)])

        # Add 2-cells and keep track of the list
        self.two_cell_list = []
        for cell in two_cell_list:
            self.add_two_cell(*cell)

        for u, v, color in self.edges.data(nbunch=a_vertices):
            if u in a_vertices and v in a_vertices:
                self.middle_link_graph.add_edge(str(v), str(color) + "_in")
                self.middle_link_graph.add_edge(str(u), str(color) + "_out")
        for u, v, color in self.edges.data(nbunch=b_vertices):
            if u in b_vertices and v in b_vertices:
                self.middle_link_graph.add_edge(str(v), str(color) + "_in")
                self.middle_link_graph.add_edge(str(u), str(color) + "_out")

    def can_add_two_cell(self, i1, j1, i2, j2):
        """
        Returns true if the given 2-cell can be added to the taiko, false otherwise.

        :param i1: the first A-coordinate
        :param j1: the first B-coordinate
        :param i2: the second A-coordinate
        :param j2: the second B-coordinate
        :return: true if the given 2-cell can be added to the taiko, false otherwise
        """
        # Convert to conventions for DiGraph
        j1 = -j1
        j2 = -j2

        # Degenerate square
        if i1 == i2 or j1 == j2:
            return False
        # Adding unecessary nodes
        elif i1 > self.M + 2 or i2 > self.M + 2 or j1 < -self.N - 2 or j2 < -self.N - 2:
            return False
        # Bipartite edges are colored
        elif self.edges[i1, j1]['color'] != 0 or self.edges[i2, j2]['color'] != 0:
            return False
        # Cannot add cell with current orientation
        elif self.has_edge(i2, i1) or self.has_edge(j2, j1):
            return False

        return True

    def add_two_cell(self, i1, j1, i2, j2):
        """
        Adds a 2-cell to the taiko.

        :param i1: the first A-coordinate
        :param j1: the first B-coordinate
        :param i2: the second A-coordinate
        :param j2: the second B-coordinate
        """
        # Add any A-vertices needed
        lambda_a = self.M
        if i1 > lambda_a:
            lambda_a = i1
        if i2 > lambda_a:
            lambda_a = i2

        for i in range(self.M + 1, lambda_a + 1):
            self.add_node(i)
            self.middle_link_graph.add_node(str(i))
            for j in range(-1, -self.N - 1, -1):
                self.add_edge(i, j)
        self.M = lambda_a

        # Add any B-vertices needed
        lambda_b = self.N
        if j1 > lambda_b:
            lambda_b = j1
        if j2 > lambda_b:
            lambda_b = j2

        for j in range(-self.N - 1, -lambda_b - 1, -1):
            self.add_node(j)
            self.middle_link_graph.add_node(str(j))
            for i in range(1, self.M):
                self.add_edge(i, j)
        self.N = lambda_b

        # Convert to conventions for DiGraph
        j1 = -j1
        j2 = -j2

        # Both horizontial edges are colored
        if self.has_edge(i1, i2) and self.has_edge(j1, j2):
            color_A = self.edges[i1, i2]['color']
            color_B = self.edges[j1, j2]['color']
            new_color = min(color_A, color_B)
            old_color = max(color_A, color_B)

            # Colors match
            if new_color == old_color:
                self.edges[i1, j1]['color'] = new_color
                self.edges[i2, j2]['color'] = new_color
            # Colors don't match
            else:
                if new_color == color_A:
                    self.edges[i1, j1]['color'] = new_color
                    self.edges[i2, j2]['color'] = new_color
                    self.edges[j1, j2]['color'] = new_color
                elif new_color == color_B:
                    self.edges[i1, j1]['color'] = new_color
                    self.edges[i2, j2]['color'] = new_color
                    self.edges[i1, i2]['color'] = new_color

                # Update all instances of old color to be new color
                for u, v, color in self.edges.data("color", default=0):
                    if color == old_color:
                        self.edges[u, v]['color'] = new_color
                        self.middle_link_graph.remove_edge(str(u), str(old_color) + "_out")
                        self.middle_link_graph.remove_edge(str(v), str(old_color) + "_in")
                        self.middle_link_graph.add_edge(str(u), str(new_color) + "_out")
                        self.middle_link_graph.add_edge(str(v), str(new_color) + "_in")
        # A-edge is colored
        elif self.has_edge(i1, i2):
            new_color = self.edges[i1, i2]['color']
            self.edges[i1, j1]['color'] = new_color
            self.edges[i2, j2]['color'] = new_color
            self.add_edge(j1, j2)
            self.edges[j1, j2]['color'] = new_color
            self.middle_link_graph.add_edge(str(j1), str(new_color) + "_out")
            self.middle_link_graph.add_edge(str(j2), str(new_color) + "_in")
        # B-edge is colored
        elif self.has_edge(j1, j2):
            new_color = self.edges[j1, j2]['color']
            self.edges[i1, j1]['color'] = new_color
            self.edges[i2, j2]['color'] = new_color
            self.add_edge(i1, i2)
            self.edges[i1, i2]['color'] = new_color
            self.middle_link_graph.add_edge(str(i1), str(new_color) + "_out")
            self.middle_link_graph.add_edge(str(i2), str(new_color) + "_in")
        # Neither edge is colored
        else:
            new_color = self.next_color
            self.add_edge(i1, i2)
            self.edges[i1, i2]['color'] = new_color
            self.middle_link_graph.add_edge(str(i1), str(new_color) + "_out")
            self.middle_link_graph.add_edge(str(i2), str(new_color) + "_in")
            self.add_edge(j1, j2)
            self.edges[j1, j2]['color'] = new_color
            self.middle_link_graph.add_edge(str(j1), str(new_color) + "_out")
            self.middle_link_graph.add_edge(str(j2), str(new_color) + "_in")
            self.edges[i1, j1]['color'] = new_color
            self.edges[i2, j2]['color'] = new_color
            self.next_color += 1

        # Update internal 2-cell list
        self.two_cell_list.append((i1, j1, i2, j2))

    def two_cell_in_taiko(self, i1, j1, i2, j2):
        """
        Returns true if the given 2-cell is in the taiko, false otherwise.

        :param i1: the first A-coordinate
        :param j1: the first B-coordinate
        :param i2: the second A-coordinate
        :param j2: the second B-coordinate
        :return: true if the given 2-cell is in the taiko, false otherwise
        """
        # Vertices don't exist
        if i1 > self.M or i2 > self.M or j1 > self.N or j2 > self.N:
            return False

        j1 = -j1
        j2 = -j2

        # Degenerate square
        if i1 == i2 or j1 == j2:
            return False
        # Bipartite edges aren't colored
        elif self.edges[i1, j1]['color'] == 0 or self.edges[i2, j2]['color'] == 0:
            return False
        # Bipartite edges don't match color
        elif self.edges[i1, j1]['color'] != self.edges[i2, j2]['color']:
            return False
        # No horizontal edges
        elif not self.has_edge(i1, i2) or not self.has_edge(j1, j2):
            return False
        # Horizontal colors don't match
        elif self.edges[i1, i2]['color'] != self.edges[j1, j2]['color']:
            return False
        # Passed checks
        else:
            return True

    def num_two_cells(self):
        """
        Returns the number of 2-cells in the taiko.

        :return: the number of 2-cells in the taiko
        """
        return len(self.two_cell_list)

    def pop_two_cell(self):
        """
        Removes the most recently added 2-cell from the taiko.

        :return: the most recently added 2-cell, as a 4-tuple of integers
        """
        # TODO speed this up by storing the most recent deleted color/edge
        output = self.two_cell_list.pop()
        self.__init__(two_cell_list=self.two_cell_list)
        return output

    def remove_two_cell(self, i1, j1, i2, j2):
        """
        Removes the given 2-cell from the taiko.

        :param i1: the first A-coordinate
        :param j1: the first B-coordinate
        :param i2: the second A-coordinate
        :param j2: the second B-coordinate
        """
        self.two_cell_list.remove((i1, j1, i2, j2))
        self.__init__(two_cell_list=self.two_cell_list)

    def is_left_aligned(self, i1, j1, i2, j2):
        """
        Takes as input a 2-cell (i1, j1, i2, j2).

        Returns true if the given 2-cell is left-aligned, false otherwise.

        Based on left-alignment as described in Section 3.2 of Garg-Minyev.

        :param i1: the first A-coordinate
        :param j1: the first B-coordinate
        :param i2: the second A-coordinate
        :param j2: the second B-coordinate
        :return: true if the given 2-cell is left-aligned, false otherwise
        """
        if i1 > self.M + 1:
            return False
        elif i2 > self.M + 1 and i2 > i1 + 1:
            return False

        if j1 > self.N + 1:
            return False
        elif j2 > self.N + 1 and j2 > j1 + 1:
            return False

        return True

    def no_fold(self):
        """
        Returns true if the taiko satisfies the no-fold condition, false otherwise.

        :return: true if the taiko satisfies the no-fold condition, false otherwise
        """
        used_colors_in = set()
        used_colors_out = set()

        # Find folds in L_A
        for i1 in range(1, self.M + 1):
            used_colors_in.clear()
            used_colors_out.clear()

            # Check vertex i1 for folds
            for u, v, color in self.in_edges(nbunch=i1, data=True):
                if u >= 1:
                    if color in used_colors_in:
                        return False
                    used_colors_in.add(color)

            for u, v, color in self.out_edges(nbunch=i1, data=True):
                if v >= 1:
                    if color in used_colors_out:
                        return False
                    used_colors_out.add(color)

        # Find folds in L_B
        for j1 in range(-1, -(self.N + 1), -1):
            used_colors_in.clear()
            used_colors_out.clear()

            # Check vertex j1 for folds
            for u, v, color in self.in_edges(nbunch=j1, data=True):
                if u <= -1:
                    if color in used_colors_in:
                        return False
                    used_colors_in.add(color)

            for u, v, color in self.out_edges(nbunch=j1, data=True):
                if v <= -1:
                    if color in used_colors_out:
                        return False
                    used_colors_out.add(color)

        return True

    @staticmethod
    def girth_at_least_n(graph, n):
        """
        Returns true if the girth of the given (undirected) graph is at least n, false otherwise.

        :param graph: the graph, directed graphs will be treated as undirected
        :param n: a positive integer >= 3
        :return: true if the girth of the given graph is at least n, false otherwise
        """
        for source_vertex in graph.nodes():
            layers = nx.bfs_layers(graph, source_vertex)
            next(layers)  # Distance 0
            next(layers)  # Distance 1
            for depth in range(3, n):
                # Distance depth-1 vertices
                for vertex in next(layers):
                    if source_vertex in graph.neighbors(vertex):
                        return False
        return True

    def is_girth_p_q(self, p, q):
        """
        Returns true if the taiko satisfies the girth(p,q) condition, false otherwise.

        :param p: a positive integer >= 3
        :param q: a positive integer
        :return: true if the taiko satisfies the girth(p,q), false otherwise
        """
        a_vertices = [i for i in range(1, self.M + 1)]
        b_vertices = [j for j in range(-1, -self.N - 1, -1)]

        L_A = self.subgraph(a_vertices).to_undirected()
        L_B = self.subgraph(b_vertices).to_undirected()
        return Taiko.girth_at_least_n(L_A, p) and Taiko.girth_at_least_n(L_B, p) and Taiko.girth_at_least_n(
            self.middle_link_graph, 2 * q)

    def layout(self):
        """
        Returns a layout for this taiko to be plotted.

        :return: a layout for this taiko to be plotted.
        """
        return nx.bipartite_layout(self, nodes=[j for j in range(-1, -(self.N + 1), -1)], align='horizontal')
