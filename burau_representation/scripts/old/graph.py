import itertools
import networkx as nx
from collections import defaultdict


def all_partitions(vertices):
    """Generate all unique partitions of a set of vertices."""
    if len(vertices) == 0:
        return [[]]
    
    first = vertices[0]
    rest_partitions = all_partitions(vertices[1:])
    
    unique_partitions = []
    for partition in rest_partitions:
        # Create a new partition with `first` as a separate subset
        unique_partitions.append([[first]] + partition)
        
        # Try adding `first` to existing subsets
        for i in range(len(partition)):
            new_partition = [subset if j != i else subset + [first] for j, subset in enumerate(partition)]
            unique_partitions.append(new_partition)
    
    return unique_partitions

def partition_and_merge_edges(graph, partition):
    """
    Merges edges with the same label in a given partitioned graph.
    Returns a new labeled graph after merging.
    """
    new_graph = nx.MultiDiGraph()
    vertex_map = {}
    
    # Assign new vertex numbers based on partition
    for new_v, group in enumerate(partition):
        for v in group:
            vertex_map[v] = new_v
    
    # Merge edges with the same label
    edge_dict = defaultdict(set)
    for u, v, data in graph.edges(data=True):
        new_u, new_v = vertex_map[u], vertex_map[v]
        label = data['label']
        edge_dict[(new_u, new_v, label)].add((u, v))
    
    # Add merged edges to new graph
    for (new_u, new_v, label), edges in edge_dict.items():
        new_graph.add_edge(new_u, new_v, label=label)
    
    return new_graph

def merge_edges(graph):
    """
    Merges multiple edges with the same label between the same pair of nodes.
    Returns a new MultiDiGraph with merged edges.
    """
    new_graph = nx.MultiDiGraph()
    
    # Copy all nodes
    new_graph.add_nodes_from(graph.nodes(data=True))

    # Dictionary to store merged edges
    edge_dict = defaultdict(set)

    for u, v, data in graph.edges(data=True):
        label = data.get('label', None)  # Ensure we handle unlabeled edges safely
        edge_dict[(u, v, label)].add((u, v))

    # Add merged edges to the new graph
    for (u, v, label), edges in edge_dict.items():
        new_graph.add_edge(u, v, label=label)

    return new_graph


def turn_to_deterministic(graph):
    """
    Transforms a labeled directed graph into a deterministic one.
    If a vertex has multiple outgoing edges with the same label,
    it merges the target vertices of those edges.
    
    Parameters:
        graph (nx.MultiDiGraph): The input labeled graph.

    Returns:
        nx.MultiDiGraph: The transformed deterministic graph.
    """
    new_graph = nx.MultiDiGraph()
    vertex_map = {v: {v} for v in graph.nodes()}  # Maps each vertex to its equivalence class

    # Check if needed
    t = 0 
    for v in graph.nodes():
        label_to_targets = defaultdict(set)
        
        for _, target, data in graph.out_edges(v, data=True):
            label = data['label']
            label_to_targets[label].add(target)
        
        # Merge target vertices if there are multiple for the same label
        for label, targets in label_to_targets.items():
            if len(targets) > 1:
                t += 1

    if t == 0:
        return graph
                
    # Find conflicting edges
    for v in graph.nodes():
        label_to_targets = defaultdict(set)
        
        for _, target, data in graph.out_edges(v, data=True):
            label = data['label']
            label_to_targets[label].add(target)
        
        # Merge target vertices if there are multiple for the same label
        for label, targets in label_to_targets.items():
            if len(targets) > 1:
                # Pick a representative vertex
                rep = min(targets)  # Choose the smallest vertex index
                for t in targets:
                    vertex_map[rep] |= vertex_map[t]  # Merge sets
                    vertex_map[t] = vertex_map[rep]  # Update mapping


    # Assign new vertex indices
    merged_vertices = {}
    for v, merged_set in vertex_map.items():
        merged_set = frozenset(merged_set)
        if merged_set not in merged_vertices:
            merged_vertices[merged_set] = len(merged_vertices)  # Assign new index

    # Rebuild the deterministic graph
    for u, v, data in graph.edges(data=True):
        new_u = merged_vertices[frozenset(vertex_map[u])]
        new_v = merged_vertices[frozenset(vertex_map[v])]
        new_graph.add_edge(new_u, new_v, label=data['label'])

    return turn_to_deterministic(new_graph)


def find_spanning_tree(graph, root=0):
    """Finds a spanning tree of the directed graph using BFS."""
    tree = nx.DiGraph()
    visited = set()
    queue = [root]
    parent = {root: None}
    
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        for neighbor in graph.successors(node):
            if neighbor not in visited:
                tree.add_edge(node, neighbor, label=graph[node][neighbor][0]['label'])
                parent[neighbor] = node
                queue.append(neighbor)
    
    return tree, parent

def compute_generators(graph, spanning_tree, parent_map):
    """
    Computes generators for the subgroup based on non-tree edges.
    """
    generators = []
    
    def get_path(v):
        path = []
        while parent_map[v] is not None:
            p = parent_map[v]
            if graph.has_edge(p, v):
                # Fetch the first edge label (assuming MultiDiGraph with possible multiple edges)
                label = list(graph[p][v].values())[0]['label']
                path.append(label)
            v = p
        return path
    
    for u, v, data in graph.edges(data=True):
        if not spanning_tree.has_edge(u, v):
            path_start = get_path(u)
            path_end = get_path(v)[::-1]  # Reverse path to end
            edge_label = [data['label']]
            generator = path_start + edge_label + [-l for l in path_end]  # Use negative integers for inversion
            generators.append(generator)
        elif spanning_tree[u][v]['label'] != data['label']:
            path_start = get_path(u)
            path_end = get_path(v)[::-1]  # Reverse path to end
            edge_label = [data['label']]
            generator = path_start + edge_label + [-l for l in path_end]  # Use negative integers for inversion
            generators.append(generator)
            
    
    return generators


def make_automaton(generators):
    G = nx.DiGraph()
    num_vertices = 1
    for i in range(len(generators)):
        for j in range(len(generators[i])):
            if len(generators[i]) == 1:
                G.add_edge(0,0, label = generators[i][j])   
            else:
                if j == 0:
                    G.add_edge(0,num_vertices, label = generators[i][j])
                    num_vertices += 1
                elif j == len(generators[i])-1:
                    G.add_edge(num_vertices-1,0, label = generators[i][j])
                else:
                    G.add_edge(num_vertices-1,num_vertices, label = generators[i][j])
                    num_vertices += 1
    return G

def express_in_graph(subgroup,graph,spanning_tree):
    return 0


def complete_to_cover(graph,k):
    vertices = {}
    for u in graph.nodes:
        vertices[u] = []
    for u, v, data in graph.edges(data=True):
        vertices[u].append(data["label"])
        vertices[v].append(-1*data["label"])
    
    for m in range(1,k+1):
        for i in range(len(vertices)):
            if m not in vertices[i]:
                for j in range(len(vertices)):
                    if -m not in vertices[j]:
                        graph.add_edge(i,j, label = m)
                        vertices[i].append(m)
                        vertices[j].append(-m)
                        break


    return graph

def find_path_in_automaton(graph, word):
    """
    Finds the path corresponding to a word in a deterministic automaton (NetworkX graph).

    :param graph: NetworkX directed graph (DiGraph) with labeled edges.
    :param start_state: The initial state (node).
    :param word: The input word (string).
    :return: List of visited states (path) if valid, otherwise None.
    """
    current_state = 0
    path = [current_state]  # Start from the initial state

    for letter in word:
        found = False
        for _, next_state, data in graph.out_edges(current_state, data=True):
            if data.get('label') == letter:  # Check label on the edge
                path.append(next_state)
                current_state = next_state
                found = True
                break  # Move to the next letter
        if not found:
            return None  # No valid transition

    return path
            

# Example Usage
G = nx.MultiDiGraph()
G.add_edge(0, 1, label = 1)
G.add_edge(1, 2, label = 1)
G.add_edge(2, 0, label = 1)
G.add_edge(0, 3, label = 2)
G.add_edge(3, 4, label = 2)
G.add_edge(4, 0, label = 2)


vertices = list(G.nodes())
partitions = list(all_partitions(vertices))
print(len(partitions))
# Store all sets of generators
all_generators = []

# Iterate over each partition and process it
for partition in partitions:
    merged_graph = merge_edges(turn_to_deterministic(partition_and_merge_edges(G, partition)))
    graph = merged_graph
    spanning_tree, parent_map = find_spanning_tree(graph)

    generators = compute_generators(graph, spanning_tree, parent_map)
    print(partition)
    print("\n")
    print(generators)
    print("\n")
    print("\n")
    all_generators.append(generators)

unique_generators = []
for gen in all_generators:
    if gen not in unique_generators:
        unique_generators.append(gen)

print(len(unique_generators))


