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
print(all_partitions([0,1,2]))
# Store all sets of generators
all_generators = []

# Iterate over each partition and process it
for partition in partitions:
    merged_graph = partition_and_merge_edges(G, partition)
    spanning_tree, parent_map = find_spanning_tree(merged_graph)
    generators = compute_generators(merged_graph, spanning_tree, parent_map)

    all_generators.append(generators)


