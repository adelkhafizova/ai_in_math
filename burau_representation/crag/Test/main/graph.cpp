#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>

using namespace boost;
using namespace std;

// Graph definition
struct EdgeLabel {
    int label;
};

typedef adjacency_list<vecS, vecS, directedS, no_property, EdgeLabel> Graph;
typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef graph_traits<Graph>::edge_descriptor Edge;

typedef pair<int, int> EdgeKey;

// Function to generate all partitions of a set
vector<vector<vector<int>>> all_partitions(const vector<int>& vertices) {
    if (vertices.empty()) return { { } };
    
    vector<vector<vector<int>>> result;
    int first = vertices[0];
    vector<int> rest(vertices.begin() + 1, vertices.end());
    auto smaller_partitions = all_partitions(rest);
    
    for (const auto& partition : smaller_partitions) {
        vector<vector<int>> new_partition = partition;
        new_partition.insert(new_partition.begin(), { first });
        result.push_back(new_partition);
        
        for (size_t i = 0; i < partition.size(); ++i) {
            vector<vector<int>> merged_partition = partition;
            merged_partition[i].push_back(first);
            result.push_back(merged_partition);
        }
    }
    return result;
}

// Function to merge edges with the same label based on partition
Graph partition_and_merge_edges(const Graph& graph, const vector<vector<int>>& partition) {
    Graph new_graph;
    map<int, int> vertex_map;
    
    for (size_t i = 0; i < partition.size(); ++i) {
        for (int v : partition[i]) {
            vertex_map[v] = i;
        }
    }
    
    map<tuple<int, int, int>, set<EdgeKey>> edge_dict;
    
    graph_traits<Graph>::edge_iterator ei, ei_end;
    for (tie(ei, ei_end) = edges(graph); ei != ei_end; ++ei) {
        int u = source(*ei, graph);
        int v = target(*ei, graph);
        int new_u = vertex_map[u];
        int new_v = vertex_map[v];
        int label = graph[*ei].label;
        edge_dict[make_tuple(new_u, new_v, label)].insert({u, v});
    }
    
    for (const auto& entry : edge_dict) {
        int new_u = get<0>(entry.first);
        int new_v = get<1>(entry.first);
        int label = get<2>(entry.first);
        add_edge(new_u, new_v, EdgeLabel{label}, new_graph);
    }
    
    return new_graph;
}

// Function to find a spanning tree using BFS
Graph find_spanning_tree(const Graph& graph, int root, map<int, int>& parent_map) {
    Graph tree;
    set<int> visited;
    queue<int> q;
    q.push(root);
    parent_map[root] = -1;
    
    while (!q.empty()) {
        int node = q.front(); q.pop();
        if (visited.count(node)) continue;
        visited.insert(node);
        
        graph_traits<Graph>::out_edge_iterator ei, ei_end;
        for (tie(ei, ei_end) = out_edges(node, graph); ei != ei_end; ++ei) {
            int neighbor = target(*ei, graph);
            if (!visited.count(neighbor)) {
                add_edge(node, neighbor, EdgeLabel{graph[*ei].label}, tree);
                parent_map[neighbor] = node;
                q.push(neighbor);
            }
        }
    }
    return tree;
}

// Function to compute generators
vector<vector<int>> compute_generators(const Graph& graph, const Graph& spanning_tree, const map<int, int>& parent_map) {
    vector<vector<int>> generators;
    
    auto get_path = [&](int v) {
        vector<int> path;
        while (parent_map.at(v) != -1) {
            int p = parent_map.at(v);
            graph_traits<Graph>::out_edge_iterator ei, ei_end;
            for (tie(ei, ei_end) = out_edges(p, graph); ei != ei_end; ++ei) {
                if (target(*ei, graph) == v) {
                    path.push_back(graph[*ei].label);
                    break;
                }
            }
            v = p;
        }
        return path;
    };
    
    graph_traits<Graph>::edge_iterator ei, ei_end;
    for (tie(ei, ei_end) = edges(graph); ei != ei_end; ++ei) {
        int u = source(*ei, graph);
        int v = target(*ei, graph);
        
        if (!edge(u, v, spanning_tree).second) {
            vector<int> path_start = get_path(u);
            vector<int> path_end = get_path(v);
            reverse(path_end.begin(), path_end.end());
            vector<int> generator = path_start;
            generator.push_back(graph[*ei].label);
            for (int& l : path_end) l = -l;
            generator.insert(generator.end(), path_end.begin(), path_end.end());
            generators.push_back(generator);
        }
    }
    return generators;
}

int main() {
    Graph G;
    add_edge(0, 1, EdgeLabel{1}, G);
    add_edge(1, 2, EdgeLabel{1}, G);
    add_edge(2, 0, EdgeLabel{1}, G);
    add_edge(0, 3, EdgeLabel{2}, G);
    add_edge(3, 4, EdgeLabel{2}, G);
    add_edge(4, 0, EdgeLabel{2}, G);
    
    vector<int> vertices = {0, 1, 2, 3, 4};
    auto partitions = all_partitions(vertices);
    cout << "Total partitions: " << partitions.size() << endl;
    
    vector<vector<vector<int>>> all_generators;
    for (const auto& partition : partitions) {
        Graph merged_graph = partition_and_merge_edges(G, partition);
        map<int, int> parent_map;
        Graph spanning_tree = find_spanning_tree(merged_graph, 0, parent_map);
        vector<vector<int>> generators = compute_generators(merged_graph, spanning_tree, parent_map);
        all_generators.push_back(generators);
    }
    
    cout << "Generated " << all_generators.size() << " sets of generators." << endl;
    return 0;
}