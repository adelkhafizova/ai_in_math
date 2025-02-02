using DataStructures
using Graphs
using MetaGraphsNext
using IterTools
using StaticArrays

include("constants.jl")

const M = 8
const N = 5
const MAX_GLUING_NUM = floor(Int, (M*N)/2) # Number of 2-cells in a refined taiko, also the maximum possible gluing number of a horizontal edge.
const WEIGHT_FUNC = edge_data -> 1.0

# Compute index maps between indices and vertical edges:
function index_to_edge()::MVector{M*N, Pair{Int,Int}}
    map = MVector{M*N, Pair{Int,Int}}(0 => 0 for _ in 1:M*N)
    index = 1
    for i in 1:M
        for j in 1:N
            map[index] = i => j
            index += 1
        end
    end
    return map
end

function edge_to_index()::Dict{Pair{Int,Int}, Int}
    map = Dict{Pair{Int,Int}, Int}()
    index = 1
    for i in 1:M
        for j in 1:N
            map[i => j] = index
            index += 1
        end
    end
    return map
end

const INDEX_TO_EDGE = index_to_edge()
const EDGE_TO_INDEX = edge_to_index()

"""
Setup based on the algorithm described in Section 3 of the recent paper by Garg and Minyev: https://arxiv.org/abs/2501.07646
"""

function graph_error(error_msg::String, problem_cell=0::Int)::MetaGraph{Int64, SimpleGraph{Int64}, String, Int64, MVector{2, Int64}, String, typeof(WEIGHT_FUNC), Float64}
    """
    Returns a MetaGraph indicating an error occurred with generating a colored oriented taiko.

    The error message stores the index of the problem 2-cell, if any.
    """
    return MetaGraph(
        Graph();
        label_type=String,
        vertex_data_type=Int,
        edge_data_type=MVector{2, Int64},
        graph_data=error_msg,
        weight_function=WEIGHT_FUNC
    )
end

function convert_edges_to_two_cell(e1_index::Int, e2_index::Int)::SVector{5,Int}
    e1 = INDEX_TO_EDGE[e1_index]
    e2 = INDEX_TO_EDGE[e2_index]
    i1 = e1[1]
    j1 = e1[2]
    i2 = e2[1]
    j2 = e2[2]
    orientation = 1
    if i1 > i2
        i1, i2 = i2, i1
        j1, j2 = j2, j1
        orientation = -1
    end

    return SVector{5,Int}(i1, i2, j1, j2, orientation)
end

function convert_two_cell_to_edges(i1::Int, i2::Int, j1::Int, j2::Int, orientation::Int)::Pair{Int,Int}
    e1_index = EDGE_TO_INDEX[i1 => j1]
    e2_index = EDGE_TO_INDEX[i2 => j2]

    if orientation < 0
        e1_index, e2_index = e2_index, e1_index
    end

    return e1_index => e2_index
end

function convert_string_to_graph(graph_str::String)::MetaGraph{Int64, SimpleGraph{Int64}, String, Int64, MVector{2, Int64}, String, typeof(WEIGHT_FUNC), Float64}
    """
    Helper function to convert partitions (in string format) into digraphs.

    Graph is implemented as an undirected MetaGraph. Vertex labels are the same as in the paper (e.g. strings "a1" and "b1")
    and vertex data is given as an int. The vertex "a1" contains an int corresponding to the next available new color when adding 2-cells.
    Edges are labelled as 2-tuples of strings with data values given as 2-tuples of signed integers: the first gives the color and orientation,
    the second gives the number of 2-cells using this edge.

    If the given string results in a non-orientable or non-colorable taiko, returns an error graph so the local search will reject it later.
    Otherwise, returns the full colored and oriented taiko.
    """
    # Parse string and verify input length
    two_cells = parse.(Int, split(graph_str, r"[,()]", keepempty=false))
    if size(two_cells, 1) % 2 != 0
        return graph_error("input error")
    elseif size(two_cells, 1) / 2 > MAX_GLUING_NUM
        return graph_error("input error")
    elseif any(x -> x < 1 || x > M*N, two_cells)
        return graph_error("input error")
    end

    # Intialize vertex label-data pairs
    vertices_description = Array{Pair{String, Int}, 1}(undef, M+N)
    for i in 1:M
        vertices_description[i] = "a" * string(i) => 1
    end

    for j in 1:N
        vertices_description[j+M] = "b" * string(j) => 1
    end

    # Initialize edge label-data pairs
    edges_description = Array{Pair{Tuple{String,String}, MVector{2, Int}}}(undef, M*N)
    for i in 1:M
        for j in 1:N
            edges_description[(i-1)*N + j] = ("a" * string(i), "b" * string(j)) => MVector(0, 1)
        end
    end

    graph = MetaGraph(complete_bipartite_graph(M,N), vertices_description, edges_description, "color+orientation",WEIGHT_FUNC)

    # Add edges to form horizontal graphs L_A and L_B

    # Check valid 2-cells
    for (e1_index, e2_index) in partition(two_cells, 2)
        i1, i2, j1, j2, orientation = convert_edges_to_two_cell(e1_index, e2_index)

        # 2-cell is in taiko but violates partition condition
        if graph[label_for(graph, i1),label_for(graph, j1+M)][1] != 0 || graph[label_for(graph, i2),label_for(graph, j2+M)][1] != 0
            return graph_error("partition error at ($(e1_index), $(e2_index))")
        # 2-cell is in taiko, orientation-related error
        elseif !can_add_two_cell(graph, i1, i2, j1, j2, orientation)
            return graph_error("orientation error at ($(e1_index), $(e2_index))")
        # Can add two cell, so adds it.
        else
            add_two_cell!(graph, i1, i2, j1, j2, orientation)
        end
    end

    return graph
end

function convert_graph_to_string(graph::MetaGraph{Int64, SimpleGraph{Int64}, String, Int64, MVector{2, Int64}, String, typeof(WEIGHT_FUNC), Float64})::String
    """
    Helper function to convert graphs into partitions (in string format).

    Partitions are represented as strings of integer pairs separated by commas. The
    string lists the 2-cells, where each integer corresponds to a unique vertical edge and each pair of integers represents a
    2-cell connecting the two vertical edges (ai1, bj1) and (ai2, bj2). The orientation is positive if i1<i2 and negative otherwise.

    Note that we can take a graph at any stage of the greedy search in here (even with unrefined partitions).

    However, any graphs that violate the partition or orientation conditions will be rejected here.
    """

    if occursin(graph[], "error")
        return "error"
    end

    entries = []
    for i1 in 1:M
        for i2 in (i1+1):M
            for j1 in 1:N
                for j2 in 1:N
                    if j1 != j2 && two_cell_in_taiko(graph, i1, i2, j1, j2)
                        if graph[label_for(graph, i1), label_for(graph, i2)][1] > 0
                            push!(entries, "($(EDGE_TO_INDEX[i1 => j1]),")
                            push!(entries, "$(EDGE_TO_INDEX[i2 => j2])),")
                        else
                            push!(entries, "($(EDGE_TO_INDEX[i2 => j2]),")
                            push!(entries, "$(EDGE_TO_INDEX[i1 => j1])),")
                        end
                    end
                end
            end
        end
    end

    return chop(join(entries))
end

function update_graph!(graph::MetaGraph{Int64, SimpleGraph{Int64}, String, Int64, MVector{2, Int64}, String, typeof(WEIGHT_FUNC), Float64}, two_cell_list::Vector{SVector{5,Int}})::Bool
    # Remove all horizontal edges
    for i1 in 1:M
        for i2 in (i1+1):M
            if has_edge(graph, i1, i2)
                rem_edge!(graph, i1, i2)
            end
        end
    end

    for j1 in 1:N
        for j2 in (j1+1):N
            if has_edge(graph, j1+M, j2+M)
                rem_edge!(graph, j1+M, j2+M)
            end
        end
    end

    # Decolor all vertical edges
    for i in 1:M
        for j in 1:N
            graph[label_for(graph, i), label_for(graph, j + M)][1] = 0
        end
    end

    # Add 2-cells according to list
    for two_cell in two_cell_list
        i1, i2, j1, j2, orientation = two_cell[1], two_cell[2], two_cell[3], two_cell[4], two_cell[5]

        # 2-cell is in taiko but violates partition condition
        if graph[label_for(graph, i1),label_for(graph, j1+M)][1] != 0 || graph[label_for(graph, i2),label_for(graph, j2+M)][1] != 0
            return false
        # 2-cell is in taiko, orientation-related error
        elseif !can_add_two_cell(graph, i1, i2, j1, j2, orientation)
            return false
        # Can add two cell, so adds it.
        else
            add_two_cell!(graph, i1, i2, j1, j2, orientation)
        end
    end

    return true
end

function can_add_two_cell(graph::MetaGraph{Int64, SimpleGraph{Int64}, String, Int64, MVector{2, Int64}, String, typeof(WEIGHT_FUNC), Float64}, i1::Int, i2::Int, j1::Int, j2::Int, orientation::Int)::Bool
    """
    Returns true if the given 2-cell with given orientation can be added to the taiko, false otherwise.

    Note that we always assume i1 < i2 when writing the vertices of the 2-cell.
    """
    # Degenerate square
    if i1 >= i2 || j1 == j2
        return false
    # Bipartite edges are colored
    elseif graph[label_for(graph, i1), label_for(graph, j1+M)][1] != 0 || graph[label_for(graph, i2), label_for(graph, j2+M)][1] != 0
        return false
    end


    # b graph orientation same
    if j1 < j2
        # Both horizontal edges filled in, check if orientation match and return falseif otherwise
        if has_edge(graph, i1, i2) && has_edge(graph, j1 + M, j2 + M)
            if sign(graph[label_for(graph, i1),label_for(graph, i2)][1]) != sign(graph[label_for(graph, j1+M),label_for(graph, j2+M)][1])
                return false
            elseif orientation != sign(graph[label_for(graph, i1),label_for(graph, i2)][1])
                return false
            else
                return true
            end
        # Exactly one horizontal edge filled in.
        # If orientation does not match with the given string, return false.
        elseif has_edge(graph, i1, i2)
            if orientation != sign(graph[label_for(graph, i1),label_for(graph, i2)][1])
                return false
            else
                return true
            end
        elseif has_edge(graph, j1 + M, j2 + M)
            if orientation != sign(graph[label_for(graph, j1+M),label_for(graph, j2+M)][1])
                return false
            else
                return true
            end
        # Neither edge filled in
        else
            return true
        end
    # b graph orientation reversed
    else
        # Both horizontal edges filled in, check if orientation match and return false if otherwise
        if has_edge(graph, i1, i2) && has_edge(graph, j1 + M, j2 + M)
            if sign(graph[label_for(graph, i1),label_for(graph, i2)][1]) != -1 * sign(graph[label_for(graph, j1+M),label_for(graph, j2+M)][1])
                return false
            elseif orientation != sign(graph[label_for(graph, i1),label_for(graph, i2)][1])
                return false
            else
                return true
            end
        # Exactly one horizontal edge filled in.
        # If orientation does not match with the given string, return false.
        elseif has_edge(graph, i1, i2)
            if orientation != sign(graph[label_for(graph, i1),label_for(graph, i2)][1])
                return false
            else
                return true
            end
        elseif has_edge(graph, j1 + M, j2 + M)
            if orientation != -1 * sign(graph[label_for(graph, j1+M),label_for(graph, j2+M)][1])
                return false
            else
                return true
            end
        # Neither edge filled in.
        else
            return true
        end
    end
end

function add_two_cell!(graph::MetaGraph{Int64, SimpleGraph{Int64}, String, Int64, MVector{2, Int64}, String, typeof(WEIGHT_FUNC), Float64}, i1::Int, i2::Int, j1::Int, j2::Int, orientation::Int)::Bool
    """
    Attempts to add the given oriented 2-cell. Returns true if successful, false otherwise.

    Note: In general, adding a 2-cell is NOT invertible. Whenever adding a 2-cell causes other horizontal edges to be recolored,
    we cannot undo adding the 2-cell.
    """
    # 2-cell is in taiko, b graph orientation same
    if j1 < j2
        # Both horizontal edges filled in, check if orientation match and throw error if otherwise
        if has_edge(graph, i1, i2) && has_edge(graph, j1 + M, j2 + M)
            # Colors don't match, update rest of structure to use smaller color
            if graph[label_for(graph, i1),label_for(graph, i2)][1] != graph[label_for(graph, j1+M),label_for(graph, j2+M)][1]
                new_color = min(abs(graph[label_for(graph, i1),label_for(graph, i2)][1]), abs(graph[label_for(graph, j1+M),label_for(graph, j2+M)][1]))
                old_color = max(abs(graph[label_for(graph, i1),label_for(graph, i2)][1]), abs(graph[label_for(graph, j1+M),label_for(graph, j2+M)][1]))

                for edge in edge_labels(graph)
                    if abs(graph[edge[1], edge[2]][1]) == old_color
                        sign_mod = sign(graph[edge[1], edge[2]][1])
                        graph[edge[1], edge[2]][1] = new_color * sign_mod
                    end
                end
                graph[label_for(graph, i1),label_for(graph, i2)][2] += 1
                graph[label_for(graph, j1+M),label_for(graph, j2+M)][2] += 1
                graph[label_for(graph, i1),label_for(graph, j1+M)][1] = new_color
                graph[label_for(graph, i2),label_for(graph, j2+M)][1] = new_color
            else
                assign_color = abs(graph[label_for(graph, i1),label_for(graph, i2)][1])
                graph[label_for(graph, i1),label_for(graph, i2)][2] += 1
                graph[label_for(graph, j1+M),label_for(graph, j2+M)][2] += 1
                graph[label_for(graph, i1),label_for(graph, j1+M)][1] = assign_color
                graph[label_for(graph, i2),label_for(graph, j2+M)][1] = assign_color
            end
        # Exactly one horizontal edge filled in, make other horizontal edge match color.
        elseif has_edge(graph, i1, i2)
            assign_color = abs(graph[label_for(graph, i1),label_for(graph, i2)][1])
            graph[label_for(graph, i1),label_for(graph, i2)][2] += 1
            graph[label_for(graph, j1+M),label_for(graph, j2+M)] = MVector(graph[label_for(graph, i1),label_for(graph, i2)][1], 1)
            graph[label_for(graph, i1),label_for(graph, j1+M)][1] = assign_color
            graph[label_for(graph, i2),label_for(graph, j2+M)][1] = assign_color
        elseif has_edge(graph, j1 + M, j2 + M)
            assign_color = abs(graph[label_for(graph, j1+M),label_for(graph, j2+M)][1])
            graph[label_for(graph, i1),label_for(graph, i2)] = MVector(graph[label_for(graph, j1+M),label_for(graph, j2+M)][1], 1)
            graph[label_for(graph, j1+M),label_for(graph, j2+M)][2] += 1
            graph[label_for(graph, i1),label_for(graph, j1+M)][1] = assign_color
            graph[label_for(graph, i2),label_for(graph, j2+M)][1] = assign_color
        # Neither edge filled in, update both with a new color.
        else
            # Find an available color by taking the maximum color in the taiko and adding 1.
            assign_color = graph["a1"]
            graph[label_for(graph, i1),label_for(graph, i2)] = MVector(assign_color * orientation, 1)
            graph[label_for(graph, j1+M),label_for(graph, j2+M)] = MVector(assign_color * orientation, 1)
            graph[label_for(graph, i1),label_for(graph, j1+M)][1] = assign_color
            graph[label_for(graph, i2),label_for(graph, j2+M)][1] = assign_color

            graph["a1"] += 1
        end
    # 2-cell is in taiko, b graph orientation reversed
    else
        # Both horizontal edges filled in, check if color/orientation match and throw error if otherwise
        if has_edge(graph, i1, i2) && has_edge(graph, j1 + M, j2 + M)
            # Colors don't match, update rest of structure to use smaller color
            if graph[label_for(graph, i1),label_for(graph, i2)][1] != -1 * graph[label_for(graph, j1+M),label_for(graph, j2+M)][1]
                new_color = min(abs(graph[label_for(graph, i1),label_for(graph, i2)][1]), abs(graph[label_for(graph, j1+M),label_for(graph, j2+M)][1]))
                old_color = max(abs(graph[label_for(graph, i1),label_for(graph, i2)][1]), abs(graph[label_for(graph, j1+M),label_for(graph, j2+M)][1]))

                for edge in edge_labels(graph)
                    if abs(graph[edge[1], edge[2]][1]) == old_color
                        sign_mod = sign(graph[edge[1], edge[2]][1])
                        graph[edge[1], edge[2]][1] = sign_mod * new_color
                    end
                end

                graph[label_for(graph, i1),label_for(graph, i2)][2] += 1
                graph[label_for(graph, j1+M),label_for(graph, j2+M)][2] += 1
                graph[label_for(graph, i1),label_for(graph, j1+M)][1] = new_color
                graph[label_for(graph, i2),label_for(graph, j2+M)][1] = new_color
            else
                assign_color = abs(graph[label_for(graph, i1),label_for(graph, i2)][1])
                graph[label_for(graph, i1),label_for(graph, i2)][2] += 1
                graph[label_for(graph, j1+M),label_for(graph, j2+M)][2] += 1
                graph[label_for(graph, i1),label_for(graph, j1+M)][1] = assign_color
                graph[label_for(graph, i2),label_for(graph, j2+M)][1] = assign_color
            end
        # Exactly one horizontal edge filled in, make other horizontal edge match color.
        # If orientation does not match with the given string, throw error.
        elseif has_edge(graph, i1, i2)
            assign_color = abs(graph[label_for(graph, i1),label_for(graph, i2)][1])
            graph[label_for(graph, i1),label_for(graph, i2)][2] += 1
            graph[label_for(graph, j1+M),label_for(graph, j2+M)] = MVector(-1 * graph[label_for(graph, i1),label_for(graph, i2)][1], 1)
            graph[label_for(graph, i1),label_for(graph, j1+M)][1] = assign_color
            graph[label_for(graph, i2),label_for(graph, j2+M)][1] = assign_color
        elseif has_edge(graph, j1 + M, j2 + M)
            assign_color = abs(graph[label_for(graph, j1+M),label_for(graph, j2+M)][1])
            graph[label_for(graph, i1),label_for(graph, i2)] = MVector(-1 * graph[label_for(graph, j1+M),label_for(graph, j2+M)][1], 1)
            graph[label_for(graph, j1+M),label_for(graph, j2+M)][2] += 1
            graph[label_for(graph, i1),label_for(graph, j1+M)][1] = assign_color
            graph[label_for(graph, i2),label_for(graph, j2+M)][1] = assign_color
        # Neither edge filled in, update both with a new color.
        else
            # Find an available color by taking the maximum color in the taiko and adding 1.
            assign_color = graph["a1"]
            graph[label_for(graph, i1),label_for(graph, i2)] = MVector(assign_color * orientation, 1)
            graph[label_for(graph, j1+M),label_for(graph, j2+M)] = MVector(-1 * assign_color * orientation, 1)
            graph[label_for(graph, i1),label_for(graph, j1+M)][1] = assign_color
            graph[label_for(graph, i2),label_for(graph, j2+M)][1] = assign_color
            graph["a1"] += 1
        end
    end

    return true
end

function two_cell_in_taiko(graph::MetaGraph{Int64, SimpleGraph{Int64}, String, Int64, MVector{2, Int64}, String, typeof(WEIGHT_FUNC), Float64}, i1::Int, i2::Int, j1::Int, j2::Int)::Bool
    """
    Returns true if valid oriented 2-cell currently in the taiko, false otherwise.
    """

    # Degenerate square
    if i1 == i2 || j1 == j2
        return false
    # Bipartite edges aren't colored
    elseif graph[label_for(graph, i1), label_for(graph, j1+M)][1] == 0 || graph[label_for(graph, i2), label_for(graph, j2+M)][1] == 0
        return false
    # Bipartite edges don't match color
    elseif graph[label_for(graph, i1), label_for(graph, j1+M)][1] != graph[label_for(graph, i2), label_for(graph, j2+M)][1]
        return false
    # No horizontal edges
    elseif !(has_edge(graph, i1, i2) && has_edge(graph, j1+M, j2+M))
        return false
    end

    # Horizontal colors/orientation don't match
    if j1 < j2 && (graph[label_for(graph, i1), label_for(graph, i2)][1] != graph[label_for(graph, j1+M), label_for(graph, j2+M)][1])
        return false
    elseif j2 < j1 && (graph[label_for(graph, i1), label_for(graph, i2)][1] != -1 * graph[label_for(graph, j1+M), label_for(graph, j2+M)][1])
        return false
    end

    # Passed checks
    return true
end

function remove_two_cell!(graph::MetaGraph{Int64, SimpleGraph{Int64}, String, Int64, MVector{2, Int64}, String, typeof(WEIGHT_FUNC), Float64}, i1::Int, i2::Int, j1::Int, j2::Int, two_cell_list::Vector{SVector{5, Int}})
    """
    Helper function to remove a given 2-cell from the given structure.

    Note that there are many possible colorings that can be used after removing the 2-cell.
    This method removes the 2-cell from a given list of 2-cells (the history of cell additions),
    then updates the graph to match the new list of 2-cells.

    i1, i2 are integers between 1 and M.
    j1, j2 are integers between 1 and N.
    """

    filter!(cell -> cell[1] != i1 || cell[2] != i2 || cell[3] != j1 || cell[4] != j2, two_cell_list)
    update_graph!(graph, two_cell_list)
end

function no_fold(graph::MetaGraph{Int64, SimpleGraph{Int64}, String, Int64, MVector{2, Int64}, String, typeof(WEIGHT_FUNC), Float64})::Bool
    """
    Returns true if the no_fold condition holds, false otherwise.
    """
    # Find folds in L_A
    for i1 in 1:M
        used_colors = Set{Int}()

        # Check vertex i1 for folds
        for i2 in neighbors(graph, i1)
            if i2 > M
                continue
            end
            color = graph[label_for(graph, i1), label_for(graph, i2)][1]

            if i1 < i2 && color in used_colors
                return false
            elseif -1 * color in used_colors
                return false
            end
        end
    end

    # Find folds in L_B
    for j1 in 1:N
        used_colors = Set{Int}()

        # Check vertex j1 for folds
        for j2_code in neighbors(graph, j1+M)
            j2 = j2_code - M
            if j2 <= 0
                continue
            end
            color = graph[label_for(graph, j1+M), label_for(graph, j2+M)][1]

            if j1 < j2 && color in used_colors
                return false
            elseif -1 * color in used_colors
                return false
            end
        end
    end

    return true
end

function remove_folds!(graph::MetaGraph{Int64, SimpleGraph{Int64}, String, Int64, MVector{2, Int64}, String, typeof(WEIGHT_FUNC), Float64}, two_cell_list::Vector{SVector{5, Int}})
    """
    Forces the given graph to obey the no-fold condition by removing bad 2-cells.

    Algorithm: For each a-vertex, list all a-neighbors, detect folds (and how many), and delete 2-cells
    until folds are removed. Do the same for b-vertices. Deletion is done randomly.

    Note that positive integers correspond to outgoing edges and vice versa.
    """

    # Find and remove folds in L_A
    for i1 in 1:M
        used_colors = Set{Int}()
        fold_colors = Dict{Int, Int}()

        # Count all folds with corresponding colors/orientation
        for i2 in neighbors(graph, i1)
            if i2 > M
                continue
            elseif i1 < i2
                current_color = graph[label_for(graph, i1), label_for(graph, i2)][1]
                if current_color in used_colors
                    if haskey(fold_colors, current_color)
                        fold_colors[current_color] += 1
                    else
                        fold_colors[current_color] = 1
                    end
                else
                    push!(used_colors, graph[label_for(graph, i1), label_for(graph, i2)][1])
                end
            else
                current_color = -1 * graph[label_for(graph, i1), label_for(graph, i2)][1]
                if current_color in used_colors
                    if haskey(fold_colors, current_color)
                        fold_colors[current_color] += 1
                    else
                        fold_colors[current_color] = 1
                    end
                else
                    push!(used_colors, current_color)
                end
            end
        end

        while !isempty(fold_colors)
            # Randomly choose a color to fold.
            color_choice = rand(collect(keys(fold_colors)))

            # Choose a-neighbor with the least copies to delete
            i2_choice = 0
            gluing_number = MAX_GLUING_NUM + 1
            for i2 in neighbors(graph, i1)
                if i2 > M
                    continue
                elseif i1 < i2 && graph[label_for(graph, i1), label_for(graph, i2)][1] == color_choice
                    if graph[label_for(graph, i1), label_for(graph, i2)][2] < gluing_number
                        i2_choice = i2
                        gluing_number = graph[label_for(graph, i1), label_for(graph, i2)][2]
                    end
                elseif i2 < i1 && graph[label_for(graph, i1), label_for(graph, i2)][1] == -1 * color_choice
                    if graph[label_for(graph, i1), label_for(graph, i2)][2] < gluing_number
                        i2_choice = i2
                        gluing_number = graph[label_for(graph, i1), label_for(graph, i2)][2]
                    end
                end
            end

            # Randomly choose an associated two-cell
            allowed_two_cells = Vector{SVector{2, Int}}()
            for j1 in 1:N
                for j2 in 1:N
                    if two_cell_in_taiko(i1, i2_choice, j1, j2)
                        push!(allowed_two_cells, SVector{Int}(j1, j2))
                    end
                end
            end
            two_cell_choice = rand(allowed_two_cells)
            j1_choice = two_cell_choice[1]
            j2_choice = two_cell_choice[2]

            remove_two_cell!(graph, i1, i2_choice, j1_choice, j2_choice, two_cell_list)

            if !has_edge(graph, i1, i2)
                if fold_colors[color_choice] == 1
                    delete!(fold_colors, color_choice)
                else
                    fold_colors[color_choice] -= 1
                end
            end
        end
    end

    # Find and remove folds in L_B
    for j1 in 1:N
        used_colors = Set{Int}()
        fold_colors = Dict{Int, Int}()

        # Count all folds with corresponding colors/orientation
        for j2 in neighbors(graph, j1+M)
            if j2 <= M
                continue
            elseif j1 < j2 - M
                current_color = graph[label_for(graph, j1+M), "b$(j2-M)"][1]
                if current_color in used_colors
                    if haskey(fold_colors, current_color)
                        fold_colors[current_color] += 1
                    else
                        fold_colors[current_color] = 1
                    end
                else
                    push!(used_colors, graph[label_for(graph, j1+M), "b$(j2-M)"][1])
                end
            else
                current_color = -1 * graph[label_for(graph, j1+M), "b$(j2-M)"][1]
                if current_color in used_colors
                    if haskey(fold_colors, current_color)
                        fold_colors[current_color] += 1
                    else
                        fold_colors[current_color] = 1
                    end
                else
                    push!(used_colors, current_color)
                end
            end
        end

        while !isempty(fold_colors)
            # Randomly choose a color to fold.
            color_choice = rand(collect(keys(fold_colors)))

            # Choose b-neighbor with the least copies to delete
            j2_choice = 0
            gluing_number = MAX_GLUING_NUM + 1
            for j2 in neighbors(graph, j1+M)
                if j2 <= M
                    continue
                elseif j1 < j2-M && graph[label_for(graph, j1+M), "b$(j2-M)"][1] == color_choice
                    if graph[label_for(graph, j1+M), "b$(j2-M)"][2] < gluing_number
                        j2_choice = j2-M
                        gluing_number = graph[label_for(graph, j1+M), "b$(j2-M)"][2]
                    end
                elseif j2-M < j1 && graph[label_for(graph, j1+M), "b$(j2-M)"][1] == -1 * color_choice
                    if graph[label_for(graph, j1+M), "b$(j2-M)"][2] < gluing_number
                        j2_choice = j2-M
                        gluing_number = graph[label_for(graph, j1+M), "b$(j2-M)"][2]
                    end
                end
            end

            # Randomly choose an associated two-cell
            allowed_two_cells = Vector{SVector{2, Int}}()
            for i1 in 1:M
                for i2 in 1:M
                    if two_cell_in_taiko(i1, i2, j1, j2_choice)
                        push!(allowed_two_cells, SVector{Int}(i1, i2))
                    end
                end
            end
            two_cell_choice = rand(allowed_two_cells)
            i1_choice = two_cell_choice[1]
            i2_choice = two_cell_choice[2]

            remove_two_cell!(graph, i1_choice, i2_choice, j1, j2_choice, two_cell_list)

            # Update fold_colors dict
            if !has_edge(graph, j1+M, j2_choice+M)
                if fold_colors[color_choice] == 1
                    delete!(fold_colors, color_choice)
                else
                    fold_colors[color_choice] -= 1
                end
            end
        end
    end
end

function no_pattern(graph::MetaGraph{Int64, SimpleGraph{Int64}, String, Int64, MVector{2, Int64}, String, typeof(WEIGHT_FUNC), Float64})::Bool
    """
    Returns true if the no_pattern condition holds, false otherwise.
    """
    used_patterns = Set{Pair{Int, Int}}()

    # Check L_A
    for i1 in 1:M
        adj_edges = Vector{Pair{Int, Int}}()
        for i2 in neighbors(graph, i1)
            if i2 <= M
                push!(adj_edges, i1 => i2)
            end
        end

        # Enumerate pairs of distinct adjacent edges and add patterns to set.
        E = size(adj_edges, 1)
        for e1 in 1:E
            for e2 in e1+1:E
                color1 = 0
                color2 = 0

                # Get the color/orientation of the two edges
                i2_1 = adj_edges[e1][2]
                i2_2 = adj_edges[e2][2]

                if i1 < i2_1
                    color1 = graph[label_for(graph, i1), label_for(graph, i2_1)][1]
                else
                    color1 = -1 * graph[label_for(graph, i1), label_for(graph, i2_1)][1]
                end

                if i1 < i2_2
                    color2 = graph[label_for(graph, i1), label_for(graph, i2_2)][1]
                else
                    color2 = -1 * graph[label_for(graph, i1), label_for(graph, i2_2)][1]
                end

                # Check if pattern is repeated.
                if (color1 => color2) in used_patterns || (color2 => color1) in used_patterns
                    return false
                else
                    push!(used_patterns, color1 => color2)
                    push!(used_patterns, color2 => color1)
                end
            end
        end
    end

    # Check L_B
    for j1 in 1:N
        adj_edges = Vector{Pair{Int, Int}}()
        for j2 in neighbors(graph, j1+M)
            if j2 > M
                push!(adj_edges, j1 => j2-M)
            end
        end

        # Enumerate pairs of distinct adjacent edges and add patterns to set.
        E = size(adj_edges, 1)
        for e1 in 1:E
            for e2 in e1+1:E
                color1 = 0
                color2 = 0

                # Get the color/orientation of the two edges
                j2_1 = adj_edges[e1][2]
                j2_2 = adj_edges[e2][2]

                if j1 < j2_1
                    color1 = graph[label_for(graph, j1+M), label_for(graph, j2_1+M)][1]
                else
                    color1 = -1 * graph[label_for(graph, j1+M), label_for(graph, j2_1+M)][1]
                end

                if j1 < j2_2
                    color2 = graph[label_for(graph, j1+M), label_for(graph, j2_2+M)][1]
                else
                    color2 = -1 * graph[label_for(graph, j1+M), label_for(graph, j2_2+M)][1]
                end

                # Check if pattern is repeated.
                if (color1 => color2) in used_patterns || (color2 => color1) in used_patterns
                    return false
                else
                    push!(used_patterns, color1 => color2)
                    push!(used_patterns, color2 => color1)
                end
            end
        end
    end

    return true
end

function find_repeated_patterns(graph::MetaGraph{Int64, SimpleGraph{Int64}, String, Int64, MVector{2, Int64}, String, typeof(WEIGHT_FUNC), Float64})::Dict{Pair{Int, Int}, Vector{Pair{Pair{String, String}, Pair{String, String}}}}
    """
    Finds repeated patterns and returns the pairs of edges realizing the repeated patterns.
    """
    # Find repeated patterns and store pairs of horizontal edges in a dict
    repeated_patterns = Dict{Pair{Int, Int}, Vector{Pair{Pair{String, String}, Pair{String, String}}}}()

    # Check L_A
    for i1 in 1:M
        adj_edges = Vector{Pair{Int, Int}}()
        for i2 in neighbors(graph, i1)
            if i2 <= M
                push!(adj_edges, i1 => i2)
            end
        end

        # Enumerate pairs of distinct adjacent edges and add patterns to set.
        E = size(adj_edges, 1)
        for e1 in 1:E
            for e2 in e1+1:E
                color1 = 0
                color2 = 0

                # Get the color/orientation of the two edges
                i2_1 = adj_edges[e1][2]
                i2_2 = adj_edges[e2][2]

                if i1 < i2_1
                    color1 = graph[label_for(graph, i1), label_for(graph, i2_1)][1]
                else
                    color1 = -1 * graph[label_for(graph, i1), label_for(graph, i2_1)][1]
                end

                if i1 < i2_2
                    color2 = graph[label_for(graph, i1), label_for(graph, i2_2)][1]
                else
                    color2 = -1 * graph[label_for(graph, i1), label_for(graph, i2_2)][1]
                end

                # Add patterns to dict
                if (color1 => color2) in repeated_patterns
                    push!(repeated_patterns[(color1 => color2)], ((label_for(graph, i1) => label_for(graph, i2_1)) => (label_for(graph, i1) => label_for(graph, i2_2))))
                elseif (color2 => color1) in repeated_patterns
                    push!(repeated_patterns[(color2 => color1)], ((label_for(graph, i1) => label_for(graph, i2_1)) => (label_for(graph, i1) => label_for(graph, i2_2))))
                else
                    repeated_patterns[(color1 => color2)] = [((label_for(graph, i1) => label_for(graph, i2_1)) => (label_for(graph, i1) => label_for(graph, i2_2)))]
                end
            end
        end
    end

    # Check L_B
    for j1 in 1:N
        adj_edges = Vector{Pair{Int, Int}}()
        for j2 in neighbors(graph, j1+M)
            if j2 > M
                push!(adj_edges, j1 => j2-M)
            end
        end

        # Enumerate pairs of distinct adjacent edges and add patterns to set.
        E = size(adj_edges, 1)
        for e1 in 1:E
            for e2 in e1+1:E
                color1 = 0
                color2 = 0

                # Get the color/orientation of the two edges
                j2_1 = adj_edges[e1][2]
                j2_2 = adj_edges[e2][2]

                if j1 < j2_1
                    color1 = graph[label_for(graph, j1+M), label_for(graph, j2_1+M)][1]
                else
                    color1 = -1 * graph[label_for(graph, j1+M), label_for(graph, j2_1+M)][1]
                end

                if j1 < j2_2
                    color2 = graph[label_for(graph, j1+M), label_for(graph, j2_2+M)][1]
                else
                    color2 = -1 * graph[label_for(graph, j1+M), label_for(graph, j2_2+M)][1]
                end

                # Add patterns to dict.
                if (color1 => color2) in keys(repeated_patterns)
                    push!(repeated_patterns[(color1 => color2)], ((label_for(graph, j1+M) => label_for(graph, j2_1+M)) => (label_for(graph, j1+M) => label_for(graph, j2_2+M))))
                elseif (color2 => color1) in keys(repeated_patterns)
                    push!(repeated_patterns[(color2 => color1)], ((label_for(graph, j1+M) => label_for(graph, j2_1+M)) => (label_for(graph, j1+M) => label_for(graph, j2_2+M))))
                else
                    repeated_patterns[(color1 => color2)] = [((label_for(graph, j1+M) => label_for(graph, j2_1+M)) => (label_for(graph, j1+M) => label_for(graph, j2_2+M)))]
                end
            end
        end
    end

    # Filter dict for repeated patterns
    for (pattern, edge_pairs) in repeated_patterns
        if size(edge_pairs, 1) == 1
            repeated_patterns = delete!(repeated_patterns, pattern)
        end
    end

    return repeated_patterns
end

function remove_patterns!(graph::MetaGraph{Int64, SimpleGraph{Int64}, String, Int64, MVector{2, Int64}, String, typeof(WEIGHT_FUNC), Float64}, two_cell_list::Vector{SVector{5, Int}})
    """
    Forces the given graph to obey the no-pattern condition by removing bad 2-cells.

    Algorithm: For each repeated pattern, list all horizontal edges realizing the pattern. Delete 2-cells corresponding
    to the edge of least occurrence until pattern no longer repeats.

    Note that positive integers correspond to outgoing edges and vice versa.
    """

    repeated_patterns = find_repeated_patterns(graph)

    # For each repeated pattern, find the edge with the least gluing number, and remove a randomly chosen 2-cell with this edge.
    while !isempty(repeated_patterns)
        # Randomly chose a repeated pattern to remove.
        pattern_choice = rand(collect(keys(repeated_patterns)))

        # Store a vector of edges with the min gluing number
        gluing_number = MAX_GLUING_NUM + 1
        edge_choices = Vector{Pair{String, String}}()
        for (edge1, edge2) in repeated_patterns[pattern_choice]
            gluing_number1 = graph[edge1[1], edge1[2]][2]
            gluing_number2 = graph[edge2[1], edge2[2]][2]

            if gluing_number1 < gluing_number
                gluing_number = gluing_number1
                edge_choices = [(edge1[1] => edge1[2])]
            elseif gluing_number1 == gluing_number
                push!(edge_choices, (edge1[1] => edge1[2]))
            end

            if gluing_number2 < gluing_number
                gluing_number = gluing_number2
                edge_choices = [(edge2[1] => edge2[2])]
            elseif gluing_number2 == gluing_number
                push!(edge_choices, (edge2[1] => edge2[2]))
            end
        end

        # Randomly choose an edge to remove
        edge_choice = rand(edge_choices)

        # Edge is in L_A
        if occursin("a", edge_choice[1])
            i1 = code_for(graph, edge_choice[1])
            i2 = code_for(graph, edge_choice[2])

            # Randomly choose an associated two-cell
            allowed_two_cells = Vector{SVector{2, Int}}()
            for j1 in 1:N
                for j2 in 1:N
                    if two_cell_in_taiko(i1, i2, j1, j2)
                        push!(allowed_two_cells, SVector{Int}(j1, j2))
                    end
                end
            end
            if !isempty(allowed_two_cells)
                two_cell_choice = rand(allowed_two_cells)
                j1_choice = two_cell_choice[1]
                j2_choice = two_cell_choice[2]

                remove_two_cell!(graph, i1, i2, j1_choice, j2_choice, two_cell_list)

                # Update repeated_patterns dict
                repeated_patterns = find_repeated_patterns(graph)
            end
        # Edge is in L_B
        elseif occursin("b", edge_choice[1])
            j1 = code_for(graph, edge_choice[1]) - M
            j2 = code_for(graph, edge_choice[2]) - M

            # Randomly choose an associated two-cell
            allowed_two_cells = Vector{SVector{2, Int}}()
            for i1 in 1:M
                for i2 in i1+1:M
                    if two_cell_in_taiko(graph, i1, i2, j1, j2)
                        push!(allowed_two_cells, SVector{2, Int}(i1, i2))
                    end
                end
            end

            if !isempty(allowed_two_cells)
                two_cell_choice = rand(allowed_two_cells)
                i1_choice = two_cell_choice[1]
                i2_choice = two_cell_choice[2]

                remove_two_cell!(graph, i1_choice, i2_choice, j1, j2, two_cell_list)

                # Update repeated_patterns dict. Note that we call find_repeated_patterns in case the 2-cell removed multiple patterns.
                repeated_patterns = find_repeated_patterns(graph)
            end
        end
    end
end

function is_left_aligned(i1::Int, i2::Int, j1::Int, j2::Int, lambda_a::Int, lambda_b::Int)::Bool
    """
    Takes as input a 2-cell (i1, i2, j1, j2) and integers lambda_a and lambda_b.

    Returns true if the given 2-cell is left-aligned, false otherwise.

    Based on left-alignment as described in Section 3.2 of Garg-Minyev.
    """

    if i1 > lambda_a + 1
        return false
    elseif i2 > lambda_a + 1 && i2 > i1 + 1
        return false
    end

    if j1 > lambda_b + 1
        return false
    elseif j2 > lambda_b + 1 && j2 > j1 + 1
        return false
    end

    return true
end

function greedy_search_from_startpoint(db, obj::OBJ_TYPE)::Vector{OBJ_TYPE}
    """
    Main greedy search algorithm.
    It starts and ends with some construction

    Algorithm (taiko concept taken from https://mineyev.web.illinois.edu/art/top-geom-uzd-origami.pdf):

    Input: A string P corresponding to a partition of the edge set.

    (1) Check if current partition is orientable, no-fold, and no-pattern. If not, remove 2-cells until it is.

    Kinds of bad things:
    1. Partition error - remove bad 2-cells and try again
    2. Orientation error - try flipping orientation of bad 2-cell, otherwise remove it

    Note that partition and orientation errors will halt further graph construction, so modifications are made at the string level.

    3. Fold - remove bad 2-cell
    4. Pattern - remove bad 2-cell

    No-fold and no-pattern are checked after graph construction, so modifications are made at the graph level.
    Note that removing a 2-cell will never make it violate partition or orientation conditions any more than it already does.

    (2) Add allowed 2-cells randomly until there are no moves left. Ensure after each step that the structure is still orientable, no-fold, and no-pattern.
    Only add left-aligned cells to prune the search space.
    """
    graph = convert_string_to_graph(obj)
    # current_reward = reward_calc(obj)

    # Reject any obviously wrong input (wrong length, too many 2-cells)
    if graph[] == "input error"
        return ["error"]
    end

    two_cells_raw = parse.(Int, split(obj, r"[,()]", keepempty=false))
    two_cell_list = Vector{SVector{5, Int}}()

    for (e1_index, e2_index) in partition(two_cells_raw, 2)
        e1 = INDEX_TO_EDGE[e1_index]
        e2 = INDEX_TO_EDGE[e2_index]
        i1 = e1[1]
        j1 = e1[2]
        i2 = e2[1]
        j2 = e2[2]
        orientation = 1
        if i1 > i2
            i1, i2 = i2, i1
            j1, j2 = j2, j1
            orientation = -1
        end

        push!(two_cell_list, SVector{5, Int}(i1, i2, j1, j2, orientation))
    end

    # Fix partition
    while occursin(graph[], "partition error")
        two_cell_str = match(r"([0-9]*,[0-9]*)", graph[])
        two_cell = parse.(Int, split(two_cell_str, r",()", keepempty=false))
        i1, i2, j1, j2, orientation = convert_edges_to_two_cell(two_cell[1], two_cell[2])

        filter!(cell => cell[1] != i1 || cell[2] != i2 || cell[3] != j1 || cell[4] != j2, two_cell_list)
        update_graph!(graph, two_cell_list)
    end

    # Fix orientation
    while occursin(graph[], "orientation error")
        two_cell_str = match(r"([0-9]*,[0-9]*)", graph[])
        two_cell = parse.(Int, split(two_cell_str, r",()", keepempty=false))
        i1, i2, j1, j2, orientation = convert_edges_to_two_cell(two_cell[1], two_cell[2])

        filter!(cell => cell[1] != i1 || cell[2] != i2 || cell[3] != j1 || cell[4] != j2, two_cell_list)
        push!(two_cell_list, SVector{5, Int}(i1, i2, j1, j2, orientation * (-1)))
        update_graph!(graph, two_cell_list)

        if occursin(graph[], "orientation error")
            two_cell_str = match(r"([0-9]*,[0-9]*)", graph[])
            two_cell = parse.(Int, split(two_cell_str, r",()", keepempty=false))
            i1, i2, j1, j2, orientation = convert_edges_to_two_cell(two_cell[1], two_cell[2])

            filter!(cell => cell[1] != i1 || cell[2] != i2 || cell[3] != j1 || cell[4] != j2, two_cell_list)
            update_graph!(graph, two_cell_list)
        end
    end

    # Remove folds and patterns
    remove_folds!(graph, two_cell_list)
    remove_patterns!(graph, two_cell_list)

    # (2) adding 2-cells to refine the taiko
    num_two_cells = 0
    allowed_two_cells = Vector{SVector{5, Int}}()
    push!(allowed_two_cells, SVector(0,0,0,0,1))
    num_two_cells = size(two_cell_list, 1)

    # Keep adding 2-cells until we are no longer able to without violating no_fold and no_pattern
    while num_two_cells < MAX_GLUING_NUM && !isempty(allowed_two_cells)
        # readline() # DEBUG
        # println(two_cell_list) # DEBUG
        # println(convert_graph_to_string(graph))
        empty!(allowed_two_cells)
        # Compute lambda_a and lambda_b
        lambda_a = 0
        lambda_b = 0
        for two_cell in two_cell_list
            if two_cell[1] > lambda_a
                lambda_a = two_cell[1]
            end

            if two_cell[2] > lambda_b
                lambda_b = two_cell[2]
            end

            if two_cell[3] > lambda_a
                lambda_a = two_cell[3]
            end

            if two_cell[4] > lambda_b
                lambda_b = two_cell[4]
            end
        end
        I1 = min(lambda_a+1, M)
        I2 = min(lambda_a+2, M)
        J1 = min(lambda_b+1, N)
        J2 = min(lambda_b+2, N)

        # Search for allowed left-aligned 2-cells
        for i1 in 1:I1
            for i2 in (i1+1):I2
                for j1 in 1:J1
                    for j2 in 1:J2
                        if is_left_aligned(i1, i2, j1, j2, lambda_a, lambda_b)
                            if can_add_two_cell(graph, i1, i2, j1, j2, 1)
                                add_two_cell!(graph, i1, i2, j1, j2, 1)
                                push!(two_cell_list, SVector{5, Int}(i1, i2, j1, j2, 1))
                                if min_girth_at_least_six(graph) && no_fold(graph) && no_pattern(graph)
                                    push!(allowed_two_cells, SVector{5, Int}(i1, i2, j1, j2, 1))
                                end
                                remove_two_cell!(graph, i1, i2, j1, j2, two_cell_list)
                            end

                            if can_add_two_cell(graph, i1, i2, j1, j2, -1)
                                add_two_cell!(graph, i1, i2, j1, j2, -1)
                                push!(two_cell_list, SVector{5, Int}(i1, i2, j1, j2, -1))
                                if min_girth_at_least_six(graph) && no_fold(graph) && no_pattern(graph)
                                    push!(allowed_two_cells, SVector{5, Int}(i1, i2, j1, j2, -1))
                                end
                                remove_two_cell!(graph, i1, i2, j1, j2, two_cell_list)
                            end
                        end
                    end
                end
            end
        end

        # Choose a random 2-cell (if any) and add it to the graph.
        if !isempty(allowed_two_cells)
            two_cell_choice = rand(allowed_two_cells)
            add_two_cell!(graph, two_cell_choice[1], two_cell_choice[2], two_cell_choice[3], two_cell_choice[4], two_cell_choice[5])
            push!(two_cell_list, two_cell_choice)
            graph_str = convert_graph_to_string(graph)
            num_two_cells = size(two_cell_list, 1)
        end
    end

    # println(two_cell_list) # DEBUG
    return [convert_graph_to_string(graph)]
end

function girth_at_least_six(graph::SimpleGraph)::Bool
    """
    Returns true if the girth, or the length of the shortest simple cycle, is >= 6. Returns false otherwise.
    """
    for vertex in vertices(graph)
        # Perform BFS on vertex to search for cycles
        explored = Dict{Int, Int}()
        explored[vertex] = 0

        parent = Dict{Int, Int}()
        parent[vertex] = 0
        bfs_queue = Queue{Int}()
        enqueue!(bfs_queue, vertex)
        depth = 0
        while !isempty(bfs_queue) && depth <= 6
            v = dequeue!(bfs_queue)
            depth = explored[v]
            for w in neighbors(graph, v)
                if !(w in keys(explored))
                    explored[w] = depth+1
                    parent[w] = v
                    enqueue!(bfs_queue, w)
                # w has been visited, is not the parent of v, and the cycle length is < girth
                elseif w != parent[v]
                    return false
                end
            end
        end
    end

    return true
end

function min_girth_at_least_six(graph::MetaGraph{Int64, SimpleGraph{Int64}, String, Int64, MVector{2, Int64}, String, typeof(WEIGHT_FUNC), Float64})::Bool
    """
    Returns true if the girth for both L_A and L_B are at least 6, false otherwise.
    """
    graph_simple = SimpleGraph(graph)
    L_A, vmap_A = induced_subgraph(graph_simple, 1:M)
    L_B, vmap_B = induced_subgraph(graph_simple, (M+1):(M+N))
    return girth_at_least_six(L_A) && girth_at_least_six(L_B)
end

function reward_calc(obj::OBJ_TYPE)::REWARD_TYPE
    """
    Function to calculate the reward of a final construction.

    In our case, calculates the number of 2-cells in the given taiko.
    """
    if occursin("error", obj)
        return -1
    end

    return count(isequal('('), obj)
end


function empty_starting_point()::OBJ_TYPE
    """
    If there is no input file, the search starts always with this object
    (E.g. empty graph, all zeros matrix, etc)

    In our case, starts with the empty partition.
    """
    return ""
end
