#include <iostream>
#include <list>
#include <vector>
#include <utility>

// Helper function to print a vector of integers
void print_vector(const std::vector<int>& vec) {
    std::cout << "{";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << (i == vec.size() - 1 ? "" : ", ");
    }
    std::cout << "}";
}

// Helper function to print a pair of vectors of integers
void print_pair_of_vectors(const std::pair<std::vector<int>, std::vector<int>>& p) {
    std::cout << "{";
    print_vector(p.first);
    std::cout << ", ";
    print_vector(p.second);
    std::cout << "}";
}
