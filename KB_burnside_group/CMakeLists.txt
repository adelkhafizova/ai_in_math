cmake_minimum_required(VERSION 3.10)
project(AI_IN_MATH)

set(CMAKE_CXX_STANDARD 17)

add_executable(main
    main.cpp
    knuth-bendix.cpp
    group.cpp
    resolve.cpp
    reduce.cpp
    shortcut.cpp
    redundancy.cpp
    crit_pairs.cpp
   vector.cpp
)