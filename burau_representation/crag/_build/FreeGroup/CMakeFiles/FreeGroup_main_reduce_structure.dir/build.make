# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/borisgolikov/Desktop/Burau/scripts/crag

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/borisgolikov/Desktop/Burau/scripts/crag/_build

# Include any dependencies generated for this target.
include FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/compiler_depend.make

# Include the progress variables for this target.
include FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/progress.make

# Include the compile flags for this target's objects.
include FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/flags.make

FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/codegen:
.PHONY : FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/codegen

FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/main/reduce_structure.cpp.o: FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/flags.make
FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/main/reduce_structure.cpp.o: /Users/borisgolikov/Desktop/Burau/scripts/crag/FreeGroup/main/reduce_structure.cpp
FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/main/reduce_structure.cpp.o: FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/main/reduce_structure.cpp.o"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FreeGroup && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/main/reduce_structure.cpp.o -MF CMakeFiles/FreeGroup_main_reduce_structure.dir/main/reduce_structure.cpp.o.d -o CMakeFiles/FreeGroup_main_reduce_structure.dir/main/reduce_structure.cpp.o -c /Users/borisgolikov/Desktop/Burau/scripts/crag/FreeGroup/main/reduce_structure.cpp

FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/main/reduce_structure.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/FreeGroup_main_reduce_structure.dir/main/reduce_structure.cpp.i"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FreeGroup && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/borisgolikov/Desktop/Burau/scripts/crag/FreeGroup/main/reduce_structure.cpp > CMakeFiles/FreeGroup_main_reduce_structure.dir/main/reduce_structure.cpp.i

FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/main/reduce_structure.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/FreeGroup_main_reduce_structure.dir/main/reduce_structure.cpp.s"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FreeGroup && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/borisgolikov/Desktop/Burau/scripts/crag/FreeGroup/main/reduce_structure.cpp -o CMakeFiles/FreeGroup_main_reduce_structure.dir/main/reduce_structure.cpp.s

# Object files for target FreeGroup_main_reduce_structure
FreeGroup_main_reduce_structure_OBJECTS = \
"CMakeFiles/FreeGroup_main_reduce_structure.dir/main/reduce_structure.cpp.o"

# External object files for target FreeGroup_main_reduce_structure
FreeGroup_main_reduce_structure_EXTERNAL_OBJECTS =

FreeGroup/bin/reduce_structure: FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/main/reduce_structure.cpp.o
FreeGroup/bin/reduce_structure: FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/build.make
FreeGroup/bin/reduce_structure: FreeGroup/lib/libSLPv2.a
FreeGroup/bin/reduce_structure: general/lib/libboost_pool_gmp_allocator.a
FreeGroup/bin/reduce_structure: general/lib/libcrag_general.a
FreeGroup/bin/reduce_structure: ranlib/lib/libranlib.a
FreeGroup/bin/reduce_structure: 3rdParty/gmp/gmp_external-prefix/lib/libgmpxx.a
FreeGroup/bin/reduce_structure: 3rdParty/gmp/gmp_external-prefix/lib/libgmp.a
FreeGroup/bin/reduce_structure: FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin/reduce_structure"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FreeGroup && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FreeGroup_main_reduce_structure.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/build: FreeGroup/bin/reduce_structure
.PHONY : FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/build

FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/clean:
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FreeGroup && $(CMAKE_COMMAND) -P CMakeFiles/FreeGroup_main_reduce_structure.dir/cmake_clean.cmake
.PHONY : FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/clean

FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/depend:
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/borisgolikov/Desktop/Burau/scripts/crag /Users/borisgolikov/Desktop/Burau/scripts/crag/FreeGroup /Users/borisgolikov/Desktop/Burau/scripts/crag/_build /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FreeGroup /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : FreeGroup/CMakeFiles/FreeGroup_main_reduce_structure.dir/depend

