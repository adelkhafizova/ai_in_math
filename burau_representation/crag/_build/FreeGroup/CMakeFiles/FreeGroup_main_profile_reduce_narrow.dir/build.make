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
include FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/compiler_depend.make

# Include the progress variables for this target.
include FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/progress.make

# Include the compile flags for this target's objects.
include FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/flags.make

FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/codegen:
.PHONY : FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/codegen

FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/main/profile_reduce_narrow.cpp.o: FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/flags.make
FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/main/profile_reduce_narrow.cpp.o: /Users/borisgolikov/Desktop/Burau/scripts/crag/FreeGroup/main/profile_reduce_narrow.cpp
FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/main/profile_reduce_narrow.cpp.o: FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/main/profile_reduce_narrow.cpp.o"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FreeGroup && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/main/profile_reduce_narrow.cpp.o -MF CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/main/profile_reduce_narrow.cpp.o.d -o CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/main/profile_reduce_narrow.cpp.o -c /Users/borisgolikov/Desktop/Burau/scripts/crag/FreeGroup/main/profile_reduce_narrow.cpp

FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/main/profile_reduce_narrow.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/main/profile_reduce_narrow.cpp.i"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FreeGroup && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/borisgolikov/Desktop/Burau/scripts/crag/FreeGroup/main/profile_reduce_narrow.cpp > CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/main/profile_reduce_narrow.cpp.i

FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/main/profile_reduce_narrow.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/main/profile_reduce_narrow.cpp.s"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FreeGroup && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/borisgolikov/Desktop/Burau/scripts/crag/FreeGroup/main/profile_reduce_narrow.cpp -o CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/main/profile_reduce_narrow.cpp.s

# Object files for target FreeGroup_main_profile_reduce_narrow
FreeGroup_main_profile_reduce_narrow_OBJECTS = \
"CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/main/profile_reduce_narrow.cpp.o"

# External object files for target FreeGroup_main_profile_reduce_narrow
FreeGroup_main_profile_reduce_narrow_EXTERNAL_OBJECTS =

FreeGroup/bin/profile_reduce_narrow: FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/main/profile_reduce_narrow.cpp.o
FreeGroup/bin/profile_reduce_narrow: FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/build.make
FreeGroup/bin/profile_reduce_narrow: FreeGroup/lib/libSLPv2.a
FreeGroup/bin/profile_reduce_narrow: general/lib/libboost_pool_gmp_allocator.a
FreeGroup/bin/profile_reduce_narrow: general/lib/libcrag_general.a
FreeGroup/bin/profile_reduce_narrow: ranlib/lib/libranlib.a
FreeGroup/bin/profile_reduce_narrow: 3rdParty/gmp/gmp_external-prefix/lib/libgmpxx.a
FreeGroup/bin/profile_reduce_narrow: 3rdParty/gmp/gmp_external-prefix/lib/libgmp.a
FreeGroup/bin/profile_reduce_narrow: FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin/profile_reduce_narrow"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FreeGroup && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/build: FreeGroup/bin/profile_reduce_narrow
.PHONY : FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/build

FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/clean:
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FreeGroup && $(CMAKE_COMMAND) -P CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/cmake_clean.cmake
.PHONY : FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/clean

FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/depend:
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/borisgolikov/Desktop/Burau/scripts/crag /Users/borisgolikov/Desktop/Burau/scripts/crag/FreeGroup /Users/borisgolikov/Desktop/Burau/scripts/crag/_build /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FreeGroup /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : FreeGroup/CMakeFiles/FreeGroup_main_profile_reduce_narrow.dir/depend

