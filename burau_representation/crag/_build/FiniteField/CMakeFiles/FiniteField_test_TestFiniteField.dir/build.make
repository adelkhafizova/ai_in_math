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
include FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/compiler_depend.make

# Include the progress variables for this target.
include FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/progress.make

# Include the compile flags for this target's objects.
include FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/flags.make

FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/codegen:
.PHONY : FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/codegen

FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/test/TestFiniteField.cpp.o: FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/flags.make
FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/test/TestFiniteField.cpp.o: /Users/borisgolikov/Desktop/Burau/scripts/crag/FiniteField/test/TestFiniteField.cpp
FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/test/TestFiniteField.cpp.o: FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/test/TestFiniteField.cpp.o"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FiniteField && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/test/TestFiniteField.cpp.o -MF CMakeFiles/FiniteField_test_TestFiniteField.dir/test/TestFiniteField.cpp.o.d -o CMakeFiles/FiniteField_test_TestFiniteField.dir/test/TestFiniteField.cpp.o -c /Users/borisgolikov/Desktop/Burau/scripts/crag/FiniteField/test/TestFiniteField.cpp

FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/test/TestFiniteField.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/FiniteField_test_TestFiniteField.dir/test/TestFiniteField.cpp.i"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FiniteField && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/borisgolikov/Desktop/Burau/scripts/crag/FiniteField/test/TestFiniteField.cpp > CMakeFiles/FiniteField_test_TestFiniteField.dir/test/TestFiniteField.cpp.i

FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/test/TestFiniteField.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/FiniteField_test_TestFiniteField.dir/test/TestFiniteField.cpp.s"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FiniteField && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/borisgolikov/Desktop/Burau/scripts/crag/FiniteField/test/TestFiniteField.cpp -o CMakeFiles/FiniteField_test_TestFiniteField.dir/test/TestFiniteField.cpp.s

# Object files for target FiniteField_test_TestFiniteField
FiniteField_test_TestFiniteField_OBJECTS = \
"CMakeFiles/FiniteField_test_TestFiniteField.dir/test/TestFiniteField.cpp.o"

# External object files for target FiniteField_test_TestFiniteField
FiniteField_test_TestFiniteField_EXTERNAL_OBJECTS =

FiniteField/test/TestFiniteField: FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/test/TestFiniteField.cpp.o
FiniteField/test/TestFiniteField: FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/build.make
FiniteField/test/TestFiniteField: /Users/borisgolikov/.hunter/_Base/90cb83d/0f1dca9/63b9aa4/Install/lib/libgtest_maind.a
FiniteField/test/TestFiniteField: FiniteField/lib/libFiniteField.a
FiniteField/test/TestFiniteField: /Users/borisgolikov/.hunter/_Base/90cb83d/0f1dca9/63b9aa4/Install/lib/libgtestd.a
FiniteField/test/TestFiniteField: FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test/TestFiniteField"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FiniteField && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FiniteField_test_TestFiniteField.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/build: FiniteField/test/TestFiniteField
.PHONY : FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/build

FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/clean:
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FiniteField && $(CMAKE_COMMAND) -P CMakeFiles/FiniteField_test_TestFiniteField.dir/cmake_clean.cmake
.PHONY : FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/clean

FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/depend:
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/borisgolikov/Desktop/Burau/scripts/crag /Users/borisgolikov/Desktop/Burau/scripts/crag/FiniteField /Users/borisgolikov/Desktop/Burau/scripts/crag/_build /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FiniteField /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : FiniteField/CMakeFiles/FiniteField_test_TestFiniteField.dir/depend

