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
include Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/compiler_depend.make

# Include the progress variables for this target.
include Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/progress.make

# Include the compile flags for this target's objects.
include Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/flags.make

Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/codegen:
.PHONY : Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/codegen

Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/test/test_kayawood.cpp.o: Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/flags.make
Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/test/test_kayawood.cpp.o: /Users/borisgolikov/Desktop/Burau/scripts/crag/Kayawood/test/test_kayawood.cpp
Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/test/test_kayawood.cpp.o: Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/test/test_kayawood.cpp.o"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Kayawood && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/test/test_kayawood.cpp.o -MF CMakeFiles/Kayawood_test_test_kayawood.dir/test/test_kayawood.cpp.o.d -o CMakeFiles/Kayawood_test_test_kayawood.dir/test/test_kayawood.cpp.o -c /Users/borisgolikov/Desktop/Burau/scripts/crag/Kayawood/test/test_kayawood.cpp

Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/test/test_kayawood.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Kayawood_test_test_kayawood.dir/test/test_kayawood.cpp.i"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Kayawood && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/borisgolikov/Desktop/Burau/scripts/crag/Kayawood/test/test_kayawood.cpp > CMakeFiles/Kayawood_test_test_kayawood.dir/test/test_kayawood.cpp.i

Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/test/test_kayawood.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Kayawood_test_test_kayawood.dir/test/test_kayawood.cpp.s"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Kayawood && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/borisgolikov/Desktop/Burau/scripts/crag/Kayawood/test/test_kayawood.cpp -o CMakeFiles/Kayawood_test_test_kayawood.dir/test/test_kayawood.cpp.s

# Object files for target Kayawood_test_test_kayawood
Kayawood_test_test_kayawood_OBJECTS = \
"CMakeFiles/Kayawood_test_test_kayawood.dir/test/test_kayawood.cpp.o"

# External object files for target Kayawood_test_test_kayawood
Kayawood_test_test_kayawood_EXTERNAL_OBJECTS =

Kayawood/test/test_kayawood: Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/test/test_kayawood.cpp.o
Kayawood/test/test_kayawood: Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/build.make
Kayawood/test/test_kayawood: /Users/borisgolikov/.hunter/_Base/90cb83d/0f1dca9/63b9aa4/Install/lib/libgtest_maind.a
Kayawood/test/test_kayawood: Kayawood/lib/libKayawood.a
Kayawood/test/test_kayawood: /Users/borisgolikov/.hunter/_Base/90cb83d/0f1dca9/63b9aa4/Install/lib/libgtestd.a
Kayawood/test/test_kayawood: BraidGroup/lib/libBraidGroup.a
Kayawood/test/test_kayawood: Polynomial/lib/libPolynomial.a
Kayawood/test/test_kayawood: FiniteField/lib/libFiniteField.a
Kayawood/test/test_kayawood: Matrix/lib/libMatrix.a
Kayawood/test/test_kayawood: Random/lib/libRandom.a
Kayawood/test/test_kayawood: Elt/lib/libElt.a
Kayawood/test/test_kayawood: Alphabet/lib/libAlphabet.a
Kayawood/test/test_kayawood: Elt/lib/libElt.a
Kayawood/test/test_kayawood: Alphabet/lib/libAlphabet.a
Kayawood/test/test_kayawood: general/lib/libcrag_general.a
Kayawood/test/test_kayawood: ranlib/lib/libranlib.a
Kayawood/test/test_kayawood: Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test/test_kayawood"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Kayawood && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Kayawood_test_test_kayawood.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/build: Kayawood/test/test_kayawood
.PHONY : Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/build

Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/clean:
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Kayawood && $(CMAKE_COMMAND) -P CMakeFiles/Kayawood_test_test_kayawood.dir/cmake_clean.cmake
.PHONY : Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/clean

Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/depend:
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/borisgolikov/Desktop/Burau/scripts/crag /Users/borisgolikov/Desktop/Burau/scripts/crag/Kayawood /Users/borisgolikov/Desktop/Burau/scripts/crag/_build /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Kayawood /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : Kayawood/CMakeFiles/Kayawood_test_test_kayawood.dir/depend

