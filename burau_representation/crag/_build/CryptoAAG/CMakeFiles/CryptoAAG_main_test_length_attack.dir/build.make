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
include CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/compiler_depend.make

# Include the progress variables for this target.
include CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/progress.make

# Include the compile flags for this target's objects.
include CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/flags.make

CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/codegen:
.PHONY : CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/codegen

CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/main/test_length_attack.cpp.o: CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/flags.make
CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/main/test_length_attack.cpp.o: /Users/borisgolikov/Desktop/Burau/scripts/crag/CryptoAAG/main/test_length_attack.cpp
CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/main/test_length_attack.cpp.o: CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/main/test_length_attack.cpp.o"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CryptoAAG && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/main/test_length_attack.cpp.o -MF CMakeFiles/CryptoAAG_main_test_length_attack.dir/main/test_length_attack.cpp.o.d -o CMakeFiles/CryptoAAG_main_test_length_attack.dir/main/test_length_attack.cpp.o -c /Users/borisgolikov/Desktop/Burau/scripts/crag/CryptoAAG/main/test_length_attack.cpp

CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/main/test_length_attack.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/CryptoAAG_main_test_length_attack.dir/main/test_length_attack.cpp.i"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CryptoAAG && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/borisgolikov/Desktop/Burau/scripts/crag/CryptoAAG/main/test_length_attack.cpp > CMakeFiles/CryptoAAG_main_test_length_attack.dir/main/test_length_attack.cpp.i

CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/main/test_length_attack.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/CryptoAAG_main_test_length_attack.dir/main/test_length_attack.cpp.s"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CryptoAAG && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/borisgolikov/Desktop/Burau/scripts/crag/CryptoAAG/main/test_length_attack.cpp -o CMakeFiles/CryptoAAG_main_test_length_attack.dir/main/test_length_attack.cpp.s

# Object files for target CryptoAAG_main_test_length_attack
CryptoAAG_main_test_length_attack_OBJECTS = \
"CMakeFiles/CryptoAAG_main_test_length_attack.dir/main/test_length_attack.cpp.o"

# External object files for target CryptoAAG_main_test_length_attack
CryptoAAG_main_test_length_attack_EXTERNAL_OBJECTS =

CryptoAAG/bin/test_length_attack: CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/main/test_length_attack.cpp.o
CryptoAAG/bin/test_length_attack: CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/build.make
CryptoAAG/bin/test_length_attack: CryptoAAG/lib/libCryptoAGG.a
CryptoAAG/bin/test_length_attack: ranlib/lib/libranlib.a
CryptoAAG/bin/test_length_attack: BraidGroup/lib/libBraidGroup.a
CryptoAAG/bin/test_length_attack: Polynomial/lib/libPolynomial.a
CryptoAAG/bin/test_length_attack: FiniteField/lib/libFiniteField.a
CryptoAAG/bin/test_length_attack: Matrix/lib/libMatrix.a
CryptoAAG/bin/test_length_attack: Random/lib/libRandom.a
CryptoAAG/bin/test_length_attack: Elt/lib/libElt.a
CryptoAAG/bin/test_length_attack: Alphabet/lib/libAlphabet.a
CryptoAAG/bin/test_length_attack: Elt/lib/libElt.a
CryptoAAG/bin/test_length_attack: Alphabet/lib/libAlphabet.a
CryptoAAG/bin/test_length_attack: general/lib/libcrag_general.a
CryptoAAG/bin/test_length_attack: ranlib/lib/libranlib.a
CryptoAAG/bin/test_length_attack: CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin/test_length_attack"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CryptoAAG && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CryptoAAG_main_test_length_attack.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/build: CryptoAAG/bin/test_length_attack
.PHONY : CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/build

CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/clean:
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CryptoAAG && $(CMAKE_COMMAND) -P CMakeFiles/CryptoAAG_main_test_length_attack.dir/cmake_clean.cmake
.PHONY : CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/clean

CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/depend:
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/borisgolikov/Desktop/Burau/scripts/crag /Users/borisgolikov/Desktop/Burau/scripts/crag/CryptoAAG /Users/borisgolikov/Desktop/Burau/scripts/crag/_build /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CryptoAAG /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CryptoAAG/CMakeFiles/CryptoAAG_main_test_length_attack.dir/depend

