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
include Elt/CMakeFiles/Elt.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include Elt/CMakeFiles/Elt.dir/compiler_depend.make

# Include the progress variables for this target.
include Elt/CMakeFiles/Elt.dir/progress.make

# Include the compile flags for this target's objects.
include Elt/CMakeFiles/Elt.dir/flags.make

Elt/CMakeFiles/Elt.dir/codegen:
.PHONY : Elt/CMakeFiles/Elt.dir/codegen

Elt/CMakeFiles/Elt.dir/src/Word.cpp.o: Elt/CMakeFiles/Elt.dir/flags.make
Elt/CMakeFiles/Elt.dir/src/Word.cpp.o: /Users/borisgolikov/Desktop/Burau/scripts/crag/Elt/src/Word.cpp
Elt/CMakeFiles/Elt.dir/src/Word.cpp.o: Elt/CMakeFiles/Elt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Elt/CMakeFiles/Elt.dir/src/Word.cpp.o"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Elt && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Elt/CMakeFiles/Elt.dir/src/Word.cpp.o -MF CMakeFiles/Elt.dir/src/Word.cpp.o.d -o CMakeFiles/Elt.dir/src/Word.cpp.o -c /Users/borisgolikov/Desktop/Burau/scripts/crag/Elt/src/Word.cpp

Elt/CMakeFiles/Elt.dir/src/Word.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Elt.dir/src/Word.cpp.i"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Elt && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/borisgolikov/Desktop/Burau/scripts/crag/Elt/src/Word.cpp > CMakeFiles/Elt.dir/src/Word.cpp.i

Elt/CMakeFiles/Elt.dir/src/Word.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Elt.dir/src/Word.cpp.s"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Elt && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/borisgolikov/Desktop/Burau/scripts/crag/Elt/src/Word.cpp -o CMakeFiles/Elt.dir/src/Word.cpp.s

Elt/CMakeFiles/Elt.dir/src/WordRep.cpp.o: Elt/CMakeFiles/Elt.dir/flags.make
Elt/CMakeFiles/Elt.dir/src/WordRep.cpp.o: /Users/borisgolikov/Desktop/Burau/scripts/crag/Elt/src/WordRep.cpp
Elt/CMakeFiles/Elt.dir/src/WordRep.cpp.o: Elt/CMakeFiles/Elt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object Elt/CMakeFiles/Elt.dir/src/WordRep.cpp.o"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Elt && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Elt/CMakeFiles/Elt.dir/src/WordRep.cpp.o -MF CMakeFiles/Elt.dir/src/WordRep.cpp.o.d -o CMakeFiles/Elt.dir/src/WordRep.cpp.o -c /Users/borisgolikov/Desktop/Burau/scripts/crag/Elt/src/WordRep.cpp

Elt/CMakeFiles/Elt.dir/src/WordRep.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Elt.dir/src/WordRep.cpp.i"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Elt && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/borisgolikov/Desktop/Burau/scripts/crag/Elt/src/WordRep.cpp > CMakeFiles/Elt.dir/src/WordRep.cpp.i

Elt/CMakeFiles/Elt.dir/src/WordRep.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Elt.dir/src/WordRep.cpp.s"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Elt && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/borisgolikov/Desktop/Burau/scripts/crag/Elt/src/WordRep.cpp -o CMakeFiles/Elt.dir/src/WordRep.cpp.s

# Object files for target Elt
Elt_OBJECTS = \
"CMakeFiles/Elt.dir/src/Word.cpp.o" \
"CMakeFiles/Elt.dir/src/WordRep.cpp.o"

# External object files for target Elt
Elt_EXTERNAL_OBJECTS =

Elt/lib/libElt.a: Elt/CMakeFiles/Elt.dir/src/Word.cpp.o
Elt/lib/libElt.a: Elt/CMakeFiles/Elt.dir/src/WordRep.cpp.o
Elt/lib/libElt.a: Elt/CMakeFiles/Elt.dir/build.make
Elt/lib/libElt.a: Elt/CMakeFiles/Elt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library lib/libElt.a"
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Elt && $(CMAKE_COMMAND) -P CMakeFiles/Elt.dir/cmake_clean_target.cmake
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Elt && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Elt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Elt/CMakeFiles/Elt.dir/build: Elt/lib/libElt.a
.PHONY : Elt/CMakeFiles/Elt.dir/build

Elt/CMakeFiles/Elt.dir/clean:
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Elt && $(CMAKE_COMMAND) -P CMakeFiles/Elt.dir/cmake_clean.cmake
.PHONY : Elt/CMakeFiles/Elt.dir/clean

Elt/CMakeFiles/Elt.dir/depend:
	cd /Users/borisgolikov/Desktop/Burau/scripts/crag/_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/borisgolikov/Desktop/Burau/scripts/crag /Users/borisgolikov/Desktop/Burau/scripts/crag/Elt /Users/borisgolikov/Desktop/Burau/scripts/crag/_build /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Elt /Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Elt/CMakeFiles/Elt.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : Elt/CMakeFiles/Elt.dir/depend

