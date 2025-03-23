# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/3rdParty/gmp/m4-prefix/src/m4")
  file(MAKE_DIRECTORY "/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/3rdParty/gmp/m4-prefix/src/m4")
endif()
file(MAKE_DIRECTORY
  "/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/3rdParty/gmp/m4-prefix/src/m4-build"
  "/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/3rdParty/gmp/m4-prefix"
  "/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/3rdParty/gmp/m4-prefix/tmp"
  "/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/3rdParty/gmp/m4-prefix/src/m4-stamp"
  "/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/3rdParty/gmp/m4-prefix/src"
  "/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/3rdParty/gmp/m4-prefix/src/m4-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/3rdParty/gmp/m4-prefix/src/m4-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/3rdParty/gmp/m4-prefix/src/m4-stamp${cfgdir}") # cfgdir has leading slash
endif()
