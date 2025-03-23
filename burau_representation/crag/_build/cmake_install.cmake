# Install script for directory: /Users/borisgolikov/Desktop/Burau/scripts/crag

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/Library/Developer/CommandLineTools/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/3rdParty/gmp/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Alphabet/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/BraidGroup/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CryptoAAG/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CryptoAE/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CryptoKL/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CryptoShftConj/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/CryptoTripleDecomposition/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Elt/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Equation/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Examples/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FiniteField/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/FreeGroup/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/general/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Graph/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Graphics/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Group/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/HigmanGroup/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Maps/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/ranlib/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/StringSimilarity/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/SbgpFG/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/TheGrigorchukGroup/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Polynomial/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Matrix/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Random/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Kayawood/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Walnut/cmake_install.cmake")
  include("/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/Test/cmake_install.cmake")

endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/borisgolikov/Desktop/Burau/scripts/crag/_build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
