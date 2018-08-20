cmake_minimum_required(VERSION 3.10)
project(SNR VERSION 3.1)
include(GNUInstallDirs)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
set(TARGET_LINK_LIBRARIES snr isa_utils isa_opencl astrodata OpenCL)
if($ENV{LOFAR})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_HDF5")
  set(TARGET_LINK_LIBRARIES ${TARGET_LINK_LIBRARIES} hdf5 hdf5_cpp z)
endif()
if($ENV{PSRDADA})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_PSRDADA")
  set(TARGET_LINK_LIBRARIES ${TARGET_LINK_LIBRARIES} psrdada cudart)
endif()

# libsnr
add_library(snr SHARED
  src/SNR.cpp
)
set_target_properties(snr PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER "include/SNR.hpp"
)
target_include_directories(snr PRIVATE include)

# SNRTesting
add_executable(SNRTesting
  src/SNRTest.cpp
)
target_include_directories(SNRTesting PRIVATE include)
target_link_libraries(SNRTesting PRIVATE ${TARGET_LINK_LIBRARIES})

# SNRTuning
add_executable(SNRTuning
  src/SNRTuning.cpp
)
target_include_directories(SNRTuning PRIVATE include)
target_link_libraries(SNRTuning PRIVATE ${TARGET_LINK_LIBRARIES})

install(TARGETS snr SNRTesting SNRTuning
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
