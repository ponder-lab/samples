cmake_minimum_required( VERSION 2.8 )

find_package( OpenCV 3 REQUIRED )

find_package( OpenMP REQUIRED )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )

include_directories( /opt/softkinetic/DepthSenseSDK/include )
link_directories( /opt/softkinetic/DepthSenseSDK/lib )

include_directories( src )

file(GLOB SRCS src/*.cpp)

if( debug )
  set( CMAKE_BUILD_TYPE Debug )
else()
  set( CMAKE_BUILD_TYPE Release )
endif()

set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -DNDEBUG -march=native" )
set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3" )

set( TARGET opencv_Viz2 )

add_executable( ${TARGET} ${SRCS}  )
target_link_libraries( ${TARGET} opencv_core opencv_calib3d opencv_viz DepthSense pthread )
