cmake_minimum_required( VERSION 2.8 )

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project( bftool )

set( CMAKE_CXX_STACK_SIZE "10000000" )
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules" )
if (WIN32)
    set( COLOR_TERM OFF )
else()
    set( COLOR_TERM ON )
endif()

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release )
endif()

set( PROJ_NAME "bftool" )
set( OUTPUT_NAME "bftool" )
set( INCLUDE_DIR "include" )

include( CheckCXXCompilerFlag )
CHECK_CXX_COMPILER_FLAG( "-std=c++11" COMPILER_SUPPORTS_CXX11 )
CHECK_CXX_COMPILER_FLAG( "-std=c++0x" COMPILER_SUPPORTS_CXX0X )

if( COMPILER_SUPPORTS_CXX11 )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
elseif( COMPILER_SUPPORTS_CXX0X )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x" )
else()
    message( STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler." )
endif()

if ( CMAKE_COMPILER_IS_GNUCXX )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-deprecated-declarations -O3 -g" )
endif ( CMAKE_COMPILER_IS_GNUCXX )

# set( Boost_USE_STATIC_LIBS ON )
# set( Boost_USE_STATIC ON )

# find_package( Boost COMPONENTS system program_options filesystem )
find_package( Eigen3 REQUIRED )

include_directories(
    ${INCLUDE_DIR}
    # ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

set(
  HEADERS
  ${INCLUDE_DIR}/boolfun.hpp
  ${INCLUDE_DIR}/lpsolver.hpp
  ${INCLUDE_DIR}/util.hpp
)

set(
  SOURCES
  main.cpp
  boolfun.cpp
  util.cpp
)

add_executable( ${PROJ_NAME} ${SOURCES} ${HEADERS} )
target_include_directories( ${PROJ_NAME} PRIVATE ${INCLUDE_DIR} )
set_target_properties( ${PROJ_NAME} PROPERTIES OUTPUT_NAME ${OUTPUT_NAME} )
target_link_libraries( ${PROJ_NAME} )
    # ${Boost_LIBRARIES} )

if ( MSVC )
    set_property(TARGET ${PROJ_NAME} APPEND PROPERTY LINK_FLAGS /DEBUG)
else ()
    target_link_libraries( ${PROJ_NAME} -pthread )
endif ( MSVC )
