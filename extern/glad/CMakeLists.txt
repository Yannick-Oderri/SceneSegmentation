CMAKE_MINIMUM_REQUIRED(VERSION 3.11)

MESSAGE(STATUS "glad")
project(glad)


set(glad_INCLUDE_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/include" PARENT_SCOPE)
set(glad_SOURCE_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_library(glad ${glad_SOURCE_DIRS}/glad.c)
target_include_directories(glad PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(glad_LIBRARIES glad PARENT_SCOPE)