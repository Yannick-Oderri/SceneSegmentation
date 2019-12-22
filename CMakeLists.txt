CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

if(WIN32 AND NOT MINGW)
  if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "d")
  endif()
endif()

IF(NOT DEFINED CMAKE_BUILD_TYPE)
  # No effect for multi-configuration generators (e.g. for Visual Studio)
  SET(CMAKE_BUILD_TYPE DEBUG CACHE STRING "Choose: RelWithDebInfo Release Debug MinSizeRel None")
ENDIF()

PROJECT(project_edge)

SET(MY_DIR ${project_edge_SOURCE_DIR})

OPTION(ENABLE_OPENGL "Enable OpenGL support" ON)
IF(ENABLE_PROFILING)
SET(LIBFREENECT2_WITH_PROFILING 1)
ENDIF()

IF(ENABLE_PROFILING)
  SET(PROJECT_EDGE_WITH_PROFILING 1)
ENDIF()

FIND_PACKAGE(freenect2) # Inlcude kinect libraries

INCLUDE_DIRECTORIES(
  ${freenect2_INCLUDE_DIR}
)

SET(project_edge_src
  app.cpp
  viewer.cpp
)

SET(project_edge_DLLS
  ${freenect2_DLLS}
)


IF(ENABLE_OPENGL)
  FIND_PACKAGE(GLFW3)
  FIND_PACKAGE(OpenGL)
  IF(GLFW3_FOUND AND OPENGL_FOUND)
    INCLUDE_DIRECTORIES(
      ${GLFW3_INCLUDE_DIRS}
      ${flextGL_INCLUDE_DIRS}
    )

    LIST(APPEND project_edge_DLLS ${GLFW3_DLL})
    LIST(APPEND project_edge_src
      viewer.cpp
      ${flextGL_SOURCES}
    )
    LIST(APPEND project_edge_LIBRARIES
      ${GLFW3_LIBRARIES}
      ${OPENGL_gl_LIBRARY}
    )
    ADD_DEFINITIONS(-DEXAMPLES_WITH_OPENGL_SUPPORT=1)
  ENDIF()
ENDIF(ENABLE_OPENGL)

ADD_EXECUTABLE(project_edge
  ${project_edge_src}
)

TARGET_LINK_LIBRARIES(project_edge
  ${project_edge_LIBRARIES}
)


