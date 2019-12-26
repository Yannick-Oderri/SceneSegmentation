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

PROJECT(project_edge
        VERSION 0.0.1)

SET(${project_edge_SOURCE_DIR} "${CMAKE_SOURCE_DIR}/src")

OPTION(ENABLE_OPENGL "Enable OpenGL support" ON)
IF(ENABLE_PROFILING)
SET(LIBFREENECT2_WITH_PROFILING 1)
ENDIF()

IF(ENABLE_PROFILING)
  SET(PROJECT_EDGE_WITH_PROFILING 1)
ENDIF()

FIND_PACKAGE(freenect2) # Inlcude kinect libraries

SET(project_edge_INCLUDE_DIR
    "${CMAKE_SOURCE_DIR}/include")


file(GLOB project_edge_SOURCES src/*.cpp)

file(GLOB project_edge_HEADERS src/*.hpp)

SET(project_edge_DLLS
  ${freenect2_DLLS}
)

INCLUDE_DIRECTORIES(
        ${freenect2_INCLUDE_DIR}
        ${project_edge_INCLUDE_DIR}
)

IF(ENABLE_OPENGL)
  # Prefer OpenGLX Over Legacy
  set(OpenGL_GL_PREFERENCE "GLVND")
  FIND_PACKAGE(glfw3 REQUIRED)
  FIND_PACKAGE(OpenGL REQUIRED)
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

IF(ENABLE_OPENCV)
  FIND_PACKAGE(OpenCV REQUIRED)
  IF(OpenCV_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(project_edge ${OpenCV_LIBS})
  ENDIF()
ENDIF(ENABLE_OPENCV)

ADD_EXECUTABLE(project_edge
  ${project_edge_SOURCES}
  ${project_edge_src}
        src/frame.cpp src/frame.h src/pipeline.cpp src/pipeline.h src/context.cpp src/context.h)

TARGET_LINK_LIBRARIES(project_edge
  ${project_edge_LIBRARIES}
)


