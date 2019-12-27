CMAKE_MINIMUM_REQUIRED(VERSION 3.11)

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
        VERSION 0.0.1
        LANGUAGES CUDA CXX)

SET(${project_edge_SOURCE_DIR} "${CMAKE_SOURCE_DIR}/src")

OPTION(ENABLE_OPENGL "Enable OpenGL support" ON)
OPTION(ENABLE_OPENCV "Enable OpenCV support" ON)
OPTION(ENABLE_BOOST "Enable Boost support" ON)
OPTION(ENABLE_FREENECT2 "Enable Kinectv2 Drivers" OFF) # not functional atm
option(PACKAGE_TESTS "Build the tests" ON)


IF(ENABLE_PROFILING)
SET(LIBFREENECT2_WITH_PROFILING 1)
ENDIF()


# ========================================================
# = Setup Source Elements
# ========================================================
SET(project_edge_SOURCE
        app.cpp
        context.cpp
        frame.cpp
        pipeline.cpp
        viewer.cpp
)
SET(project_edge_HEADERS
        app.h
        context.h
        frame.h
        pipeline.h
        viewer.h
)

FUNCTION(PREPEND var prefix)
  SET(listVar "")
  FOREACH(f ${ARGN})
    LIST(APPEND listVar "${prefix}/${f}")
  ENDFOREACH(f)
  SET(${var} "${listVar}" PARENT_SCOPE)
ENDFUNCTION(PREPEND)

PREPEND(project_edge_src "${CMAKE_SOURCE_DIR}/src" ${project_edge_SOURCE})
PREPEND(project_edge_hdr "${CMAKE_SOURCE_DIR}/src" ${project_edge_HEADERS})

SET(project_edge_INCLUDE_DIR
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src"
)



# file(GLOB project_edge_SOURCES src/*.cpp)

# file(GLOB project_edge_HEADERS src/*.h)


INCLUDE_DIRECTORIES(
        ${project_edge_INCLUDE_DIR}
)

set(project_edge_LIBRARIES "")

# ========================================================
# = OpenGL Library
# ========================================================
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

# ========================================================
# = OpenCV Library
# ========================================================
IF(ENABLE_OPENCV)
  FIND_PACKAGE(OpenCV REQUIRED)
  IF(OpenCV_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIR})
    LIST(APPEND project_edge_LIBRARIES
            ${OpenCV_LIBS})
  ELSE()
    MESSAGE("Failed to Locate OpenCV Library")
  ENDIF()
ENDIF(ENABLE_OPENCV)

# ========================================================
# = Boost Library
# ========================================================
IF(ENABLE_BOOST)
  SET(Boost_USE_STATIC_LIBS ON)
  FIND_PACKAGE(Boost
          REQUIRED
          COMPONENTS log thread)
  IF(Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
    LIST(APPEND project_edge_LIBRARIES
            ${Boost_LIBRARIES})
  ENDIF()
ENDIF(ENABLE_BOOST)

# ========================================================
# = Freenect2 Library
# ========================================================
IF(ENABLE_FREENECT2)
  FIND_PACKAGE(freenect2
          REQUIRED)
  IF(freenect2_FOUND)
    INCLUDE_DIRECTORIES(${freenect2_INCLUDE_DIR})
    LIST(APPEND project_edge_DLLS
            ${freenect2_DLLS}
            )
  ENDIF()
ENDIF(ENABLE_FREENECT2)

# ========================================================
# = Target Runtime Declarations
# ========================================================
IF(PACKAGE_RUNTIME)
  ADD_EXECUTABLE(project_edge
          ${project_edge_src})
  TARGET_LINK_LIBRARIES(project_edge
    ${project_edge_LIBRARIES}
  )
ENDIF()

# ========================================================
# = Google Test Library
# ========================================================
IF(PACKAGE_TESTS)
  ADD_DEFINITIONS(-DAS_LIB=1)
  ADD_LIBRARY(project_edge_lib
          ${project_edge_src})
  TARGET_LINK_LIBRARIES(project_edge_lib
          ${project_edge_LIBRARIES}
  )

  ENABLE_TESTING()
  INCLUDE(GoogleTest)
  ADD_SUBDIRECTORY(test)
endif()
