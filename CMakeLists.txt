CMAKE_MINIMUM_REQUIRED(VERSION 3.11)

# ========================================================
# = Project Options
# ========================================================
OPTION(ENABLE_CUDA "Enable CUDA support" ON)
OPTION(ENABLE_IMGUI "Enable Dear IMGUI support" ON)
OPTION(ENABLE_QT "Enable Qt UI" OFF)
OPTION(ENABLE_OPENGL "Enable OpenGL support" ON)
OPTION(ENABLE_OPENCV "Enable OpenCV support" ON)
OPTION(ENABLE_BOOST "Enable Boost support" ON)
OPTION(ENABLE_FREENECT2 "Enable Kinectv2 Drivers" ON) # not functional atm
OPTION(PACKAGE_TESTS "Build the tests" ON)
OPTION(PACKAGE_RUNTIME "Build runtime files" OFF)

# ========================================================
# = Project Settings
# ========================================================
#ADD_DEFINITIONS(-DWITH_KINECTV2)
#ADD_DEFINITIONS(-DRESIZE_KINECTV2_FRAME)
IF(-DWITH_KINECTV2)
  IF(-DRESIZE_KINECTV2_FRAME)
    ADD_DEFINITIONS(-DVIEWPORT_WIDTH=640 -DVIEWPORT_HEIGHT=480)
  ELSE()
    ADD_DEFINITIONS(-DVIEWPORT_WIDTH=512 -DVIEWPORT_HEIGHT=424)
  ENDIF()
ELSE()
  ADD_DEFINITIONS(-DVIEWPORT_WIDTH=640 -DVIEWPORT_HEIGHT=480 -DRESIZE_KINECTV2_FRAME)
ENDIF(-DWITH_KINECTV2)


# ========================================================
# = Setup Compiler Parameters
# ========================================================
if(WIN32 AND NOT MINGW)
  if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "d")
  endif()
endif()

IF(NOT DEFINED CMAKE_BUILD_TYPE)
  # No effect for multi-configuration generators (e.g. for Visual Studio)
  SET(CMAKE_BUILD_TYPE DEBUG CACHE STRING "Choose: RelWithDebInfo Release Debug MinSizeRel None")
ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -rdynamic")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========================================================
# = Declare Project
# ========================================================
IF(ENABLE_CUDA)
  #ADD_DEFINITIONS(-DWITH_CUDA=1)
  PROJECT(project_edge
          VERSION 0.0.1
          LANGUAGES CUDA CXX)
ELSE()
  PROJECT(project_edge
          VERSION 0.0.1
          LANGUAGES CXX)
ENDIF()

# ========================================================
# = Setup Source Elements
# ========================================================
SET(project_edge_SOURCE
        app.cpp
        context/context.cpp
        context/context_factory.cpp
        res/resource_mgr.cpp
        viewer/gl_viewport.cpp
        viewer/gl_camera.cpp
        shader.cpp

        filter/gl_edge_disc_filter.cpp
        filter/gl_depth_img_filter.cpp
        filter/contour_processor.cpp

        component/contour_policy.cpp
        component/depth_img_policy.cpp
        component/cuda_contour_operations.cu
        component/cuda_depth_img_opr.cu

        dataflow/pipeline.cpp
        dataflow/pipeline_filter.cpp

        utils/helper.cpp
        # utils/cuda_helper.cu
        utils/wiener_filter.cpp


        draw/drawable.cpp
        draw/line_sgmnt.cpp
        draw/line_pair.cpp

)
SET(project_edge_HEADERS
        app.h
        context.h
        frame.h
        viewer.h
        cuda_sobel_filter.h
        cuda_sobel_filter_kernel.h
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


SET(project_edge_SOURCE_TEST
)
SET(project_edge_HEADERS_TEST
)

IF(PACKAGE_TESTS)
  PREPEND(project_edge_src_test "${CMAKE_SOURCE_DIR}/src" ${project_edge_SOURCE_TEST})
  PREPEND(project_edge_hdr_test "${CMAKE_SOURCE_DIR}/src" ${project_edge_HEADERS_TEST})
ENDIF()

SET(project_edge_VENDOR_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/extern")

SET(project_edge_INCLUDE_DIR
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src"
)

INCLUDE_DIRECTORIES(
        ${project_edge_INCLUDE_DIR}
        "${CMAKE_SOURCE_DIR}/extern/nvidia-common/inc"
        "${CMAKE_SOURCE_DIR}/src/dataflow"
)

# ========================================================
# = Cuda Support
# ========================================================
IF(ENABLE_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
  find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

  INCLUDE_DIRECTORIES(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  LIST(APPEND project_edge_LIBRARIES
          ${CUDART_LIBRARY})
  #LIST(APPEND project_edge_src
   #       cuda_ransac.cu)
ENDIF()

# ========================================================
# = Qt
# ========================================================
IF(ENABLE_QT)
  SET(CMAKE_AUTOMOC ON)
  SET(CMAKE_AUTORCC ON)
  SET(CMAKE_AUTOUIC ON)

  FIND_PACKAGE(Qt5 COMPONENTS Widgets REQUIRED)
ENDIF()

# ========================================================
# = Dear IMGUI
# ========================================================
IF(ENABLE_IMGUI)
  INCLUDE_DIRECTORIES( "${project_edge_VENDOR_DIR}/imgui" )
  INCLUDE_DIRECTORIES( "${project_edge_VENDOR_DIR}/imgui/examples" )
  SET(imgui_src
          "${project_edge_VENDOR_DIR}/imgui/imgui.cpp"
          "${project_edge_VENDOR_DIR}/imgui/imgui_widgets.cpp"
          "${project_edge_VENDOR_DIR}/imgui/imgui_draw.cpp"
          "${project_edge_VENDOR_DIR}/imgui/examples/imgui_impl_opengl3.cpp"
          "${project_edge_VENDOR_DIR}/imgui/examples/imgui_impl_glfw.cpp")
  LIST(APPEND project_edge_src ${imgui_src})
  MESSAGE(STATUS "Including Dear imgui ${project_edge_src}")
ENDIF()

# ========================================================
# = OpenGL Library
# ========================================================
IF(ENABLE_OPENGL)
  # Prefer OpenGLX Over Legacy
  set(OpenGL_GL_PREFERENCE "GLVND")
  FIND_PACKAGE(GLEW REQUIRED)
  IF(NOT ${GLEW_FOUND})
    MESSAGE(ERROR "Failed to find GLEW package.")
  ENDIF()

  FIND_PACKAGE(glfw3 REQUIRED)
  FIND_PACKAGE(OpenGL REQUIRED)
  ADD_SUBDIRECTORY("${project_edge_VENDOR_DIR}/glad")
  INCLUDE_DIRECTORIES(
    ${GLFW3_INCLUDE_DIRS}
    ${flextGL_INCLUDE_DIRS}
    ${glad_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
  )

  MESSAGE(STATUS "glad libs ${glad_INCLUDE_DIRS}")
  LIST(APPEND project_edge_DLLS ${GLFW3_DLL})
  LIST(APPEND project_edge_src
    ${flextGL_SOURCES}
  )
  LIST(APPEND project_edge_LIBRARIES
    ${GLFW3_LIBRARIES}
    ${OPENGL_gl_LIBRARY}
    ${glad_LIBRARIES}
    ${GLEW_LIBRARIES}
      OpenGL::GL
    glfw
    ${CMAKE_DL_LIBS} ## Required for dlclose | dlopen
  )
  ADD_DEFINITIONS(-DEXAMPLES_WITH_OPENGL_SUPPORT=1)
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
  SET(Boost_NO_BOOST_CMAKE ON) # OSX
  FIND_PACKAGE(Boost
          REQUIRED
          COMPONENTS log thread timer)
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
  MESSAGE("Finding freenect2")
  FIND_PACKAGE(freenect2
          REQUIRED)
  IF(freenect2_FOUND)
    INCLUDE_DIRECTORIES(${freenect2_INCLUDE_DIR})
    LIST(APPEND project_edge_LIBRARIES
            ${freenect2_LIBRARIES})
    LIST(APPEND project_edge_DLLS
            ${freenect2_DLLS}
            )
      IF(ENABLE_PROFILING)
        SET(LIBFREENECT2_WITH_PROFILING 1)
      ENDIF()
  ENDIF()
ENDIF(ENABLE_FREENECT2)

# ========================================================
# = Target Runtime Declarations
# ========================================================
IF(PACKAGE_RUNTIME)
  ADD_EXECUTABLE(project_edge
          ${project_edge_src})
  TARGET_COMPILE_FEATURES(project_edge
          PUBLIC cxx_std_14)
  SET_TARGET_PROPERTIES(project_edge
          PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  TARGET_LINK_LIBRARIES(project_edge
    ${project_edge_LIBRARIES}
  )
  TARGET_INCLUDE_DIRECTORIES(
          project_edge
          PRIVATE
          ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
ENDIF()

# ========================================================
# = Google Test Library
# ========================================================
IF(PACKAGE_TESTS)
  ADD_DEFINITIONS(-DAS_LIB=1)
  ADD_LIBRARY(project_edge_lib
          ${project_edge_src})
  TARGET_COMPILE_FEATURES(project_edge_lib
          PUBLIC cxx_std_14)
  SET_TARGET_PROPERTIES(project_edge_lib
          PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  TARGET_LINK_LIBRARIES(project_edge_lib
          ${project_edge_LIBRARIES}
  )
  TARGET_INCLUDE_DIRECTORIES(
          project_edge_lib
          PRIVATE
          ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

  ENABLE_TESTING()
  INCLUDE(GoogleTest)
  ADD_SUBDIRECTORY(test)
endif()
