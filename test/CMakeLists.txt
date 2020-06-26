CMAKE_MINIMUM_REQUIRED(VERSION 3.11)


add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")

MARK_AS_ADVANCED(
        BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
        gmock_build_tests gtest_build_samples gtest_build_tests
        gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)

SET_TARGET_PROPERTIES(gtest PROPERTIES FOLDER extern)
SET_TARGET_PROPERTIES(gtest_main PROPERTIES FOLDER extern)
SET_TARGET_PROPERTIES(gmock PROPERTIES FOLDER extern)
SET_TARGET_PROPERTIES(gmock_main PROPERTIES FOLDER extern)

MACRO(PACKAGE_ADD_TEST TESTNAME)
    # create an exectuable in which the tests will be stored
    ADD_EXECUTABLE(${TESTNAME}
            ${ARGN}
            ${project_edge_src_test}
    )

    TARGET_INCLUDE_DIRECTORIES(${TESTNAME} PUBLIC
            ${project_edge_INCLUDE_DIR}
    )

    TARGET_COMPILE_FEATURES(${TESTNAME}
            PUBLIC cxx_std_14)

    # link the Google test infrastructure, mocking library, and a default main fuction to
    # the test executable.  Remove g_test_main if writing your own main function.
    TARGET_LINK_LIBRARIES(${TESTNAME} project_edge_lib gtest gmock gtest_main)

    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TESTNAME}
            # set a working directory so your project root so that you can find test data via paths relative to the project root
            WORKING_DIRECTORY ${PROJECT_DIR}
            PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
            )
    SET_TARGET_PROPERTIES(${TESTNAME} PROPERTIES FOLDER tests)
ENDMACRO()

# PACKAGE_ADD_TEST(app_test freenect_filter_test.cpp)
PACKAGE_ADD_TEST(app_test cuda_contour_operation_test.cpp)
