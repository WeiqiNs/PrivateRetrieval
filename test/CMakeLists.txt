find_package(GTest QUIET)
if (NOT GTest_FOUND)
    FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.17.0
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif ()
include(GoogleTest)

foreach (source IN ITEMS test_format.cpp test_search.cpp)
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE privret_core GTest::gtest_main)
    gtest_discover_tests(${name})
endforeach ()

add_executable(privret_fixture fixture.cpp)
target_link_libraries(privret_fixture PRIVATE privret_core)

add_test(NAME smoke_fixture COMMAND privret_fixture ${CMAKE_CURRENT_BINARY_DIR}/smoke)
set_tests_properties(smoke_fixture PROPERTIES FIXTURES_SETUP smoke)
add_test(NAME smoke COMMAND ${PROJECT_SOURCE_DIR}/scripts/search.sh
        --bin $<TARGET_FILE:privret>
        --data ${CMAKE_CURRENT_BINARY_DIR}/smoke
        --out ${CMAKE_CURRENT_BINARY_DIR}/smoke/out
        --dim 4 --procs 3 --top-k 5 --max-queries 2
)
set_tests_properties(smoke PROPERTIES FIXTURES_REQUIRED smoke)
