add_library(split-lookahead-wire STATIC split-wire.cpp)
target_include_directories(split-lookahead-wire PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(split-lookahead-wire PUBLIC cxx_std_17)

add_executable(llama-split-lookahead split-lookahead.cpp)
target_link_libraries(llama-split-lookahead PRIVATE llama split-lookahead-wire vendor::hash Threads::Threads)
target_compile_features(llama-split-lookahead PRIVATE cxx_std_17)

if (LLAMA_BUILD_TESTS)
    add_executable(test-split-wire test-split-wire.cpp)
    target_link_libraries(test-split-wire PRIVATE split-lookahead-wire Threads::Threads)
    add_test(NAME test-split-wire COMMAND test-split-wire)

    add_executable(test-split-stage test-stage.cpp)
    target_link_libraries(test-split-stage PRIVATE llama)
    target_compile_features(test-split-stage PRIVATE cxx_std_17)
endif()
