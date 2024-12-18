cmake_minimum_required(VERSION 3.19)
project(neuro_fuzzy)

find_package(nlohmann_json REQUIRED)

add_library(sanitizers INTERFACE)
target_compile_options(sanitizers INTERFACE -fsanitize=address -fsanitize=undefined)
target_link_options(sanitizers INTERFACE -fsanitize=address -fsanitize=undefined)

file(GLOB_RECURSE NEURO_FUZZY_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/source/**/*.cpp")

add_library(neuro_fuzzy SHARED
	${NEURO_FUZZY_SOURCES}
)

target_link_libraries(neuro_fuzzy PUBLIC nlohmann_json::nlohmann_json)

target_compile_options(neuro_fuzzy PRIVATE
	-Wall
	-Wextra
	-Wno-unused-parameter
	-Wno-sign-compare
	-Wno-inconsistent-missing-override
)

set_target_properties(${PROJECT_NAME} PROPERTIES
	CXX_STANDARD 23
	CXX_STANDARD_REQUIRED ON
)

add_executable(experiment "${CMAKE_CURRENT_SOURCE_DIR}/source/main.cpp")
target_link_libraries(experiment neuro_fuzzy)
