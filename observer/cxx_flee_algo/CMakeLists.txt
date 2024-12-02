# 최소 CMake 버전 설정
cmake_minimum_required(VERSION 3.10)

# 프로젝트 이름 및 버전
project(FleeAlgorithm VERSION 1.0)

# 옵션 설정 (Windows용 DLL 생성 여부)
option(BUILD_SHARED_LIBS "Build as shared library" ON)

# 파일 추가
# 소스 파일
set(SOURCES flee_algorithm.cpp)

# 헤더 파일
set(HEADERS flee_algorithm.h)

# 공유 라이브러리 생성
add_library(FleeAlgorithm SHARED ${SOURCES})

# 헤더 파일 포함 경로 추가
target_include_directories(FleeAlgorithm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# 크로스 플랫폼 설정
if(WIN32)
    # Windows에서 DLL 빌드
    message(STATUS "Configuring for Windows")
    target_compile_definitions(FleeAlgorithm PRIVATE BUILD_DLL)
elseif(UNIX)
    # Linux에서 .so 빌드
    message(STATUS "Configuring for Linux")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON) # PIC 설정 (공유 라이브러리 필요)
endif()

# 설치 옵션
install(TARGETS FleeAlgorithm
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include)