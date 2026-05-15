cmake_minimum_required(VERSION 3.22.1)

project("document_detector")

# Include the headers directory
include_directories(include)

# Find Android libraries
find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)

# Import OpenCV
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../sdk/native/jni)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../sdk/native/jni/include)
find_package(OpenCV REQUIRED java)

# Add our source files
add_library(
        document_detector
        SHARED
        document_detector.cpp
        android_utils.cpp
        ColorSimplificationTransform.cpp
        DocumentDetector.cpp
        # DocumentOCR.cpp
        # QRCode.cpp
        Utils.cpp
        WhitePaperTransform.cpp
        WhitePaperTransform2.cpp
)

# Link the libraries together
target_link_libraries(
        document_detector
        ${OpenCV_LIBS}
        ${log-lib}
        ${jnigraphics-lib}
)