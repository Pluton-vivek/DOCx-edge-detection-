cmake_minimum_required(VERSION 3.22.1)

project("document_detector")

# Include the headers directory
include_directories(include)

# Find Android libraries
find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)

# Import OpenCV via Prefab (AAR)
find_package(OpenCV REQUIRED CONFIG)
# The prefab AAR exposes target opencv::opencv_java (or OpenCV::opencv_java4, we will try opencv::opencv_java first)

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
        OpenCV::opencv_java4
        ${log-lib}
        ${jnigraphics-lib}
)