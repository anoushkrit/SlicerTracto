cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerTracto)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/SlicerTracto")
set(EXTENSION_CATEGORY "SlicerTracto")
set(EXTENSION_CONTRIBUTORS "Gagan Gupta (IIT MANDI), Mahir Jain (IIT MANDI), Ankita Joshi (IIT MANDI), Anoushkrit Goel (IIT MANDI), Aditya Nigam (IIT MANDI)")
set(EXTENSION_DESCRIPTION "The slicer extension is a specialized toolkit for advanced neuroimaging analysis. It features three core modules:

    Generate FODF: Computes Fiber Orientation Distribution Functions (FODFs) from diffusion MRI data, enabling accurate modeling of complex fiber structures in the brain.
    Tractography: Facilitates the visualization and reconstruction of brain white matter pathways using advanced tractography techniques.
    Metric Analysis: Provides tools for quantitative analysis of diffusion and tractography metrics, enabling insights into structural connectivity and tissue properties.

This extension is ideal for researchers in neuroscience and clinical imaging.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/MTP.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/MTP/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(MTP)
add_subdirectory(DMRI_Tractography)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
