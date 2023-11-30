add_mlir_conversion_library(XeGPUToSPIRV
  XeGPUToSPIRV.cpp
  XeGPUToSPIRVPass.cpp
  TypeConverter.cpp
  Utility.cpp

  ADDITIONAL_HEADER_DIRS
   ${PROJECT_SOURCE_DIR}/include/triton/Conversion/XeGPUToSPIRV

  DEPENDS
  XeGPUToSPIRVConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSPIRVDialect
  MLIRSPIRVConversion
  MLIRSupport
  MLIRTransforms
  MLIRGPUOps
  MLIRGPUToNVVMTransforms
  MLIRGPUTransforms
  XeGPUIR
  TritonAnalysis
  TritonGPUTransforms
  )
