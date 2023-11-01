#ifndef TRITON_CONVERSION_XEGPU_TO_SPIRV_TYPECONVERTER_H
#define TRITON_CONVERSION_XEGPU_TO_SPIRV_TYPECONVERTER_H

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "triton/Conversion/MLIRTypes.h"

using namespace mlir;
using namespace mlir::triton;

class XeGPUToSPIRVTypeConverter : public SPIRVTypeConverter {
public:
  using TypeConverter::convertType;

  XeGPUToSPIRVTypeConverter(spirv::TargetEnvAttr &targetAttr,
                                SPIRVConversionOptions &option);

  Type convertXeGPUMemRefType(mlir::MemRefType type);

  Type convertXeGPUVectorType(mlir::VectorType type);
};

#endif
