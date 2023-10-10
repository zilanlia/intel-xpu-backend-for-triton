#ifndef XEGPU_CONVERSION_PASSES_H
#define XEGPU_CONVERSION_PASSES_H

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "triton/Conversion/XeGPUToSPIRV/XeGPUToSPIRVPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/XeGPUToSPIRV/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
