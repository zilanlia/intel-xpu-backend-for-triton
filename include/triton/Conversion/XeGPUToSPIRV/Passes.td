#ifndef XEGPU_CONVERSION_PASSES
#define XEGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def ConvertGPUXToSPIRV : Pass<"convert-xegpu-to-spirv", "mlir::ModuleOp"> {
    let summary = "Convert XeGPU to SPIRV";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertXeGPUToSPIRVPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::xegpu::XeGPUDialect",
                             "mlir::spirv::SPIRVDialect"];

    let options = [
    Option<"computeCapability", "compute-capability",
            "int32_t", /*default*/"80",
            "device compute capability">
    ];
}

#endif
