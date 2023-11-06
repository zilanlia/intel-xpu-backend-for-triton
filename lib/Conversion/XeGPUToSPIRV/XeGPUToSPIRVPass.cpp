//===- GPUToSPIRVPass.cpp -  --------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file extends upstream GPUToSPIRV Pass that converts GPU ops to SPIR-V
/// by adding more conversion patterns like SCF, math and control flow. This
/// pass only converts gpu.func ops inside gpu.module op.
///
//===----------------------------------------------------------------------===//
#include "triton/Conversion/XeGPUToSPIRV/XeGPUToSPIRVPass.h"
#include "triton/Dialect/XeGPU/IR/XeGPUOps.h"

#include <mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h>
#include <mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h>
#include <mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h>
#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h>
#include <mlir/Conversion/MathToSPIRV/MathToSPIRV.h>
#include <mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h>
#include <mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h>
#include <mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "XeGPUToSPIRV.h"
#include "TypeConverter.h"

using namespace mlir;
using namespace mlir::triton;
// using namespace mlir::triton::xegpu;

#define GEN_PASS_CLASSES
#include "triton/Conversion/XeGPUToSPIRV/Passes.h.inc"

class XeGPUSPIRVFunctionConversionTarget : public ConversionTarget {
public:
  explicit XeGPUSPIRVFunctionConversionTarget(MLIRContext &ctx, SPIRVTypeConverter& typeConverter)
          : ConversionTarget(ctx) {
    addLegalDialect<spirv::SPIRVDialect>();
    addIllegalOp<mlir::gpu::GPUFuncOp>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class XeGPUSPIRVConversionTarget : public ConversionTarget {
public:
  explicit XeGPUSPIRVConversionTarget(MLIRContext &ctx, XeGPUToSPIRVTypeConverter& typeConverter)
          : ConversionTarget(ctx) {

    addIllegalDialect<triton::xegpu::XeGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addIllegalDialect<::mlir::arith::ArithDialect>();
    addIllegalDialect<::mlir::vector::VectorDialect>();
    addIllegalDialect<::mlir::memref::MemRefDialect>();
    addLegalDialect<spirv::SPIRVDialect>();
    addLegalOp<UnrealizedConversionCastOp>();

    //vc-backend do not support spirv::CL extension
    addIllegalOp<spirv::CLExpOp>();
    addIllegalOp<spirv::CLFMaxOp>();
    addIllegalOp<spirv::FConvertOp>();

    addDynamicallyLegalOp<spirv::ConstantOp>([](spirv::ConstantOp op) -> bool {
      ::mlir::Attribute value = op.getValue();
      
      if(DenseElementsAttr denseValue = dyn_cast<DenseElementsAttr>(value)){
        Type type = denseValue.getType();

        if(auto vectorType = type.cast<VectorType>()){
          auto shape = vectorType.getShape();
          // convert 2d constatn val to 1D to avoid error for spirv
          if(shape.size() >= 2){
            return false;
          }
        }
      }
      return true;
    });

    addDynamicallyLegalOp<spirv::VectorShuffleOp>([](spirv::VectorShuffleOp op) -> bool {
      Value vector1 = op.getVector1();
      Type type = vector1.getType();

      if(auto vectorType = type.cast<VectorType>()){
        auto shape = vectorType.getShape();
        // convert 2d constatn val to 1D to avoid error for spirv
        if(shape.size() >= 2){
          return false;
        }
      }
      
      return true;
    });
  }
};

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
struct FuncOpConversion : public OpConversionPattern<mlir::gpu::GPUFuncOp> {
  FuncOpConversion(XeGPUToSPIRVTypeConverter &converter, MLIRContext *context, int numWarps,
                   PatternBenefit benefit)
      : OpConversionPattern<mlir::gpu::GPUFuncOp>(converter, context, benefit), numWarps(numWarps) {}

  LogicalResult
  matchAndRewrite(mlir::gpu::GPUFuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto mod = dyn_cast<mlir::gpu::GPUModuleOp>(funcOp->getParentOp());
    auto mod1 = dyn_cast<mlir::ModuleOp>(mod->getParentOp());

    if (!mod)
      return failure();

    auto fnType = funcOp.getFunctionType();
    if (fnType.getNumResults() > 1)
      return failure();

    int num_inputs = fnType.getNumInputs();

    mlir::MLIRContext *context = funcOp.getContext();
    spirv::Capability caps_opencl[1];
    spirv::Extension exts_opencl[1];
    auto triple = spirv::VerCapExtAttr::get(
          spirv::Version::V_1_4, caps_opencl, exts_opencl, context);
    auto targetAttr = spirv::TargetEnvAttr::get(
          triple, spirv::getDefaultResourceLimits(context),
          spirv::ClientAPI::OpenCL,
          spirv::Vendor::Unknown,
          spirv::DeviceType::Unknown,
          spirv::TargetEnvAttr::kUnknownDeviceID);

    mlir::SPIRVConversionOptions options;
    options.use64bitIndex = true;
    XeGPUToSPIRVTypeConverter xeGPUtypeConverter(targetAttr, options);

    TypeConverter::SignatureConversion signatureConverter(num_inputs);
    for (const auto &argType : enumerate(fnType.getInputs())) {
      // llvm::outs() << "\n\nFuncOp argType.value(): "<<argType.value();
      auto convertedType = xeGPUtypeConverter.convertType(argType.value());
      // llvm::outs() << "\n\nFuncOp convertedType: "<<convertedType;
      if (!convertedType)
        return failure();
      signatureConverter.addInputs(argType.index(), convertedType);
    }

    Type resultType;
    if (fnType.getNumResults() == 1) {
      resultType = xeGPUtypeConverter.convertType(fnType.getResult(0));
      if (!resultType)
        return failure();
    }

    rewriter.setInsertionPointToStart(mod1.getBody());
    // Create the converted spv.func op.
    auto newFuncOp = rewriter.create<spirv::FuncOp>(
            funcOp.getLoc(), funcOp.getName(),
            rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                                     resultType ? TypeRange(resultType)
                                                : TypeRange()));

    // Set the SPIRV kernel entry point
    newFuncOp->setAttr(spirv::getEntryPointABIAttrName(), spirv::EntryPointABIAttr::get(getContext(), nullptr, std::nullopt));

    llvm::outs()<<"\n\nset attribute\n";
    // Copy over all attributes other than the function name and type.
    for (const auto &namedAttr : funcOp->getAttrs()) {
        if (namedAttr.getName() != funcOp.getFunctionTypeAttrName() &&
            namedAttr.getName() != SymbolTable::getSymbolAttrName() &&
            namedAttr.getName() != funcOp.getArgAttrsAttrName())
        newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    // llvm::outs()<<"\n\nfuncOp.getAllArgAttrs()\n";
    ArrayAttr attrs = funcOp.getAllArgAttrs();

    // llvm::outs()<<"\n\nrewriter.inlineRegionBefore\n";
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(
            &newFuncOp.getBody(), *getTypeConverter(), &signatureConverter)))
      return failure();

    rewriter.eraseOp(mod);
    // llvm::outs()<<"\n\nmod: "<<mod<<"\n";
    // llvm::outs()<<"\n\nmod1: "<<mod1<<"\n";
    return success();
  }

private:
  int numWarps{0};
};

/// Pass to lower XeGPU Dialect to SPIR-V. 
/// 1) Create a spirv::ModuleOp, and clone the function into spirv::ModuleOp
/// (the original function is still needed by the gpu::LaunchKernelOp, so cannot
/// replace it).
///
/// 2) Lower the body of the spirv::ModuleOp.
class GPUXToSPIRVPass : public ConvertGPUXToSPIRVBase<GPUXToSPIRVPass> {
public:
  explicit GPUXToSPIRVPass(bool mapMemorySpace)
      : mapMemorySpace(mapMemorySpace) {}
  void runOnOperation() override;

private:
  bool mapMemorySpace;
};

void GPUXToSPIRVPass::runOnOperation() {
  mlir::MLIRContext *context = &getContext();
  mlir::ModuleOp module = getOperation();

  llvm::SmallVector<mlir::Operation *, 1> gpuModules;
  mlir::OpBuilder builder(context);

  auto gpuModule = getOperation();

  spirv::Capability caps_opencl[] = {
            spirv::Capability::Addresses,
            spirv::Capability::Float16Buffer,
            spirv::Capability::Int64,
            spirv::Capability::Int16,
            spirv::Capability::Int8,
            spirv::Capability::Kernel,
            spirv::Capability::Linkage,
            spirv::Capability::Vector16,
            spirv::Capability::GenericPointer,
            spirv::Capability::Groups,
            spirv::Capability::Float16,
            spirv::Capability::Float64,
            spirv::Capability::AtomicFloat32AddEXT,
            spirv::Capability::ExpectAssumeKHR,
            spirv::Capability::VectorComputeINTEL,
            spirv::Capability::VectorAnyINTEL
  };
  spirv::Extension exts_opencl[] = {
          // spirv::Extension::SPV_EXT_shader_atomic_float_add,
          spirv::Extension::SPV_KHR_expect_assume,
          spirv::Extension::SPV_INTEL_vector_compute};
  auto triple = spirv::VerCapExtAttr::get(
          spirv::Version::V_1_4, caps_opencl, exts_opencl, context);
  auto targetAttr = spirv::TargetEnvAttr::get(
          triple, spirv::getDefaultResourceLimits(context),
          spirv::ClientAPI::OpenCL,
          spirv::Vendor::Unknown,
          spirv::DeviceType::Unknown,
          spirv::TargetEnvAttr::kUnknownDeviceID);

  {
    // auto targetAttr = mlir::spirv::lookupTargetEnvOrDefault(gpuModule);
    // llvm::outs()<<"\n\ntargetAttr: " << targetAttr <<"\n";
    std::unique_ptr<mlir::ConversionTarget> target =
        mlir::SPIRVConversionTarget::get(targetAttr);

    mlir::RewritePatternSet patterns(context);
    mlir::SPIRVConversionOptions options;
    options.use64bitIndex = true;

    XeGPUToSPIRVTypeConverter typeConverter(targetAttr, options);
    XeGPUSPIRVFunctionConversionTarget funcTarget(*context, typeConverter);
    XeGPUSPIRVConversionTarget spirvTarget(*context, typeConverter);
  
    RewritePatternSet funcPatterns(context);
    funcPatterns.add<FuncOpConversion>(typeConverter, context, 0, 1 /*benefit*/);

    if (failed(
            applyPartialConversion(gpuModule, funcTarget, std::move(funcPatterns))))
      return signalPassFailure();

    mlir::OpBuilder builder(gpuModule);
    llvm::SmallVector<mlir::Operation *, 16> eraseOps;

    //------- Upstream Conversion------------
    mlir::populateGPUToSPIRVPatterns(typeConverter, patterns);
    mlir::arith::populateArithToSPIRVPatterns(typeConverter, patterns);
    mlir::populateMemRefToSPIRVPatterns(typeConverter, patterns);
    mlir::populateFuncToSPIRVPatterns(typeConverter, patterns);
    // ---------------------------------------

    // triton GPUToSPIRV extension
    mlir::ScfToSPIRVContext scfToSpirvCtx;
    mlir::populateSCFToSPIRVPatterns(typeConverter, scfToSpirvCtx, patterns);
    mlir::cf::populateControlFlowToSPIRVPatterns(typeConverter, patterns);
    mlir::populateMathToSPIRVPatterns(typeConverter, patterns);
    mlir::populateVectorToSPIRVPatterns(typeConverter, patterns);
    populateXeGPUToVCIntrinsicsPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(gpuModule, spirvTarget, std::move(patterns)))){
      return signalPassFailure();
    }
  }
};

namespace mlir {
namespace triton {

std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertXeGPUToSPIRVPass(bool mapMemorySpace) {
  return std::make_unique<GPUXToSPIRVPass>(mapMemorySpace);
}

} // namespace triton
} // namespace mlir

