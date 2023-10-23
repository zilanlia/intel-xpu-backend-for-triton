//===- XeGPUToSPIRV.cpp -  --------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements patterns to convert XeGPU to SPIRV with
/// VC-Intrinsics/JointMatrix
///
//===----------------------------------------------------------------------===//
#include "XeGPUToSPIRV.h"
#include "triton/Dialect/XeGPU/IR/XeGPUOps.h"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVAttributes.h.inc>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

#include <mlir/Pass/Pass.h>

using namespace mlir;
using namespace mlir::triton::xegpu;
using namespace mlir::triton;

Value createConstantI32(Location loc, PatternRewriter &rewriter, int32_t v) {
  auto i32ty = rewriter.getIntegerType(32);
  return rewriter.create<spirv::ConstantOp>(loc, i32ty,
                                           IntegerAttr::get(i32ty, v));
}

Value createConstantI64(Location loc, PatternRewriter &rewriter, int64_t v) {
  auto i64ty = rewriter.getIntegerType(64);
  return rewriter.create<spirv::ConstantOp>(loc, i64ty,
                                           IntegerAttr::get(i64ty, v));
}

#define zext(...) rewriter.create<spirv::UConvertOp>(loc, __VA_ARGS__)
#define logic_shl(...) rewriter.create<spirv::ShiftLeftLogicalOp>(loc, __VA_ARGS__)
#define bitwise_or(...) rewriter.create<spirv::BitwiseOrOp>(loc, __VA_ARGS__)
#define bitwise_and(...) rewriter.create<spirv::BitwiseAndOp>(loc, __VA_ARGS__)
#define i32_val(...) createConstantI32(loc, rewriter, __VA_ARGS__)
#define i64_val(...) createConstantI64(loc, rewriter, __VA_ARGS__)
#define i16_val(value) rewriter.create<spirv::ConstantOp>(loc, rewriter.getIntegerType(16), rewriter.getI16IntegerAttr(value))
#define i8_val(value) rewriter.create<spirv::ConstantOp>(loc, rewriter.getIntegerType(8), rewriter.getI8IntegerAttr(value))
#define i1_val(value) rewriter.create<spirv::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getBoolAttr(value))

/// @brief encodeVectorType(xxx, 8x8x2xf16, true) returns ["v64i32", 64xi32]
std::pair<std::string, VectorType>
encodeVectorType(ConversionPatternRewriter &rewriter, VectorType type,
                 bool use64bitData = false) {
  auto elemType = type.getElementType();
  auto bitWidth = elemType.getIntOrFloatBitWidth();
  int size = type.getNumElements() * bitWidth / 32;
  if (use64bitData) {
    size /= 2;
  }
  std::string str;
  switch (size) {
  case 1:
    str += "v1";
    break;
  case 2:
    str += "v2";
    break;
  case 4:
    str += "v4";
    break;
  case 8:
    str += "v8";
    break;
  case 16:
    str += "v16";
    break;
  case 32:
    str += "v32";
    break;
  case 64:
    str += "v64";
    break;
  case 128:
    str += "v128";
    break;
  default:
    assert(0 && "add more support");
    break;
  }
  if (use64bitData) {
    str += "i64";
    elemType = rewriter.getI64Type();
  } else if (elemType == rewriter.getF32Type())
    str += "f32";
  else if (elemType == rewriter.getF16Type()) {
    str += "i32";
    elemType = rewriter.getI32Type();
  } else
    assert(0 && "add more support");
  auto newType = VectorType::get(size, elemType);
  return std::make_pair(str, newType);
}
unsigned encodeDataum(Type type) {
  switch (type.getIntOrFloatBitWidth()) {
  case 8:
    return 1;
  case 16:
    return 2;
  case 32:
    return 3;
  case 64:
    return 4;
  default:
    assert(0 && "add more support");
    return 0;
  }
}

template <typename OpType> unsigned encodeCacheHint(OpType op) {
  auto l1hint = op.getL1Hint();
  // auto l2hint = op.getL2Hint();
  auto l3hint = op.getL3Hint();
  constexpr bool isWrite = std::is_same_v<OpType, StoreNDOp> ||
                           std::is_same_v<OpType, StoreScatterOp>;
  unsigned cacheHint = 1;
  if constexpr (!isWrite) {
    auto l1CacheValue =
        l1hint.has_value() ? l1hint.value() : xegpu::CacheReadHint::UNCACHED;
    auto l3CacheValue =
        l3hint.has_value() ? l3hint.value() : xegpu::CacheReadHint::UNCACHED;
    if (l1CacheValue == xegpu::CacheReadHint::UNCACHED) {
      if (l3CacheValue == xegpu::CacheReadHint::UNCACHED)
        cacheHint = 1;
      else if (l3CacheValue == xegpu::CacheReadHint::CACHED)
        cacheHint = 2;
    } else if (l1CacheValue == xegpu::CacheReadHint::CACHED) {
      if (l3CacheValue == xegpu::CacheReadHint::UNCACHED)
        cacheHint = 3;
      else if (l3CacheValue == xegpu::CacheReadHint::CACHED)
        cacheHint = 4;
    } else if (l1CacheValue == xegpu::CacheReadHint::STREAMING) {
      if (l3CacheValue == xegpu::CacheReadHint::UNCACHED)
        cacheHint = 5;
      else if (l3CacheValue == xegpu::CacheReadHint::CACHED)
        cacheHint = 6;
    } else if (l1CacheValue == xegpu::CacheReadHint::READ_INVALIDATE) {
      if (l3CacheValue == xegpu::CacheReadHint::CACHED)
        cacheHint = 7;
    }
  } else {
    auto l1CacheValue =
        l1hint.has_value() ? l1hint.value() : xegpu::CacheWriteHint::UNCACHED;
    auto l3CacheValue =
        l3hint.has_value() ? l3hint.value() : xegpu::CacheWriteHint::UNCACHED;
    if (l1CacheValue == xegpu::CacheWriteHint::UNCACHED) {
      if (l3CacheValue == xegpu::CacheWriteHint::UNCACHED)
        cacheHint = 1;
      else if (l3CacheValue == xegpu::CacheWriteHint::WRITE_BACK)
        cacheHint = 2;
    } else if (l1CacheValue == xegpu::CacheWriteHint::WRITE_THROUGH) {
      if (l3CacheValue == xegpu::CacheWriteHint::UNCACHED)
        cacheHint = 3;
      else if (l3CacheValue == xegpu::CacheWriteHint::WRITE_BACK)
        cacheHint = 4;
    } else if (l1CacheValue == xegpu::CacheWriteHint::STREAMING) {
      if (l3CacheValue == xegpu::CacheWriteHint::UNCACHED)
        cacheHint = 5;
      else if (l3CacheValue == xegpu::CacheWriteHint::WRITE_BACK)
        cacheHint = 6;
    } else if (l1CacheValue == xegpu::CacheWriteHint::WRITE_BACK) {
      if (l3CacheValue == xegpu::CacheWriteHint::WRITE_BACK)
        cacheHint = 7;
    }
  }
  return cacheHint;
}

void lookupOrInsertIntrinsic(ConversionPatternRewriter &rewriter, Operation *op,
                             std::string name, FunctionType funcType) {
  auto funcAttr = StringAttr::get(rewriter.getContext(), name);
  Operation *found = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
  if (!found) {
    OpBuilder::InsertionGuard guard(rewriter);
    // ModuleOp mod = op->getParentOfType<ModuleOp>();
    // llvm::outs()<<"\n\nmod: "<<mod<<"\n";
    //auto kernel = op->getParentOfType<mlir::func::FuncOp>();
    auto kernel = op->getParentOfType<spirv::FuncOp>();
    //llvm::outs()<<"\n\nkernel: "<<kernel<<"\n";
    rewriter.setInsertionPoint(kernel);
    auto func = rewriter.create<spirv::FuncOp>(rewriter.getUnknownLoc(), name, funcType);
    auto linkageTypeAttr =
        rewriter.getAttr<spirv::LinkageTypeAttr>(spirv::LinkageType::Import);
    std::replace(name.begin(), name.end(), '_', '.');
    //name = "\"" + name + "\"";
    //auto linkage = spirv::LinkageAttributesAttr::get(rewriter.getContext(),
    //                                                  name, linkageTypeAttr);
    //func.setLinkageAttributesAttr(linkage);
    //func->setAttr("VectorComputeFunctionINTEL", rewriter.getUnitAttr());
    //auto funcOp = builder.create<spirv::FuncOp>(builder.getUnknownLoc(), funcName, funcType);
    ::llvm::StringRef arrayAttr(name);
    func->setAttr("linkage_attributes", rewriter.getStrArrayAttr({arrayAttr, "Import"}));
    func->setAttr("VectorComputeFunctionINTEL", rewriter.getUnitAttr());
    //llvm::outs()<<"\n\nfunc: "<<func<<"\n";
    //mod.push_back(func);
    //mod = op->getParentOfType<ModuleOp>();
    //llvm::outs()<<"\n\nmod: "<<mod<<"\n";
  }
}

class CreateNdDescToVCPattern : public OpConversionPattern<CreateNdDescOp> {
public:
  using OpConversionPattern<CreateNdDescOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CreateNdDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSource();
    Type type = src.getType();
    // llvm::outs()<<"\n\nCreateNdDescToVCPattern Type: "<<type<<"\n";
    if(isa<mlir::IntegerType>(type)){
      rewriter.replaceOp(op, src);
      // llvm::outs()<<"\n\nCreateNdDescToVCPattern Type isa<mlir::IntegerType>\n";
    }else{
      rewriter.replaceOpWithNewOp<spirv::ConvertPtrToUOp>(
          op, rewriter.getI64Type(), adaptor.getSource());
    }

    return success();
  }
};

template <typename OpType>
class LoadStorePrefetchNdToLsc : public OpConversionPattern<OpType> {
public:
  using OpConversionPattern<OpType>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileType = op.getTensorDesc().getType();
    int rank = tileType.getRank();
    assert(rank <= 2 && "only support 1d/2d load/store/prefetch for now");
    auto loc = op.getLoc();
    ::mlir::VectorType vecType;
    std::string funcName;
    constexpr bool isLoad = std::is_same_v<OpType, LoadNDOp>;
    constexpr bool isPrefetch = std::is_same_v<OpType, PrefetchNDOp>;
    if constexpr (isLoad) {
      vecType = cast<VectorType>(op.getResult().getType());
      funcName = rank == 2 ? "llvm_genx_lsc_load2d_stateless_"
                           : "llvm_genx_lsc_load_stateless_";
    } else if constexpr (isPrefetch) {
      vecType = VectorType::get({8, 16}, rewriter.getF32Type());
      funcName = rank == 2 ? "llvm_genx_lsc_prefetch2d_stateless_i1_i64"
                           : "llvm_genx_lsc_prefetch_stateless_";
    } else {
      vecType = cast<VectorType>(op.getValue().getType());
      funcName = rank == 2 ? "llvm_genx_lsc_store2d_stateless_i1_i64_"
                           : "llvm_genx_lsc_store_stateless_i1_i64_";
    }
    auto createIntConstant = [&](Type type, unsigned value) {
      auto attr = rewriter.getIntegerAttr(type, value);
      return rewriter.create<spirv::ConstantOp>(loc, type, attr);
    };
    auto i8Type = rewriter.getI8Type();
    auto i16Type = rewriter.getI16Type();
    auto i32Type = rewriter.getI32Type();
    auto vnni = false;
    auto transpose = false;
    if constexpr (isLoad) {
      auto vnniValue = op.getVnniAxis();
      vnni = vnniValue.has_value() && vnniValue.value() == 0 ? true : false;
      auto transposeValue = op.getTranspose();
      transpose = transposeValue.has_value() && transposeValue.value()[0] == 1
                      ? true
                      : false;
    }
    auto l1hint = op.getL1Hint();
    // auto l2hint = op.getL2Hint();
    auto l3hint = op.getL3Hint();

    // predicate(true for now)
    auto pred = createIntConstant(rewriter.getI1Type(), 1);
    auto l1CacheHint =
        createIntConstant(i8Type, l1hint.has_value() ? (int)l1hint.value() : 0);
    auto l3CacheHint =
        createIntConstant(i8Type, l3hint.has_value() ? (int)l3hint.value() : 0);
    unsigned dataSize = encodeDataum(vecType.getElementType());
    auto dataum = createIntConstant(i8Type, dataSize);
    auto trans = createIntConstant(i8Type, transpose ? 2 : 1);
    auto base = adaptor.getTensorDesc();
    // number of blocks(1 for now)
    auto nBlks = createIntConstant(i8Type, 1);
    std::string typeStr;
    VectorType newType;
    std::tie(typeStr, newType) = encodeVectorType(rewriter, vecType, rank == 1);
    SmallVector<Value> args;
    if (rank == 2) {
      auto blockWidth = tileType.getShape()[1];
      auto blockHeight = tileType.getShape()[0];
      auto blockW = createIntConstant(i32Type, blockWidth);
      auto blockH = createIntConstant(i32Type, blockHeight);
      auto transform = createIntConstant(i8Type, vnni ? 1 : 0);
      // static memref for now
      auto createDescOp =
          op.getTensorDesc().template getDefiningOp<CreateNdDescOp>();
      auto memType = cast<MemRefType>(createDescOp.getSource().getType());
      unsigned bitWidth = memType.getElementType().getIntOrFloatBitWidth();
      auto surfaceWidth = memType.getShape()[1] * (bitWidth / 8) - 1;
      auto surfaceHeight = memType.getShape()[0] - 1;
      // pitch = width for now
      auto surfacePitch = surfaceWidth;
      auto surfaceW = createIntConstant(i32Type, surfaceWidth);
      auto surfaceH = createIntConstant(i32Type, surfaceHeight);
      auto surfaceP = createIntConstant(i32Type, surfacePitch);
      auto createOffset = [&](unsigned idx) -> Value {
        Value val;
        if (ShapedType::isDynamic(createDescOp.getStaticOffsets()[idx])) {
          val = createDescOp.getOffsets()[idx];
          val = rewriter.create<arith::TruncIOp>(loc, i32Type, val);
        } else {
          val =
              createIntConstant(i32Type, createDescOp.getStaticOffsets()[idx]);
        }
        return val;
      };
      auto offsetX = createOffset(1);
      auto offsetY = createOffset(0);
      args.assign({pred, l1CacheHint, l3CacheHint, dataum, trans, nBlks, blockW,
                   blockH, transform, base, surfaceW, surfaceH, surfaceP,
                   offsetX, offsetY});
      if constexpr (!isLoad && !isPrefetch) {
        args.push_back(adaptor.getValue());
      }
    } else if (rank == 1) {
      auto subOpcode =
          createIntConstant(i8Type, (isLoad || isPrefetch) ? 0 : 4);
      auto addrScale = createIntConstant(i16Type, 1);
      auto immOffset = createIntConstant(i32Type, 0);
      auto dataumSize = createIntConstant(i8Type, 4);
      int lscVecSize = 0;
      int numElts = newType.getNumElements();
      if (numElts <= 4) {
        lscVecSize = numElts;
      } else {
        lscVecSize = log2(numElts) + 2;
      }
      auto vecSize = createIntConstant(i8Type, lscVecSize);
      auto transposed = createIntConstant(i8Type, 2); // transpose
      auto mask = createIntConstant(i8Type, 0);
      auto surface = createIntConstant(i32Type, 0);
      args.assign({
          pred,
          subOpcode,
          l1CacheHint,
          l3CacheHint,
          addrScale,
          immOffset,
          dataumSize,
          vecSize,
          transposed,
          mask,
          base,
      });
      if constexpr (!isLoad && !isPrefetch) {
        auto cast =
            rewriter.create<spirv::BitcastOp>(loc, newType, adaptor.getValue());
        args.push_back(cast);
      }
      args.push_back(surface);
    }
    if constexpr (!isPrefetch)
      funcName += typeStr;
    if constexpr (isLoad) {
      funcName += "_i1_i64";
      auto retType = newType;
      auto funcType =
          rewriter.getFunctionType(ValueRange(args).getTypes(), retType);
      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
      auto funcOp =
          rewriter.create<spirv::FunctionCallOp>(loc, retType, funcName, args);
      if (rank == 2) {
        rewriter.replaceOp(op, funcOp->getResult(0));
      } else {
        auto cast = rewriter.create<spirv::BitcastOp>(loc, op.getType(),
                                                      funcOp->getResult(0));
        rewriter.replaceOp(op, cast->getResult(0));
      }
    } else {
      auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});
      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
      rewriter.create<spirv::FunctionCallOp>(loc, TypeRange(), funcName, args);
      rewriter.eraseOp(op);
    }
    return success();
  }
};

template <typename OpType>
class LoadStorePrefetchNdToRawSend : public OpConversionPattern<OpType> {
public:
  using OpConversionPattern<OpType>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs()<<"\n\nLoadStorePrefetchNdToRawSend: "<<"\n";
    auto createDescOp =
          op.getTensorDesc().template getDefiningOp<CreateNdDescOp>();
    auto memoryScope = createDescOp.getMemoryScope();
    bool usingSLM = memoryScope == xegpu::MemoryScope::SLM;
    llvm::outs()<<"\n\nmemory_scope: "<<memoryScope<<"\n";

    auto tileType = op.getTensorDesc().getType();
    auto rank = tileType.getRank();
    assert(rank <= 2 && "only support 1d/2d load/store/prefetch for now");
    auto loc = op->getLoc();
    constexpr bool isLoad = std::is_same_v<OpType, LoadNDOp>;
    constexpr bool isPrefetch = std::is_same_v<OpType, PrefetchNDOp>;
    auto createIntConstant = [&](Type type, unsigned value) {
      auto attr = rewriter.getIntegerAttr(type, value);
      return rewriter.create<spirv::ConstantOp>(loc, type, attr);
    };

    /// collect common info
    auto i1Type = rewriter.getI1Type();
    auto i8Type = rewriter.getI8Type();
    auto i32Type = rewriter.getI32Type();
    auto i64Type = rewriter.getI64Type();
    auto vnni = false;
    auto transpose = false;
    if constexpr (isLoad) {
      auto vnniValue = op.getVnniAxis();
      vnni = vnniValue.has_value() && vnniValue.value() == 0 ? true : false;
      auto transposeValue = op.getTranspose();
      transpose = transposeValue.has_value() && transposeValue.value()[0] == 1
                      ? true
                      : false;
    }
    auto base = adaptor.getTensorDesc();
    auto elmType = tileType.getElementType();
    VectorType newType = VectorType::get(1, i32Type);
    std::string funcName;
    if constexpr (isPrefetch) {
      funcName = "llvm_genx_raw_send2_noresult_i1_v8i32";
    } else {
      VectorType vecType;
      if constexpr (isLoad) {
        vecType = cast<VectorType>(op.getResult().getType());
        funcName = "llvm_genx_raw_send2_";
      } else {
        vecType = cast<VectorType>(op.getValue().getType());
        funcName = "llvm_genx_raw_sends2_noresult_i1_v8i32_";
      }
      std::string typeStr;
      std::tie(typeStr, newType) =
          encodeVectorType(rewriter, vecType);
      funcName += typeStr;
    }
    unsigned cacheHint = encodeCacheHint(op);

    /// fill in parameters for raw.send
    // bit[1:0] EOT,sendc
    auto modifier = createIntConstant(i8Type, 0);
    auto execSize = createIntConstant(i8Type, 0);
    auto pred = createIntConstant(i1Type, 1);
    auto numSrc1 = createIntConstant(i8Type, 1);
    unsigned numElem = newType.getNumElements();
    unsigned numDstVal = (numElem / 16) < 1 ? 1: numElem / 16;
    // if (rank == 1) {
    //   numDstVal *= 2;
    // }
    auto numDst = createIntConstant(i8Type, numDstVal);
    // 15 for ugm
    auto sfid = usingSLM ? createIntConstant(i8Type, 14) 
                      : createIntConstant(i8Type, 15); ;
    auto extMsg = createIntConstant(i32Type, 0);
    // message descriptor
    uint32_t rawSendMsg = 0;
    if (rank == 2) {
      // https://gfxspecs.intel.com/Predator/Home/Index/53680
      rawSendMsg |= (isLoad || isPrefetch) ? 3 : 7;
      rawSendMsg |= (vnni ? 1 : 0) << 7;
      rawSendMsg |= (encodeDataum(elmType) - 1) << 9;
      rawSendMsg |= (transpose ? 1 : 0) << 15;
      rawSendMsg |= cacheHint << 17;
      rawSendMsg |= (isLoad ? numDstVal : 0) << 20;
      rawSendMsg |= 1 << 25;
    } else {
      int vecSize;
      if (numElem <= 4) {
        vecSize = numElem - 1;
      } else {
        vecSize = log2(numElem) + 1;
      }
      // rank == 1
      // https://gfxspecs.intel.com/Predator/Home/Index/53523
      rawSendMsg |= (isLoad || isPrefetch) ? 0 : 4;
      rawSendMsg |= usingSLM ? (2 << 7) : (3 << 7);
      rawSendMsg |= 2 << 9;
      rawSendMsg |= vecSize << 12;
      rawSendMsg |= 1 << 15;
      rawSendMsg |= cacheHint << 17;
      rawSendMsg |= (isLoad ? numDstVal : 0) << 20;
      rawSendMsg |= 1 << 25;
    }
    auto msg = createIntConstant(i32Type, rawSendMsg);
    // payload
    auto v8i32 = VectorType::get(8, i32Type);
    auto v4i64 = VectorType::get(4, i64Type);
    
    auto idx0 = createIntConstant(i32Type, 0);
    auto idx2 = createIntConstant(i32Type, 2);
    auto idx3 = createIntConstant(i32Type, 3);
    auto idx4 = createIntConstant(i32Type, 4);
    auto idx5 = createIntConstant(i32Type, 5);
    auto idx6 = createIntConstant(i32Type, 6);
    auto idx7 = createIntConstant(i32Type, 7);
    Value payLoad;
    if(!usingSLM){
      payLoad = rewriter.create<spirv::UndefOp>(loc, v4i64);
      payLoad =
          rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad, base, idx0);
      payLoad = rewriter.create<spirv::BitcastOp>(loc, v8i32, payLoad);
    } else {
      payLoad = rewriter.create<spirv::UndefOp>(loc, v8i32);
      payLoad =
          rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad, base, idx0);
    }
    
    if (rank == 2) {
      auto blockWidth = tileType.getShape()[1];
      auto blockHeight = tileType.getShape()[0];

      // fixme: support memref for now
      auto memType = cast<MemRefType>(createDescOp.getSource().getType());
      unsigned bitWidth = memType.getElementType().getIntOrFloatBitWidth();
      auto surfaceWidth = memType.getShape()[1] * (bitWidth / 8) - 1;
      auto surfaceHeight = memType.getShape()[0] - 1;
      // fixme: pitch = width for now
      auto surfacePitch = surfaceWidth;
      auto surfaceW = createIntConstant(i32Type, surfaceWidth);
      auto surfaceH = createIntConstant(i32Type, surfaceHeight);
      auto surfaceP = createIntConstant(i32Type, surfacePitch);
      auto createOffset = [&](unsigned idx) -> Value {
        Value val;
        if (ShapedType::isDynamic(createDescOp.getStaticOffsets()[idx])) {
          val = createDescOp.getOffsets()[idx];
          val = rewriter.create<arith::TruncIOp>(loc, i32Type, val);
        } else {
          val =
              createIntConstant(i32Type, createDescOp.getStaticOffsets()[idx]);
        }
        return val;
      };
      auto offsetX = createOffset(1);
      auto offsetY = createOffset(0);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              surfaceW, idx2);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              surfaceH, idx3);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              surfaceP, idx4);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              offsetX, idx5);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              offsetY, idx6);
      unsigned blockVal = ((blockHeight - 1) << 8) | (blockWidth - 1);
      auto blockInfo = createIntConstant(i32Type, blockVal);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              blockInfo, idx7);
    }
    SmallVector<Value> args{modifier, execSize, pred, numSrc1, numDst,
                            sfid,     extMsg,   msg,  payLoad};
    if constexpr (isLoad) {
      funcName += "_i1_v8i32";
      auto old = rewriter.create<spirv::UndefOp>(loc, newType);
      args.push_back(old);
      auto retType = newType;
      auto funcType =
          rewriter.getFunctionType(ValueRange(args).getTypes(), retType);
      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
      auto funcOp =
          rewriter.create<spirv::FunctionCallOp>(loc, retType, funcName, args);
      if (rank == 2) {
        rewriter.replaceOp(op, funcOp->getResult(0));
      } else {
        auto cast = rewriter.create<spirv::BitcastOp>(loc, op.getType(),
                                                      funcOp->getResult(0));
        rewriter.replaceOp(op, cast->getResult(0));
      }
    } else {
      if constexpr (isPrefetch)
        args.erase(args.begin() + 4);
      else {
        if (rank == 2) {
          args.push_back(adaptor.getValue());
        } else if (rank == 1) {
          if(adaptor.getValue().getType() != newType){
            auto cast = rewriter.create<spirv::BitcastOp>(loc, newType,
                                                          adaptor.getValue());
            args.push_back(cast);
          } else {
            args.push_back(adaptor.getValue());
          }
          
        }
      }
      auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});
      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
      rewriter.create<spirv::FunctionCallOp>(loc, TypeRange(), funcName, args);
      rewriter.eraseOp(op);
    }
    return success();
  }
};

class DpasToVCPattern : public OpConversionPattern<DpasOp> {
public:
  using OpConversionPattern<DpasOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(DpasOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhsType = op.getLhs().getType().cast<VectorType>();
    auto rhsType = op.getRhs().getType().cast<VectorType>();
    auto resultType = op.getResultType().cast<VectorType>();
    uint8_t rc = lhsType.getShape()[0];
    uint8_t sd = lhsType.getShape()[1];
    // refer to IGC/visa/Common_ISA_util.cpp#87
    auto encodePrecision = [&](Type type) -> uint8_t {
      if (type == rewriter.getBF16Type())
        return 9;
      else if (type == rewriter.getF16Type())
        return 10;
      // else if (type == rewriter.getTF32Type())
      //   return 12;
      else {
        assert(0 && "add more support");
        return 0;
      }
    };
    uint8_t prec1 = encodePrecision(rhsType.getElementType());
    uint8_t prec2 = encodePrecision(lhsType.getElementType());
    unsigned infoVal = (rc << 24) | (sd << 16) | (prec2 << 8) | (prec1);
    auto infoAttr = rewriter.getIntegerAttr(rewriter.getI32Type(), infoVal);
    auto info = rewriter.create<spirv::ConstantOp>(loc, rewriter.getI32Type(),
                                                   infoAttr);
    auto newResultType = encodeVectorType(rewriter, resultType).second;
    SmallVector<Value, 4> args{adaptor.getRhs(), adaptor.getLhs(), info};
    std::string funcName = "llvm_genx_dpas_nosrc0_";
    if (op.getAcc()) {
      funcName = "llvm_genx_dpas2_";
      auto i32Type = rewriter.getI32Type();
      auto createIntConstant = [&](Type type, unsigned value) {
        auto attr = rewriter.getIntegerAttr(type, value);
        return rewriter.create<spirv::ConstantOp>(loc, type, attr);
      };
      auto prec1Arg = createIntConstant(i32Type, prec1);
      auto prec2Arg = createIntConstant(i32Type, prec2);
      auto sdArg = createIntConstant(i32Type, sd);
      auto rcArg = createIntConstant(i32Type, rc);
      auto signless = createIntConstant(i32Type, 0);
      args.assign({adaptor.getAcc(), adaptor.getRhs(), adaptor.getLhs(),
                   prec1Arg, prec2Arg, sdArg, rcArg, signless, signless});
    }
    funcName += encodeVectorType(rewriter, resultType).first;
    funcName += "_";
    funcName += encodeVectorType(rewriter, rhsType).first;
    funcName += "_";
    funcName += encodeVectorType(rewriter, lhsType).first;
    auto funcType =
        rewriter.getFunctionType(ValueRange(args).getTypes(), newResultType);
    Operation *opPtr = op;
    lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
    auto funcOp = rewriter.create<spirv::FunctionCallOp>(loc, newResultType,
                                                         funcName, args).getResults()[0];
    rewriter.replaceOp(op, funcOp);
    return success();
  }
};

template <typename OpType>
class GatherScatterToRawSend : public OpConversionPattern<OpType> {
public:
  using OpConversionPattern<OpType>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto createDescOp =
        op.getTensorDesc().template getDefiningOp<CreateDescOp>();
    auto memoryScope = createDescOp.getMemoryScope();
    bool usingSLM = memoryScope == xegpu::MemoryScope::SLM;
    auto offsets = rewriter.getRemappedValue(createDescOp.getOffsets());
    Type type = offsets.getType();

    auto offsetType = type.dyn_cast<VectorType>();
    auto SIMD = offsetType.getNumElements();

    // llvm::outs()<<"\n\nmemory_scope: "<<memoryScope<<"\n";
    // llvm::outs()<<"\n\nSIMD : "<<SIMD<<"\n";

    auto tileType = op.getTensorDesc().getType();
    auto rank = tileType.getRank();
    assert(rank <= 2 && "only support 1d/2d for now");
    auto loc = op->getLoc();
    constexpr bool isLoad = std::is_same_v<OpType, LoadGatherOp>;
    auto createIntConstant = [&](Type type, unsigned value) {
      auto attr = rewriter.getIntegerAttr(type, value);
      return rewriter.create<spirv::ConstantOp>(loc, type, attr);
    };

    /// collect common info
    auto i1Type = rewriter.getI1Type();
    auto i8Type = rewriter.getI8Type();
    auto i32Type = rewriter.getI32Type();
    auto i64Type = rewriter.getI64Type();
    auto base = adaptor.getTensorDesc();
    VectorType newType = VectorType::get(1, i32Type);
    std::string funcName;
    VectorType vecType;
    uint32_t elems;
    if constexpr (isLoad) {
      vecType = cast<VectorType>(op.getResult().getType());
      elems = vecType.getNumElements();
      funcName = "llvm_genx_raw_send2_";
    } else {
      vecType = cast<VectorType>(op.getValue().getType());
      elems = vecType.getNumElements();
      funcName = "llvm_genx_raw_sends2_noresult_i1_v8i32_";
    }
    // llvm::outs()<<"\n\nelems : "<<elems<<"\n";
    std::string typeStr;
    std::tie(typeStr, newType) = encodeVectorType(rewriter, vecType);
    funcName += typeStr;
    unsigned cacheHint = encodeCacheHint(op);

    /// fill in parameters for raw.send
    // bit[1:0] EOT,sendc
    auto modifier = createIntConstant(i8Type, 0);
    auto execSize =  SIMD == 16 ? createIntConstant(i8Type, 4) 
                : createIntConstant(i8Type, 5);
    auto pred = createIntConstant(i1Type, 1);
    auto numSrc1 = SIMD == 16 ? createIntConstant(i8Type, 2) 
                : createIntConstant(i8Type, 4);
    unsigned numDstVal = newType.getNumElements() / 16;
    auto numDst = createIntConstant(i8Type, numDstVal);
    // 15 for ugm
    auto sfid = usingSLM ? createIntConstant(i8Type, 14) 
                      : createIntConstant(i8Type, 15); //to do slm
    auto extMsg = createIntConstant(i32Type, 0);
    auto vecSize = 0;
    // llvm::outs()<<"\n\nnumDstVal: " <<numDstVal<<"\n";
    // llvm::outs()<<"\n\nexecSize: " <<execSize<<"\n";
    if (numDstVal <= 4) {
      vecSize = numDstVal - 1;
    } else {
      vecSize = log2(numDstVal) + 1;
    }
    vecSize = elems / SIMD - 1;
    // llvm::outs()<<"\n\nvecSize: " <<vecSize<<"\n";
    // message descriptor
    uint32_t rawSendMsg = 0;
    rawSendMsg |= (isLoad) ? 0 : 4;
    rawSendMsg |= usingSLM ? (2 << 7) : (3 << 7); // A64 for global, A32 for slm
    rawSendMsg |= 2 << 9; // D32
    rawSendMsg |= vecSize << 12;
    //rawSendMsg |= 1 << 15;
    rawSendMsg |= cacheHint << 17;
    rawSendMsg |= (isLoad ? numDstVal : 0) << 20;
    rawSendMsg |= usingSLM ? ((SIMD / 16) << 25) : ((SIMD / 8) << 25);
    auto msg = createIntConstant(i32Type, rawSendMsg);

    // payload
    auto payLoadType = usingSLM ? VectorType::get(SIMD, i32Type) 
              : VectorType::get(SIMD, i64Type);
    Value payLoad = rewriter.create<spirv::UndefOp>(loc, payLoadType);
    auto idx0 = createIntConstant(i32Type, 0);
    if(usingSLM){
      base = rewriter.create<UnrealizedConversionCastOp>(loc, i32Type, base).getResult(0);
    }
    payLoad =
        rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad, base, idx0);

    if(SIMD == 16){
      SmallVector<int32_t, 16> indices(16, 0);
      payLoad = rewriter.create<spirv::VectorShuffleOp>(
          loc, payLoadType, payLoad, payLoad, rewriter.getI32ArrayAttr(indices));
    } else if(SIMD == 32) {
      SmallVector<int32_t, 32> indices(32, 0);
      payLoad = rewriter.create<spirv::VectorShuffleOp>(
          loc, payLoadType, payLoad, payLoad, rewriter.getI32ArrayAttr(indices));
    }

    Value dataSize = i64_val(4);
    Value dataSizes = rewriter.create<vector::SplatOp>(loc, payLoadType, dataSize);
    if(!usingSLM){
      offsets = rewriter.create<spirv::UConvertOp>(loc, payLoadType, offsets);
    }
    offsets = rewriter.create<spirv::IMulOp>(loc, payLoadType, offsets, dataSizes);
    payLoad = rewriter.create<spirv::IAddOp>(loc, payLoadType, payLoad, offsets);
    // llvm::outs()<<"\n\nbase: "<<base<<"\n";
    // llvm::outs()<<"\n\npayload: "<<payLoad<<"\n";

    SmallVector<Value> args{modifier, execSize, pred, numSrc1, numDst,
                            sfid,     extMsg,   msg,  payLoad};
    if constexpr (isLoad) {
      funcName += usingSLM ? "_i1_v32i32" : "_i1_v32i64";
      auto old = rewriter.create<spirv::UndefOp>(loc, newType);
      args.push_back(old);
      auto retType = newType;
      auto funcType =
          rewriter.getFunctionType(ValueRange(args).getTypes(), retType);
      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
      Value data =
          rewriter.create<spirv::FunctionCallOp>(loc, retType, funcName, args).getResult(0);
      
      auto castTy = this->getTypeConverter()->convertType(op.getType());
      if(data.getType() != castTy){
        data = rewriter.create<spirv::BitcastOp>(loc, castTy, data);
      }
      rewriter.replaceOp(op, data);
    } else {
      Value data = adaptor.getValue();
      if (data.getType() != newType) {
        data = rewriter.create<spirv::BitcastOp>(loc, newType, data);
      }
      args.push_back(data);
      auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});
      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
      rewriter.create<spirv::FunctionCallOp>(loc, TypeRange(), funcName, args);
      rewriter.eraseOp(op);
    }
    // llvm::outs()<<"\n\nAfter GatherScatterToRawSend\n";
    // Operation *opPtr = op;
    // auto mod = opPtr->getParentOfType<mlir::ModuleOp>();
    // mod->print(llvm::outs());
    return success();
  }
};

class AllocNbarrierToVCPattern : public OpConversionPattern<AllocNbarrierOp> {
public:
  using OpConversionPattern<AllocNbarrierOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AllocNbarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value nbarrier_count = op.getNbarrierCount();
    auto countOp = nbarrier_count.getDefiningOp<arith::ConstantOp>();
    auto value = countOp.getValue().cast<IntegerAttr>();
    int32_t count = value.getValue().getZExtValue();
    llvm::outs()<<"\n\ncount: "<<count<<"\n";

    ModuleOp sprivModule = op->getParentOfType<ModuleOp>();

    // llvm::outs()<<"\n\nsprivModule: "<<sprivModule<<"\n";

    // sprivModule.walk([&](spirv::FuncOp funcOp){
    //   auto spirvModule = funcOp->getParentOfType<ModuleOp>();
    //   rewriter.setInsertionPointToEnd(spirvModule.getBody());

    //   rewriter.create<spirv::ExecutionModeOp>(funcOp.getLoc(), funcOp,
    //                                     spirv::ExecutionMode::NamedBarrierCountINTEL,
    //                                     count);
    // });

    rewriter.eraseOp(op);
    return success();
  }
};

class CreateNbarrierToVCPattern : public OpConversionPattern<CreateNbarrierOp> {
public:
  using OpConversionPattern<CreateNbarrierOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CreateNbarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto nbarrier_id = op.getNbarrierId();
    auto nbarrier_role = op.getNbarrierRole();
    auto num_producers = op.getNumProducers();
    auto num_consumers = op.getNumConsumers();

    auto i32Type = rewriter.getIntegerType(32);
    auto v8i32Type = mlir::VectorType::get(8, i32Type);

    DenseElementsAttr constantData = DenseElementsAttr::get(v8i32Type, ArrayRef<int>(std::vector<int>(1, 0)));
    Value nbarrier_src = rewriter.create<spirv::ConstantOp>(loc, v8i32Type, constantData);

    //payload format https://gfxspecs.intel.com/Predator/Home/Index/72064
    Value payload = zext(i32Type, nbarrier_id);

    Value payload_nbarrier_role = logic_shl(i32Type, zext(i32Type, nbarrier_role), i32_val(14));
    payload = bitwise_or(i32Type, payload, payload_nbarrier_role);

    Value payload_num_producers = logic_shl(i32Type, i32_val(num_producers), i32_val(16));
    payload = bitwise_or(i32Type, payload, payload_num_producers);

    Value payload_num_consumers = logic_shl(i32Type, i32_val(num_consumers), i32_val(24));
    payload = bitwise_or(i32Type, payload, payload_num_consumers);

    nbarrier_src = rewriter.create<spirv::VectorInsertDynamicOp>(loc, v8i32Type, nbarrier_src, payload, i32_val(2));
    rewriter.replaceOp(op, nbarrier_src);

    return success();
  }
};

class NbarrierArriveToVCPattern : public OpConversionPattern<NbarrierArriveOp> {
public:
  using OpConversionPattern<NbarrierArriveOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NbarrierArriveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto payload = op.getPayload();

    std::string funcName = "llvm_genx_raw_send2_noresult_i1_v8i32";

    // desc format 
    // https://github.com/intel-innersource/drivers.gpu.compute.vc-intrinsics/blob/cmc_experimental/GenXIntrinsics/include/llvm/GenXIntrinsics/Intrinsic_definitions.py#L4595
    Value modifier = i8_val(0);
    Value exec_size = i8_val(0);
    Value predicate = i1_val(1);
    Value numsrc1 = i8_val(1);           //register nums of payload
    Value sfid = i8_val(3);              //https://gfxspecs.intel.com/Predator/Home/Index/47532
    Value etDesc = i32_val(0);
    Value msg_desc = i32_val(0x2000004); //https://gfxspecs.intel.com/Predator/Home/Index/53524

    SmallVector<Value> args{modifier, exec_size, predicate, numsrc1,
                      sfid, etDesc, msg_desc, payload};

    auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});

    Operation *opPtr = op;
    lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
    rewriter.create<spirv::FunctionCallOp>(loc, TypeRange(),
                                                          funcName, args);

    rewriter.eraseOp(op);
    return success();
  }
};

class NbarrierWaitToVCPattern : public OpConversionPattern<NbarrierWaitOp> {
public:
  using OpConversionPattern<NbarrierWaitOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NbarrierWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto payload = op.getPayload();

    auto i8Type = rewriter.getIntegerType(8);
    auto i32Type = rewriter.getIntegerType(32);
    auto nbarrier_src = rewriter.create<spirv::VectorExtractDynamicOp>(loc, i32Type, payload, i32_val(2));
    auto nbarrier_id = zext(i8Type, bitwise_and(i32Type, nbarrier_src, i32_val(0xFF)));

    Value signal_flag = i8_val(0);   //0b0: wait 0b1: signal
    Value num_threads = i8_val(0);   //This field is ignored for nbarrier.wait

    std::string funcName = "llvm_genx_nbarrier";
    SmallVector<Value> args{signal_flag, nbarrier_id, num_threads};

    auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});

    Operation *opPtr = op;
    lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
    rewriter.create<spirv::FunctionCallOp>(loc, TypeRange(),
                                                          funcName, args);

    rewriter.eraseOp(op);
    return success();
  }
};

class CompilerHintToVCPattern : public OpConversionPattern<CompilerHintOp> {
public:
  using OpConversionPattern<CompilerHintOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CompilerHintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    std::string funcName = "llvm_genx_fence";
    Value fence_flag = i8_val(-128);
    SmallVector<Value> args{fence_flag};
    auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});

    Operation *opPtr = op;
    lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
    rewriter.create<spirv::FunctionCallOp>(loc, TypeRange(),
                                                          funcName, args);

    rewriter.eraseOp(op);
    return success();
  }
};

class MfenceToVCPattern : public OpConversionPattern<MfenceOp> {
public:
  using OpConversionPattern<MfenceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MfenceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto pred = i1_val(1);
    auto fence_op_attr = op.getFenceOpAttr().str();
    auto fence_scope_attr = op.getFenceScopeAttr().str();
    auto memory_kind_attr = op.getMemoryKindAttr().str();
    
    std::vector<std::string> lscFenceOp{"none", "evict", "invalidate", "discard", "clean", "flushl3"};
    std::vector<std::string> lscFenceScope{"group", "local", "tile", "gpu", "gpus", "system", "sysacq"};
    std::vector<std::string> lscSFID{"ugm", "ugml", "tgm", "slm"};

    uint8_t fence_op, fence_scope, sfid;

    auto it = std::find(lscFenceOp.begin(), lscFenceOp.end(), fence_op_attr);
    if(it != lscFenceOp.end()){
      fence_op = std::distance(lscFenceOp.begin(), it);
    }else{
      llvm_unreachable("unsupported value for lsc_fence_op attribute");
    }

    it = std::find(lscFenceScope.begin(), lscFenceScope.end(), fence_scope_attr);
    if(it != lscFenceScope.end()){
      fence_scope = std::distance(lscFenceScope.begin(), it);
    }else{
      llvm_unreachable("unsupported value for lsc_fence_scope attribute");
    }

    it = std::find(lscSFID.begin(), lscSFID.end(), memory_kind_attr);
    if(it != lscSFID.end()){
      sfid = std::distance(lscSFID.begin(), it);
    }else{
      llvm_unreachable("unsupported value for memory_kind attribute");
    }

    SmallVector<Value> args{pred, i8_val(sfid), i8_val(fence_op), i8_val(fence_scope)};
    auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});

    std::string funcName = "llvm.genx.lsc.fence.i1";

    Operation *opPtr = op;
    lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
    rewriter.create<spirv::FunctionCallOp>(loc, TypeRange(),
                                                          funcName, args);

    rewriter.eraseOp(op);
    return success();
  }
};

class CreateDescOpToVCPattern : public OpConversionPattern<CreateDescOp> {
public:
  using OpConversionPattern<CreateDescOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CreateDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    //llvm::outs() << "\nCreateDescOpToVCPattern: \n";
    auto loc = op.getLoc();
    auto src = adaptor.getSource();

    Value baseAddr;
    Type type = src.getType();
    if(isa<mlir::IntegerType>(type)){
      baseAddr = src;
    }else{
      baseAddr = rewriter.create<spirv::ConvertPtrToUOp>(loc, 
              rewriter.getI64Type(), adaptor.getSource()).getResult();
    }

    rewriter.replaceOp(op, baseAddr);
    return success();
  }
};

class ExpOpToVCPattern : public OpConversionPattern<spirv::CLExpOp> {
public:
  using OpConversionPattern<spirv::CLExpOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(spirv::CLExpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    llvm::outs() << "\nExpOpToVCPattern: \n";
    VectorType vecType = op.getResult().getType().cast<VectorType>();
    Type elemType = vecType.getElementType();
    auto src = adaptor.getOperand();
    llvm::outs()<<"\n\nExpOpToVCPattern vecType: "<<vecType<<"\n";

    Type f32Type = rewriter.getF32Type();
    Type f64Type = rewriter.getF32Type();
    Type i32Type = rewriter.getI32Type();

    Value log2eConst;
    if(elemType == f32Type){
      float log2e = 1.44269504089f;
      log2eConst = rewriter.create<spirv::ConstantOp>(loc, f32Type, rewriter.getF32FloatAttr(log2e));
    } else if(elemType == f64Type){
      //todo
    }

    Value log2eVec = rewriter.create<spirv::UndefOp>(loc, vecType);
    auto idx0 = rewriter.create<spirv::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(0));
    log2eVec =
        rewriter.create<spirv::VectorInsertDynamicOp>(loc, log2eVec, log2eConst, idx0);
    SmallVector<int32_t, 32> indices(32, 0);
    log2eVec = rewriter.create<spirv::VectorShuffleOp>(
          loc, vecType, log2eVec, log2eVec, rewriter.getI32ArrayAttr(indices));

    //log2(e) * x
    auto log2eX = rewriter.create<spirv::FMulOp>(loc, vecType, src, log2eVec);

    std::string funcName = "llvm.genx.exp.";
    funcName += encodeVectorType(rewriter, vecType, false).first;
    SmallVector<Value> args{log2eX};
    auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {vecType});

    Operation *opPtr = op;
    lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);

    auto expBase2 = rewriter.create<spirv::FunctionCallOp>(loc, ArrayRef<mlir::Type>{vecType},
                                                            funcName, args).getResults()[0];

    rewriter.replaceOp(op, expBase2);
    return success();
  }
};

class FMaxOpToVCPattern : public OpConversionPattern<spirv::CLFMaxOp> {
public:
  using OpConversionPattern<spirv::CLFMaxOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(spirv::CLFMaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs() << "\nFMaxOpToVCPattern: \n";
    auto loc = op.getLoc();
    auto type = op.getResult().getType();
    //  process v1
    if(!type.isa<VectorType>()){
      return success();
    }

    VectorType vecType = op.getResult().getType().cast<VectorType>();
    Type elemType = vecType.getElementType();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    std::string funcName = "llvm.genx.fmax.";
    funcName += encodeVectorType(rewriter, vecType, false).first;
    SmallVector<Value> args{lhs, rhs};
    auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {vecType});

    Operation *opPtr = op;
    lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);

    auto ret = rewriter.create<spirv::FunctionCallOp>(loc, ArrayRef<mlir::Type>{vecType},
                                                            funcName, args).getResults()[0];
    llvm::outs() << "\nFMaxOpToVCPattern ret: \n" << ret << "\n";
    rewriter.replaceOp(op, ret);

    return success();
  }
};

class LoadGatherOpToVCPattern : public OpConversionPattern<LoadGatherOp> {
public:
  using OpConversionPattern<LoadGatherOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LoadGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs() << "\n\nLoadGatherOpToVCPattern: \n";
    auto loc = op.getLoc();
    auto result = op.getODSResults(0)[0];

    auto src = adaptor.getTensorDesc();
    llvm::outs()<<"\n\nsrc: "<<src<<"\n";
    Type type = src.getType();

    auto cerate_desc_op = dyn_cast_or_null<CreateDescOp>(op.getTensorDesc().getDefiningOp());

    Value offsets = cerate_desc_op.getOffsets();
    auto vecType = result.getType().dyn_cast<mlir::VectorType>();
    auto shape = vecType.getShape();
    auto memoryScope = cerate_desc_op.getMemoryScope();
    llvm::outs()<<"\n\noffsets: "<<offsets<<"\n";

    Value vectorSize = i32_val(log2(shape[0]));
    if(memoryScope == xegpu::MemoryScope::SLM){
      std::string funcName = "llvm_genx_lsc_load_slm_";
      funcName += encodeVectorType(rewriter, vecType, false).first;
      funcName += "_i1_i64";
      llvm::outs() << "\n\nLoadGatherOpToVCPattern funcName: " << funcName << "\n";
      Value baseAddr = rewriter.create<UnrealizedConversionCastOp>(loc, rewriter.getI32Type(), adaptor.getTensorDesc()).getResult(0);
      Value offset = rewriter.create<spirv::VectorExtractDynamicOp>(loc, rewriter.getI32Type(), offsets, i32_val(0));
      Value addr = rewriter.create<spirv::IAddOp>(loc, rewriter.getI32Type(), baseAddr, offset);

      SmallVector<Value> args{i1_val(1), i8_val(0), i8_val(0), i8_val(0), i16_val(1), i32_val(0), i8_val(3), i8_val(7), i8_val(2), i8_val(0), baseAddr ,i32_val(0)};
      auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {vecType});

      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);

      auto retType = mlir::VectorType::get(shape[0], mlir::FloatType::getF32(rewriter.getContext()));
      auto loadOp = rewriter.create<spirv::FunctionCallOp>(loc, ArrayRef<mlir::Type>{retType},
                                                            funcName, args);
      auto vec = loadOp.getResults()[0];

      rewriter.replaceOp(op, vec);
    } else {
      std::string funcName = "llvm_genx_lsc_load_stateless_";
      funcName += encodeVectorType(rewriter,vecType,false).first;
      funcName += "_i1_i64";
      llvm::outs() << "\n\nLoadGatherOpToVCPattern funcName: " << funcName << "\n";

      Value baseAddr = rewriter.create<UnrealizedConversionCastOp>(loc, rewriter.getI64Type(), adaptor.getTensorDesc()).getResult(0);
      SmallVector<Value> args{i1_val(1), i8_val(0), i8_val(0), i8_val(0), i16_val(1), i32_val(0), i8_val(3), i8_val(7), i8_val(2), i8_val(0), baseAddr ,i32_val(0)};
      auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {vecType});

      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);

      auto retType = mlir::VectorType::get(shape[0], mlir::FloatType::getF32(rewriter.getContext()));
      auto loadOp = rewriter.create<spirv::FunctionCallOp>(loc, ArrayRef<mlir::Type>{retType},
                                                            funcName, args);
      auto vec = loadOp.getResults()[0];

      rewriter.replaceOp(op, vec);
    }

    return success();
  }
};

class StoreScatterOpToVCPattern : public OpConversionPattern<StoreScatterOp> {
public:
  using OpConversionPattern<StoreScatterOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(StoreScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs() << "\n\n StoreScatterOpToVCPattern;\n";
    auto loc = op.getLoc();
    auto value = adaptor.getValue();

    auto cerate_desc_op = dyn_cast_or_null<CreateDescOp>(op.getTensorDesc().getDefiningOp());
    //auto offsets = cerate_desc_op.getOffsets().Value;
    auto vecType = value.getType().dyn_cast<mlir::VectorType>();
    auto shape = vecType.getShape();
    auto memoryScope = cerate_desc_op.getMemoryScope();

    Value vectorSize = i32_val(log2(shape[0]));
    if(memoryScope == xegpu::MemoryScope::SLM){
      std::string funcName = "llvm_genx_lsc_store_slm_i1_i64_";
      funcName += encodeVectorType(rewriter, vecType, false).first;
      llvm::outs() << "\n\nStoreGatherOpToVCPattern funcName: " << funcName << "\n";

      Value baseAddr = rewriter.create<UnrealizedConversionCastOp>(loc, rewriter.getI32Type(), adaptor.getTensorDesc()).getResult(0);
      // Value offset = rewriter.create<spirv::VectorExtractDynamicOp>(loc, rewriter.getI32Type(), offsets, i32_val(0));
      // Value addr = rewriter.create<spirv::IAddOp>(loc, rewriter.getI32Type(), baseAddr, offset);
      SmallVector<Value> args{i1_val(1), i8_val(4), i8_val(0), i8_val(0), i16_val(1), i32_val(0), i8_val(3), i8_val(2), i8_val(2), i8_val(0), baseAddr, value, i32_val(0)};
      auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});

      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);

      rewriter.create<spirv::FunctionCallOp>(loc, ArrayRef<mlir::Type>{},
                                                            funcName, args);
      rewriter.eraseOp(op);
    } else {
      std::string funcName = "llvm_genx_lsc_store_stateless_i1_i64_";
      funcName += encodeVectorType(rewriter,vecType,false).first;
      llvm::outs() << "\n\nStoreGatherOpToVCPattern funcName: " << funcName << "\n";

      Value baseAddr = rewriter.create<UnrealizedConversionCastOp>(loc, rewriter.getI64Type(), adaptor.getTensorDesc()).getResult(0);
      SmallVector<Value> args{i1_val(1), i8_val(4), i8_val(0), i8_val(0), i16_val(1), i32_val(0), i8_val(3), i8_val(7), i8_val(2), i8_val(0), baseAddr, value, i32_val(0)};
      auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});

      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);

      rewriter.create<spirv::FunctionCallOp>(loc, ArrayRef<mlir::Type>{},
                                                            funcName, args);
      rewriter.eraseOp(op);
    }

    return success();
  }
};

class AllocaOpToVCPattern : public OpConversionPattern<memref::AllocaOp> {
public:
  using OpConversionPattern<memref::AllocaOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::AllocaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    
    MemRefType allocType = op.getType();
    Type spirvType = getTypeConverter()->convertType(allocType);

    Operation *parent =
      SymbolTable::getNearestSymbolTable(op->getParentOp());
    spirv::GlobalVariableOp varOp;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block &entryBlock = *parent->getRegion(0).begin();
      rewriter.setInsertionPointToStart(&entryBlock);
      auto varOps = entryBlock.getOps<spirv::GlobalVariableOp>();
      std::string varName =
        std::string("__workgroup_mem__") +
        std::to_string(std::distance(varOps.begin(), varOps.end()));
      varOp = rewriter.create<spirv::GlobalVariableOp>(loc, spirvType, varName,
                                                      /*initializer=*/nullptr);
    }
    Value ret = rewriter.create<spirv::AddressOfOp>(loc, varOp);
    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateXeGPUToVCIntrinsicsPatterns(
    SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns) {
  llvm::outs()<<"\n\npopulateXeGPUToVCIntrinsicsPatterns\n";
  patterns.add<DpasToVCPattern,
               AllocNbarrierToVCPattern, CreateNbarrierToVCPattern,
               NbarrierArriveToVCPattern, NbarrierWaitToVCPattern,
               CompilerHintToVCPattern, MfenceToVCPattern, 
               CreateDescOpToVCPattern, AllocaOpToVCPattern,
               CreateNdDescToVCPattern,
               //LoadGatherOpToVCPattern, StoreScatterOpToVCPattern, 
               GatherScatterToRawSend<LoadGatherOp>,
               GatherScatterToRawSend<StoreScatterOp>>(
      typeConverter, patterns.getContext());

  // math function
  patterns.add<ExpOpToVCPattern, FMaxOpToVCPattern>(
      typeConverter, patterns.getContext());

  if (getenv("IMEX_NOT_PREFER_RAWSEND")){
    patterns.add<LoadStorePrefetchNdToLsc<LoadNDOp>,
                 LoadStorePrefetchNdToLsc<StoreNDOp>,
                 LoadStorePrefetchNdToLsc<PrefetchNDOp>>(typeConverter,
                                                         patterns.getContext());
  }
  else {
    patterns.add<LoadStorePrefetchNdToRawSend<LoadNDOp>,
                 LoadStorePrefetchNdToRawSend<StoreNDOp>,
                 LoadStorePrefetchNdToRawSend<PrefetchNDOp>>(
        typeConverter, patterns.getContext());
  }
}