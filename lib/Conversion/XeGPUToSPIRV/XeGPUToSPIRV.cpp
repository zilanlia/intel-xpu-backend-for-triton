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

/// @brief encodeVectorType(xxx, 8x8x2xf16, true) returns ["v64i32", 64xi32]
std::pair<std::string, VectorType>
encodeVectorType(ConversionPatternRewriter &rewriter, VectorType type,
                 bool cast = true) {
  auto elemType = type.getElementType();
  auto bitWidth = elemType.getIntOrFloatBitWidth();
  auto rank = type.getRank();
  auto shape = type.getShape();
  auto size = shape[0];
  for(int i=1;i<shape.size();i++){
    size *= shape[1];
  }

  if (!cast && bitWidth == 16) {
    assert(shape[rank - 1] == 2);
    size *= 2;
  }
  std::string str;
  switch (size) {
  case 2:
    str += "v2";
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
  if (elemType == rewriter.getF32Type())
    str += "f32";
  else if (elemType == rewriter.getI32Type())
    str += "i32";
  else if (elemType == rewriter.getF16Type()) {
    if (cast) {
      assert(shape[rank - 1] == 2);
      str += "i32";
      elemType = rewriter.getI32Type();
    } else {
      str += "f16";
    }
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

// class InitTileToVCPattern : public OpConversionPattern<InitTileOp> {
// public:
//   using OpConversionPattern<InitTileOp>::OpConversionPattern;
//   LogicalResult
//   matchAndRewrite(InitTileOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     rewriter.replaceOpWithNewOp<spirv::ConvertPtrToUOp>(
//         op, rewriter.getI64Type(), adaptor.getSource());

//     return success();
//   }
// };

// template <typename OpType>
// class LoadStore2dToVCPattern : public OpConversionPattern<OpType> {
// public:
//   using OpConversionPattern<OpType>::OpConversionPattern;
//   LogicalResult
//   matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     llvm::outs()<<"\n\nLoadStore2dToVCPattern\n";
//     auto loc = op.getLoc();
//     ::mlir::VectorType vecType;
//     std::string funcName;
//     constexpr bool isLoad = std::is_same_v<OpType, Load2DOp>;
//     if constexpr (isLoad) {
//       vecType = cast<VectorType>(op.getResult().getType());
//       funcName = "llvm_genx_lsc_load2d_stateless_";
//     } else {
//       vecType = cast<VectorType>(op.getValue().getType());
//       funcName = "llvm_genx_lsc_store2d_stateless_i1_i64_";
//     }
//     auto createIntConstant = [&](Type type, unsigned value) {
//       auto attr = rewriter.getIntegerAttr(type, value);
//       return rewriter.create<spirv::ConstantOp>(loc, type, attr);
//     };
//     auto i8Type = rewriter.getI8Type();
//     auto i32Type = rewriter.getI32Type();
//     // predicate(true for now)
//     auto pred = createIntConstant(rewriter.getI1Type(), 1);
//     // cached(0 for now)
//     auto cacheHint = createIntConstant(i8Type, 0);
//     unsigned cst = encodeDataum(vecType.getElementType());
//     auto dataum = createIntConstant(i8Type, cst);
//     auto transpose =
//         createIntConstant(i8Type, op->hasAttr("Transpose") ? 2 : 1);
//     // number of blocks(1 for now)
//     auto nBlks = createIntConstant(i8Type, 1);
//     // tile shape is in-memory shape?
//     auto tileType = op.getTile().getType();
//     auto blockWidth = tileType.getShape()[1];
//     auto blockHeight = tileType.getShape()[0];
//     auto blockW = createIntConstant(i32Type, blockWidth);
//     auto blockH = createIntConstant(i32Type, blockHeight);
//     auto transform =
//         createIntConstant(i8Type, op->hasAttr("VNNI_AXIS") ? 1 : 0);
//     auto base = adaptor.getTile();
//     auto initOp = op.getTile().template getDefiningOp<InitTileOp>();
//     auto memType = cast<MemRefType>(initOp.getSource().getType());
//     unsigned bitWidth = memType.getElementType().getIntOrFloatBitWidth();
//     auto surfaceWidth = memType.getShape()[1] * (bitWidth / 8) - 1;
//     auto surfaceHeight = memType.getShape()[0] - 1;
//     // FIXME: pitch = width for now
//     auto surfacePitch = surfaceWidth;
//     auto surfaceW = createIntConstant(i32Type, surfaceWidth);
//     auto surfaceH = createIntConstant(i32Type, surfaceHeight);
//     auto surfaceP = createIntConstant(i32Type, surfacePitch);
//     auto offsetX = createIntConstant(i32Type, initOp.getStaticOffsets()[1]);
//     auto offsetY = createIntConstant(i32Type, initOp.getStaticOffsets()[0]);
//     SmallVector<Value> args{pred,      cacheHint, cacheHint, dataum,
//                             transpose, nBlks,     blockW,    blockH,
//                             transform, base,      surfaceW,  surfaceH,
//                             surfaceP,  offsetX,   offsetY};
//     std::string typeStr;
//     VectorType newType;
//     std::tie(typeStr, newType) = encodeVectorType(rewriter, vecType);
//     if constexpr (!isLoad) {
//       args.push_back(adaptor.getValue());
//     }
//     funcName += typeStr;
//     if constexpr (isLoad) {
//       funcName += "_i1_i64";
//       auto retType = newType;
//       auto funcType =
//           rewriter.getFunctionType(ValueRange(args).getTypes(), retType);
//       Operation *opPtr = op;
//       lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
//       Value funcOp =
//           rewriter.create<spirv::FunctionCallOp>(
//             loc, 
//             retType, 
//             mlir::FlatSymbolRefAttr::get(rewriter.getContext(), funcName), 
//             args).getResults()[0];
//       rewriter.replaceOp(op, funcOp);
//     } else {
//       auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});
//       Operation *opPtr = op;
//       lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
//       auto funcOp = rewriter.create<spirv::FunctionCallOp>(loc, TypeRange(),
//                                                            funcName, args);
//       rewriter.eraseOp(op);
//     }
//     return success();
//   }
// };

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
    unsigned rank = lhsType.getRank();
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
    uint8_t prec1 = encodePrecision(lhsType.getElementType());
    uint8_t prec2 = encodePrecision(rhsType.getElementType());
    unsigned infoVal = (rc << 24) | (sd << 16) | (prec2 << 8) | (prec1);
    auto infoAttr = rewriter.getIntegerAttr(rewriter.getI32Type(), infoVal);
    auto info = rewriter.create<spirv::ConstantOp>(loc, rewriter.getI32Type(),
                                                   infoAttr);
    auto newResultType = encodeVectorType(rewriter, resultType).second;
    SmallVector<Value, 4> args{adaptor.getRhs(), adaptor.getLhs(), info};
    std::string funcName = "llvm_genx_dpas_nosrc0_";
    funcName += encodeVectorType(rewriter, resultType).first;
    funcName += "_";
    funcName += encodeVectorType(rewriter, lhsType).first;
    funcName += "_";
    funcName += encodeVectorType(rewriter, rhsType).first;
    auto funcType =
        rewriter.getFunctionType(ValueRange(args).getTypes(), newResultType);
    Operation *opPtr = op;
    lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
    Value funcOp = rewriter.create<spirv::FunctionCallOp>(
      loc, 
      ArrayRef<mlir::Type>{newResultType},
      mlir::FlatSymbolRefAttr::get(rewriter.getContext(), funcName), 
      ArrayRef<Value>(args)).getResults()[0];

    rewriter.replaceOp(op, funcOp);
    return success();
  }
};

Value createConstantI32(Location loc, PatternRewriter &rewriter, int32_t v) {
  auto i32ty = rewriter.getIntegerType(32);
  return rewriter.create<spirv::ConstantOp>(loc, i32ty,
                                           IntegerAttr::get(i32ty, v));
}

#define zext(...) rewriter.create<spirv::UConvertOp>(loc, __VA_ARGS__)
#define logic_shl(...) rewriter.create<spirv::ShiftLeftLogicalOp>(loc, __VA_ARGS__)
#define bitwise_or(...) rewriter.create<spirv::BitwiseOrOp>(loc, __VA_ARGS__)
#define bitwise_and(...) rewriter.create<spirv::BitwiseAndOp>(loc, __VA_ARGS__)
#define i32_val(...) createConstantI32(loc, rewriter, __VA_ARGS__)
#define i16_val(value) rewriter.create<spirv::ConstantOp>(loc, rewriter.getIntegerType(16), rewriter.getI16IntegerAttr(value))
#define i8_val(value) rewriter.create<spirv::ConstantOp>(loc, rewriter.getIntegerType(8), rewriter.getI8IntegerAttr(value))
#define i1_val(value) rewriter.create<spirv::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getBoolAttr(value))

class AllocNbarrierToVCPattern : public OpConversionPattern<AllocNbarrierOp> {
public:
  using OpConversionPattern<AllocNbarrierOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AllocNbarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto nbarrier_count = op.getNbarrierCount();
    auto countOp = nbarrier_count.getDefiningOp<arith::ConstantOp>();
    int32_t count = countOp.getValue().cast<IntegerAttr>().getValue().getZExtValue();

    spirv::ModuleOp sprivModule = op->getParentOfType<spirv::ModuleOp>();

    sprivModule.walk([&](spirv::FuncOp funcOp){
      auto spirvModule = funcOp->getParentOfType<spirv::ModuleOp>();
      rewriter.setInsertionPointToEnd(spirvModule.getBody());

      rewriter.create<spirv::ExecutionModeOp>(funcOp.getLoc(), funcOp,
                                        spirv::ExecutionMode::NamedBarrierCountINTEL,
                                        count);
    });

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
    llvm::outs() << "\n\nCreateDescOpToVCPattern: ";
    auto loc = op.getLoc();
    auto src = adaptor.getSource();
    llvm::outs()<<"\n\nCreateDescOpToVCPattern src.getDefiningOp(): "<<*src.getDefiningOp()<<"\n";
    // if(auto castOp = dyn_cast_or_null<UnrealizedConversionCastOp>(src.getDefiningOp())){
    //   auto i64Type = rewriter.getI64Type();
    //   Value smemBase = rewriter.create<spirv::ConstantOp>(loc, i64Type,
    //                                        IntegerAttr::get(i64Type, 0));
    //   rewriter.replaceOp(op, smemBase);
    //   return success();
    // }

    Value baseAddr = rewriter.create<spirv::ConvertPtrToUOp>(loc, 
              rewriter.getI64Type(), adaptor.getSource()).getResult();
    
    auto offset = adaptor.getOffsets();
    llvm::outs()<<"\n\nop: "<<op<<"\n";

    //todo genereate the desc
    llvm::outs()<<"\n\nadaptor.getBaseAddr(): "<<adaptor.getSource()<<"\n";
    llvm::outs()<<"\n\nbaseAddr: "<<baseAddr<<"\n";
    rewriter.replaceOp(op, baseAddr);
    llvm::outs()<<"\n\nreturn success()\n";
    // Operation *opPtr = op;
    // auto mod = op->getParentOfType<ModuleOp>();
    // mod->print(llvm::outs());
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

    auto cerate_desc_op = dyn_cast_or_null<CreateDescOp>(op.getTensorDesc().getDefiningOp());
    Value baseAddr = rewriter.create<UnrealizedConversionCastOp>(loc, rewriter.getI64Type(), adaptor.getTensorDesc()).getResult(0);
    auto offset = cerate_desc_op.getOffsets();
    auto vecType = result.getType().dyn_cast<mlir::VectorType>();
    auto shape = vecType.getShape();
    auto memoryScope = cerate_desc_op.getMemoryScope();
    llvm::outs()<<"\n\nvecType: "<<vecType<<"\n";
    llvm::outs()<<"\n\nbaseAddr: "<<baseAddr<<"\n";
    llvm::outs()<<"\n\nmemoryScope: "<<memoryScope<<"\n";
    llvm::outs()<<"\n\nshape[0]: "<<shape[0]<<"\n";

    Value vectorSize = i32_val(log2(shape[0]));
    if(memoryScope == xegpu::MemoryScope::SLM){
      std::string funcName = "llvm_genx_lsc_load_slm_";
      funcName += encodeVectorType(rewriter, vecType, false).first;
      funcName += "_i1_i64";
      llvm::outs() << "\n\nLoadGatherOpToVCPattern funcName: " << funcName << "\n";

      //%true, %uchar_0, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_3, %uchar_7, %uchar_2, %uchar_0, %arg_2, %uint_0
      
      SmallVector<Value> args{i1_val(1), i8_val(0), i8_val(0), i8_val(0), i16_val(1), i32_val(0), i8_val(3), vectorSize, i8_val(2), i8_val(0), baseAddr ,i8_val(0)};
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

      //%true, %uchar_0, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_3, %uchar_7, %uchar_2, %uchar_0, %arg_2, %uint_0
      SmallVector<Value> args{i1_val(1), i8_val(0), i8_val(0), i8_val(0), i16_val(1), i32_val(0), i8_val(3), vectorSize, i8_val(2), i8_val(0), baseAddr ,i8_val(0)};
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
    Value baseAddr = rewriter.create<UnrealizedConversionCastOp>(loc, rewriter.getI64Type(), adaptor.getTensorDesc()).getResult(0);
    auto offset = cerate_desc_op.getOffsets();
    auto vecType = value.getType().dyn_cast<mlir::VectorType>();
    auto shape = vecType.getShape();
    auto memoryScope = cerate_desc_op.getMemoryScope();

    llvm::outs()<<"\n\nvecType: "<<vecType<<"\n";
    llvm::outs()<<"\n\nbaseAddr: "<<baseAddr<<"\n";
    llvm::outs()<<"\n\nmemoryScope: "<<memoryScope<<"\n";
    llvm::outs()<<"\n\nshape[0]: "<<shape[0]<<"\n";

    Value vectorSize = i32_val(log2(shape[0]));
    if(memoryScope == xegpu::MemoryScope::SLM){
      std::string funcName = "llvm_genx_lsc_store_slm_";
      funcName += encodeVectorType(rewriter, vecType, false).first;
      funcName += "_i1_i64";
      llvm::outs() << "\n\nLoadGatherOpToVCPattern funcName: " << funcName << "\n";

      //%true, %uchar_4, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_3, %uchar_8, %uchar_2, %uchar_0, %arg_0, %dpas_result, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, vector<64 x f32>, i32
      SmallVector<Value> args{i1_val(1), i8_val(0), i8_val(0), i8_val(0), i16_val(1), i32_val(0), i8_val(3), vectorSize, i8_val(2), i8_val(0), baseAddr, value, i8_val(0)};
      auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});

      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);

      rewriter.create<spirv::FunctionCallOp>(loc, ArrayRef<mlir::Type>{},
                                                            funcName, args);
      rewriter.eraseOp(op);
    } else {
      std::string funcName = "llvm_genx_lsc_store_stateless_";
      funcName += encodeVectorType(rewriter,vecType,false).first;
      funcName += "_i1_i64";
      llvm::outs() << "\n\nLoadGatherOpToVCPattern funcName: " << funcName << "\n";

      //%true, %uchar_4, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_3, %uchar_8, %uchar_2, %uchar_0, %arg_0, %dpas_result, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, vector<64 x f32>, i32
      SmallVector<Value> args{i1_val(1), i8_val(4), i8_val(0), i8_val(0), i16_val(1), i32_val(0), i8_val(3), vectorSize, i8_val(2), i8_val(0), baseAddr, value, i8_val(0)};
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

void populateXeGPUToVCIntrinsicsPatterns(
    SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns) {
  llvm::outs()<<"\n\npopulateXeGPUToVCIntrinsicsPatterns\n";
  patterns.add<DpasToVCPattern,
               AllocNbarrierToVCPattern, CreateNbarrierToVCPattern,
               NbarrierArriveToVCPattern, NbarrierWaitToVCPattern,
               CompilerHintToVCPattern, MfenceToVCPattern, 
               CreateDescOpToVCPattern, LoadGatherOpToVCPattern,
               StoreScatterOpToVCPattern>(
      typeConverter, patterns.getContext());
}