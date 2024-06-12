#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOpTraits.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <string>

using namespace mlir;

Value createConstantI32(Location loc, OpBuilder &rewriter, int32_t v);

Value createConstantI64(Location loc, OpBuilder &rewriter, int64_t v);

#define add(...) rewriter.create<spirv::IAddOp>(loc, __VA_ARGS__)
#define udiv(...) rewriter.create<spirv::UDivOp>(loc, __VA_ARGS__)
#define sub(...) rewriter.create<spirv::ISubOp>(loc, __VA_ARGS__)
#define mul(...) rewriter.create<spirv::IMulOp>(loc, __VA_ARGS__)
#define zext(...) rewriter.create<spirv::UConvertOp>(loc, __VA_ARGS__)
#define i1Type rewriter.getI1Type()
#define i8Type rewriter.getI8Type()
#define i16Type rewriter.getI16Type()
#define i32Type rewriter.getI32Type()
#define i64Type rewriter.getI64Type()
#define f16Type rewriter.getF16Type()
#define f32Type rewriter.getF32Type()
#define f64Type rewriter.getF64Type()
#define bf16Type rewriter.getBF16Type()
#define v8i32 VectorType::get(8, i32Type)
#define v4i64 VectorType::get(4, i64Type)
#define logic_shl(...) rewriter.create<spirv::ShiftLeftLogicalOp>(loc, __VA_ARGS__)
#define bitwise_or(...) rewriter.create<spirv::BitwiseOrOp>(loc, __VA_ARGS__)
#define bitwise_and(...) rewriter.create<spirv::BitwiseAndOp>(loc, __VA_ARGS__)
#define i32_val(value) rewriter.create<spirv::ConstantOp>(loc, rewriter.getIntegerType(32), rewriter.getI32IntegerAttr(value))
#define i64_val(value) rewriter.create<spirv::ConstantOp>(loc, rewriter.getIntegerType(64), rewriter.getI64IntegerAttr(value))
#define i16_val(value) rewriter.create<spirv::ConstantOp>(loc, rewriter.getIntegerType(16), rewriter.getI16IntegerAttr(value))
#define i8_val(value) rewriter.create<spirv::ConstantOp>(loc, rewriter.getIntegerType(8), rewriter.getI8IntegerAttr(value))
#define i1_val(value) rewriter.create<spirv::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getBoolAttr(value))

void dbgInfo(const std::string message);

void dbgInfo(const std::string message, mlir::Value value);

void dbgInfo(const std::string message, mlir::Type type);

void dbgInfo(const std::string message, int value);

void dbgInfo(const std::string message, mlir::Operation op);

void dbgInfo(const std::string message, mlir::Attribute attr);