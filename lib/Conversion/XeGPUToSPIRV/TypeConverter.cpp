#include "TypeConverter.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "triton/Conversion/MLIRTypes.h"
#include "../TritonGPUToSPIRV/Utility.h"
#include "../TritonGPUToSPIRV/TypeConverter.h"
#include "triton/Dialect/XeGPU/IR/XeGPUOps.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::xegpu;

XeGPUToSPIRVTypeConverter::XeGPUToSPIRVTypeConverter(
       spirv::TargetEnvAttr &targetAttr, SPIRVConversionOptions &option)
        : SPIRVTypeConverter(targetAttr, option) {
  addConversion([&](mlir::MemRefType type) -> llvm::Optional<Type> {
    return convertXeGPUMemRefType(type);
  });
  addConversion([&](VectorType type) -> llvm::Optional<Type> {
    return convertXeGPUVectorType(type);
  });
  addConversion([&](RankedTensorType type) -> llvm::Optional<Type> {
    return type;
  });

  addConversion([&](::mlir::triton::xegpu::TensorDescType type) -> llvm::Optional<Type> {
    return VectorType::get(8, IntegerType::get(type.getContext(), 32));
  });

  // Internally store bfloat16 as int16
  addConversion([&](BFloat16Type type) -> llvm::Optional<Type> {
    return IntegerType::get(type.getContext(), 16);
  });
  addConversion([&](Float16Type type) -> llvm::Optional<Type> {
    return Float16Type::get(type.getContext());
  });
  addConversion([&](IndexType type) -> llvm::Optional<Type> {
    return IntegerType::get(type.getContext(), 32); //getIndexType();
    //return IntegerType::get(type.getContext(), 64); //getIndexType();
  });

  addConversion([&](IntegerType type) -> llvm::Optional<Type> {
    return type; //getIndexType();
  });

  // Add generic source and target materializations to handle cases where
  // non-SPIRV types persist after an SPIRV conversion.
  addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> Optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
  });

  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> Optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
  });
}

Type XeGPUToSPIRVTypeConverter::convertXeGPUMemRefType(
        mlir::MemRefType type)  {
  Optional<spirv::StorageClass> storageClass = spirv::StorageClass::CrossWorkgroup;  //crossworkgroup

  assert(storageClass && "uncompatible pointer address type in SPIRV");

  auto ret = spirv::PointerType::get(convertType(type.getElementType()), *storageClass);
  return ret;
}

Type XeGPUToSPIRVTypeConverter::convertXeGPUVectorType(
        VectorType type)  {
  auto elemType = type.getElementType();
  auto bitWidth = elemType.getIntOrFloatBitWidth();
  int size = type.getNumElements() * bitWidth / 32;
  auto shape = type.getShape();

  if(shape.size() == 3 && shape[2] == 2){
    if (elemType == Float16Type::get(type.getContext()) || 
            elemType == BFloat16Type::get(type.getContext())) {
      elemType = IntegerType::get(type.getContext(), 32);
    }
    return VectorType::get(size, elemType);
  }else {
    return type;
  }
}