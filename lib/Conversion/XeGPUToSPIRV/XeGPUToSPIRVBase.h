#ifndef TRITON_CONVERSION_XEGPU_TO_SPIRV_H
#define TRITON_CONVERSION_XEGPU_TO_SPIRV_H

#include "TypeConverter.h"

class ConvertXeGPUToSPIRVPatternBase {
public:
  explicit ConvertXeGPUToSPIRVPatternBase(
      XeGPUToSPIRVTypeConverter &typeConverter)
      : converter(&typeConverter) {}

  XeGPUToSPIRVTypeConverter *getTypeConverter() const { return converter; }

protected:
  XeGPUToSPIRVTypeConverter *converter;
};

template <typename SourceOp>
class ConvertXeGPUToSPIRVPattern
    : public OpConversionPattern<SourceOp>,
      public ConvertXeGPUToSPIRVPatternBase {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ConvertXeGPUToSPIRVPattern(
      XeGPUToSPIRVTypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<SourceOp>(typeConverter, context),
        ConvertXeGPUToSPIRVPatternBase(typeConverter) {}

protected:
  XeGPUToSPIRVTypeConverter *getTypeConverter() const {
    XeGPUToSPIRVTypeConverter *ret =
        ((ConvertXeGPUToSPIRVPatternBase *)this)->getTypeConverter();
    return (XeGPUToSPIRVTypeConverter *)ret;
  }
};

#endif
