//===- GPUToSPIRVPass.h - GPUToSPIRV conversion  ---------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the GPUToSPIRV conversion,
///
//===----------------------------------------------------------------------===//

#ifndef TRITON_XEGPUTOSPIRV_PASS_H_
#define TRITON_XEGPUTOSPIRV_PASS_H_

#include <memory>

namespace mlir {
class Pass;
struct ScfToSPIRVContextImpl;
class ModuleOp;
template <typename T> class OperationPass;

namespace triton {
/// Create a pass
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertXeGPUToSPIRVPass(bool mapMemorySpace = true);

} // namespace triton
} // namespace mlir

#endif
