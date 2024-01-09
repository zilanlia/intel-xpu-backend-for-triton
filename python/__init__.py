import functools
import hashlib
import os
import re
import sysconfig
import tempfile
from pathlib import Path

import setuptools
import torch
import triton._C.libintel_xpu_backend_for_triton.triton as _triton  # noqa:E402
from triton._C.libtriton.triton import ir as triton_ir
from triton.common.backend import BaseBackend, register_backend  # noqa:E402
from triton.compiler.make_launcher import make_so_cache_key  # noqa:E402
from triton.runtime.cache import get_cache_manager  # noqa:E402
from triton.runtime.driver import DriverBase  # noqa:E402
from triton.runtime.jit import version_key  # noqa:E402

from .extensions import SYCLBuildExtension, SYCLExtension, use_profile  # noqa:E402


def is_ws_supported(module):
    # Override triton's `is_ws_supported` function, so that
    # the module id is kept the same within the overall context.
    if isinstance(module, _triton.module):
        return False
    else:
        return triton_ir.is_ws_supported


triton_ir.is_ws_supported = is_ws_supported


def _add_external_libs(mod, libs):
    for name, path in libs.items():
        if len(name) == 0 or len(path) == 0:
            return
    _triton.add_external_libs(mod, list(libs.keys()), list(libs.values()))


def ttir_to_ttgir(mod, num_warps):
    context = _triton.context()
    mod = _triton.parse_mlir_module(str(mod), context)
    mod.context = context
    pm = _triton.pass_manager(mod.context)
    pm.enable_debug()
    pm.add_convert_triton_to_tritongpu_pass(num_warps)
    pm.run(mod)
    return mod


def optimize_ttgir(mod, num_stages, arch):
    pm = _triton.pass_manager(mod.context)
    pm.enable_debug()
    pm.add_tritongpu_coalesce_pass()
    pm.add_tritongpu_remove_layout_conversions_pass()
    # pm.add_triton_intel_gpu_accelerate_matmul_pass(arch)
    pm.add_tritongpu_remove_layout_conversions_pass()
    pm.add_tritongpu_optimize_dot_operands_pass()
    pm.add_tritongpu_pipeline_pass(num_stages)
    pm.add_tritongpu_prefetch_pass()
    pm.add_tritongpu_optimize_dot_operands_pass()
    pm.add_tritongpu_remove_layout_conversions_pass()
    pm.add_tritongpu_decompose_conversions_pass()
    pm.add_tritongpu_reorder_instructions_pass()
    pm.add_cse_pass()
    pm.add_symbol_dce_pass()
    pm.run(mod)
    return mod


# SPIRV translation

def ttgir_to_spirv(mod, extern_libs, arch):
    if extern_libs:
        _add_external_libs(mod, extern_libs)
    spirv_code, share_memory_size = _triton.translate_triton_gpu_to_spirv(str(mod), arch)  # noqa: E501
    mod.share_memory_size = share_memory_size
    return spirv_code

def ttgir_to_llir(mod, extern_libs, arch, tma_infos):
    return _triton.translate_triton_gpu_to_llvmir(mod, 80, tma_infos)


def spirv_to_spvbin(spirv: str, compute_capability: int):
    # return _triton.compile_spirv_to_spvbin(spirv, compute_capability)
    return _triton.compile_spirv_to_spvbin(spirv, 80)


def spirv_get_kernel_name(spirv: str) -> str:
    '''
    Get kernel name from SPIRV code.
    This Kernel name is required when launching the kernel.
    '''
    assert spirv
    decl_ops = []
    for line in spirv.split('\n'):
        line = line.strip()
        if line.startswith('OpName'):
            decl_ops += [line.split()[-1]]
    def_ops = []
    for line in spirv.split('\n'):
        line = line.strip()
        if re.compile(r'\bOpEntryPoint\b').search(line):
            def_op = line.split()[2][1:]
            if '"{}"'.format(def_op) in decl_ops:
                def_ops += [def_op]
    assert len(def_ops) == 1, "expect only one kernel per spriv"
    return def_ops[0]


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def generate_launcher(constants, signature):
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())  # noqa: E501

    def _extracted_type_pybind11(ty):
        if ty[0] == '*':
            return "py::object"
        return {
            'i1': 'int32_t',
            'i32': 'int32_t',
            'i64': 'int64_t',
            'u32': 'uint32_t',
            'u64': 'uint64_t',
            'fp32': 'float',
            'f32': 'float',
            'fp64': 'double',
        }[ty]

    # Ipex available src
    return f"""
#include <pybind11/pybind11.h>
#include <sycl/sycl.hpp>
#include <cstdlib>
#ifdef TRITON_XPU_PROFILE
#include <ipex.h>
#include <ATen/record_function.h>
#endif

namespace py = pybind11;

namespace {{

bool getBoolEnv(const std::string &env) {{
        const char *s = std::getenv(env.c_str());
        std::string str(s ? s : "");
        std::transform(str.begin(), str.end(), str.begin(),
                        [](unsigned char c) {{ return std::tolower(c); }});
        return (str == "on" || str == "true" || str == "1");
}}

}}

static inline void* getPointer(const py::object& _obj, int idx) {{
  PyObject* obj = _obj.ptr();
  if (PyLong_Check(obj)) {{
    auto ptrValue = PyLong_AsVoidPtr(obj);
    if (PyErr_Occurred()) {{
      PyErr_Print();
    }}
    return (void*)ptrValue;
  }}
  if (obj == Py_None) {{
    return (void*)0;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if (ptr) {{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
    }}
    return (void*)PyLong_AsVoidPtr(ret);
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return (void*)0;
}}

static void set_scalar_arg(
        sycl::handler& cgh,
        int index,
        size_t size,
        const void* value) {{
    switch (size) {{
    case sizeof(uint8_t):
    cgh.set_arg(index, *static_cast<const uint8_t*>(value));
    break;
    case sizeof(uint16_t):
    cgh.set_arg(index, *static_cast<const uint16_t*>(value));
    break;
    case sizeof(uint32_t):
    cgh.set_arg(index, *static_cast<const uint32_t*>(value));
    break;
    case sizeof(uint64_t):
    cgh.set_arg(index, *static_cast<const uint64_t*>(value));
    break;
    default:
    assert(false && "wrong scalar size in sycl gen.");
    }}
}}

static void sycl_kernel_launch(int gridX, int gridY, int gridZ, int num_warps, int threads_per_warp, int shared_memory, sycl::queue& stream, sycl::kernel& kernel_ptr{', ' if signature.items() else ''} {arg_decls}) {{
  std::string kernel_name = kernel_ptr.get_info<sycl::info::kernel::function_name>();
#ifdef TRITON_XPU_PROFILE
RECORD_FUNCTION("XPU Triton kernel:" + kernel_name, {{}});
#endif
  void *params[] = {{ {', '.join(f"&arg{i}" for i in signature.keys() if i not in constants)} }};
  uint32_t num_params = sizeof(params)/sizeof(params[0]);
  uint32_t expected_num_params = kernel_ptr.get_info<sycl::info::kernel::num_args>();

  size_t global_range_x = gridX*threads_per_warp*num_warps;
  size_t global_range_y = gridY;
  size_t global_range_z = gridZ;

  size_t local_range_x = num_warps*threads_per_warp;
  size_t local_range_y = 1;
  size_t local_range_z = 1;

  sycl::range<3> global_range(global_range_z, global_range_y, global_range_x);
  sycl::range<3> local_range(local_range_z, local_range_y, local_range_x);
  sycl::nd_range<3> parallel_work_size(global_range, local_range);

  if (1 || getBoolEnv("MLIR_ENABLE_DUMP")){{
    std::cout << "kernel info name:" << kernel_name << " @" << &kernel_ptr << std::endl;
    std::cout << "kernel info attributes:" << kernel_ptr.get_info<sycl::info::kernel::attributes>() << std::endl;
    std::cout << "kernel info reference_count:" << kernel_ptr.get_info<sycl::info::kernel::reference_count>() << std::endl;
    std::cout << "kernel info num_args:" << kernel_ptr.get_info<sycl::info::kernel::num_args>() << std::endl;

    std::cout << "launch num param:" << num_params << std::endl;
    std::cout << "  gridx: " << gridX << std::endl;
    std::cout << "  gridY: " << gridY << std::endl;
    std::cout << "  gridZ: " << gridZ << std::endl;
    std::cout << "  num_warps: " << num_warps << std::endl;
    std::cout << "  threads_per_warp: " << threads_per_warp << std::endl;
    std::cout << "  global range:[" << "x:"<< global_range_x << ", y:" << global_range_y << ", z:" << global_range_z << "]" << std::endl;
    std::cout << "  local range:[" << "x:"<< local_range_x << ", y:" << local_range_y << ", z:" << local_range_z << "]" << std::endl;
    std::cout << "  shared_memory: " << shared_memory << std::endl;

    // param
    {" ".join(f'std::cout << "  param {idx}:" << *({ty_to_cpp(item)}*)params[{idx}] << std::endl;' for idx, item in enumerate([signature[i] for i in signature if i not in constants]))}
  }}

  if (shared_memory) {{
    expected_num_params -= 1;
  }}
  assert(num_params == expected_num_params && "number of kernel param not matched");

  // Submit the imported kernel.
  auto cgf = [&](sycl::handler &cgh) {{

    {" ".join(f'set_scalar_arg(cgh, {idx}, sizeof({ty_to_cpp(item)}), params[{idx}]);' for idx, item in enumerate([signature[i] for i in signature if i not in constants]))}

    if (shared_memory) {{
        using share_mem_t = sycl::accessor<int8_t, 1, sycl::access::mode::read_write, sycl::access::target::local>;
        share_mem_t local_buffer = share_mem_t(shared_memory, cgh);
        cgh.set_arg(num_params, local_buffer);
        //cgh.parallel_for(sycl::nd_range{{sycl::range{{(uint32_t)gridX*threads_per_warp*num_warps}}, sycl::range{{work_group_size}}}}, kernel_ptr);
        cgh.parallel_for(parallel_work_size, kernel_ptr);
    }} else {{
        cgh.parallel_for(parallel_work_size, kernel_ptr);
    }}

    }};

  auto event = stream.submit(cgf);
#ifdef TRITON_XPU_PROFILE
xpu::profiler_record(kernel_name, event);
#endif
}}

PYBIND11_MODULE(__triton_launcher, m) {{
  m.doc() = "triton bindings to the C++ launcher API";
    m.def("launch", [](int grid_x,
                       int grid_y,
                       int grid_z,
                       int num_warps,
                       int num_ctas,
                       int clusterDimX,
                       int clusterDimY,
                       int clusterDimZ,
                       int shared_memory,
                       void* _stream,
                       void* _kernel,
                       py::object &launch_enter_hook,
                       py::object &launch_exit_hook,
                       py::object &compiled_kernel{', ' if signature.items() else ''}
                       {', '.join([f"{_extracted_type_pybind11(ty)} _arg{i}" for i, ty in signature.items()])}){{
      int threads_per_warp = 32;
      if(py::hasattr(compiled_kernel, "threads_per_warp"))
        threads_per_warp = compiled_kernel.attr("threads_per_warp").cast<int>();
      sycl::queue* stream = static_cast<sycl::queue*>(_stream);
      sycl::kernel* kernel = static_cast<sycl::kernel*>(_kernel);
      sycl_kernel_launch(grid_x, grid_y, grid_z, num_warps, threads_per_warp, shared_memory, *stream, *kernel{', ' if signature.items() else ''}
             {', '.join(f"getPointer(_arg{i},{i})" if ty[0] == "*" else f"_arg{i}" for i, ty in signature.items())});
    }});
}}

"""  # noqa: E501

def generate_launcher_l0(constants, signature):
    print(signature)
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())  # noqa: E501
    arg_decls_list = []  # 创建一个空列表来存储生成的字符串
    for i, ty in signature.items():  # 遍历 signature 字典的每个元素
        arg_decl = f"{ty_to_cpp(ty)} arg{i}"  # 生成一个字符串
        arg_decls_list.append(arg_decl)  # 将这个字符串添加到列表中
    print(arg_decls_list)

    def _extracted_type_pybind11(ty):
        if ty[0] == '*':
            return "py::object"
        return {
            'i1': 'int32_t',
            'i32': 'int32_t',
            'i64': 'int64_t',
            'u32': 'uint32_t',
            'u64': 'uint64_t',
            'fp32': 'float',
            'f32': 'float',
            'fp64': 'double',
        }[ty]

    # Ipex available src
    return f"""
#include <pybind11/pybind11.h>
#include <level_zero/ze_api.h>
#include <cstdlib>
#include <iostream>
#ifdef TRITON_XPU_PROFILE
#include <ipex.h>
#include <ATen/record_function.h>
#endif

namespace py = pybind11;

namespace {{

bool getBoolEnv(const std::string &env) {{
        const char *s = std::getenv(env.c_str());
        std::string str(s ? s : "");
        std::transform(str.begin(), str.end(), str.begin(),
                        [](unsigned char c) {{ return std::tolower(c); }});
        return (str == "on" || str == "true" || str == "1");
}}

}}

static inline void* getPointer(const py::object& _obj, int idx) {{
  PyObject* obj = _obj.ptr();
  if (PyLong_Check(obj)) {{
    auto ptrValue = PyLong_AsVoidPtr(obj);
    if (PyErr_Occurred()) {{
      PyErr_Print();
    }}
    return (void*)ptrValue;
  }}
  if (obj == Py_None) {{
    return (void*)0;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if (ptr) {{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
    }}
    return (void*)PyLong_AsVoidPtr(ret);
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return (void*)0;
}}

inline auto findXPUDevice()
{{
    zeInit(ZE_INIT_FLAG_GPU_ONLY);

    // Discover all the driver instances
    uint32_t driverCount = 0;
    zeDriverGet(&driverCount, nullptr);

    ze_driver_handle_t *allDrivers = (ze_driver_handle_t *)malloc(driverCount * sizeof(ze_driver_handle_t));
    zeDriverGet(&driverCount, allDrivers);

    ze_driver_handle_t hDriver = nullptr;
    ze_device_handle_t hDevice = nullptr;
    for (uint32_t i = 0; i < driverCount; ++i) {{
        uint32_t deviceCount = 0;
        hDriver = allDrivers[i];
        zeDeviceGet(hDriver, &deviceCount, nullptr);

        ze_device_handle_t *allDevices = (ze_device_handle_t *)malloc(deviceCount * sizeof(ze_device_handle_t));
        zeDeviceGet(hDriver, &deviceCount, allDevices);
        for (uint32_t d = 0; d < deviceCount; ++d) {{
            ze_device_properties_t device_properties;
            zeDeviceGetProperties(allDevices[d], &device_properties);
            if (ZE_DEVICE_TYPE_GPU == device_properties.type) {{
                hDevice = allDevices[d];
                break;
            }}
        }}
        free(allDevices);
        if (nullptr != hDevice) {{
            break;
        }}
    }}
    free(allDrivers);
    assert(hDriver);
    assert(hDevice);

    ze_context_desc_t contextDesc = {{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0}};
    ze_context_handle_t hContext = nullptr;
    zeContextCreate(hDriver, &contextDesc, &hContext);

    return std::make_tuple(hDriver, hDevice, hContext);
}}

static void l0_kernel_launch(int gridX, int gridY, int gridZ, int num_warps, int threads_per_warp, int shared_memory, ze_kernel_handle_t kernel{', ' if signature.items() else ''} {arg_decls}) {{

  std::cout<<"l0_kernel_launch"<<std::endl;
  void *params[] = {{ {', '.join(f"&arg{i}" for i in signature.keys() if i not in constants)} }};
  uint32_t num_params = sizeof(params)/sizeof(params[0]);

  size_t global_range_x = gridX*threads_per_warp*num_warps;
  size_t global_range_y = gridY;
  size_t global_range_z = gridZ;

  size_t local_range_x = num_warps*threads_per_warp;
  size_t local_range_y = 1;
  size_t local_range_z = 1;
  
  auto castSz = [](size_t val) {{ return static_cast<uint32_t>(val); }};
  
  zeKernelSetGroupSize(kernel, castSz(local_range_x), 
                               castSz(local_range_y),
                               castSz(local_range_z));
  ze_group_count_t launchArgs = {{castSz(global_range_x), 
                                  castSz(global_range_y), 
                                  castSz(global_range_z)}};
  std::cout<<"set_scalar_arg"<<std::endl;
  //{" ".join(f'set_scalar_arg(kernel, {idx}, sizeof({ty_to_cpp(item)}), params[{idx}]);' for idx, item in enumerate([signature[i] for i in signature if i not in constants]))}
  
  {" ".join(f'std::cout << "  param {idx}:" << "{item} :" << *({ty_to_cpp(item)}*)params[{idx}] << std::endl;' for idx, item in enumerate([signature[i] for i in signature if i not in constants]))}

    std::cout << "launch num param:" << num_params << std::endl;
    std::cout << "  gridx: " << gridX << std::endl;
    std::cout << "  gridY: " << gridY << std::endl;
    std::cout << "  gridZ: " << gridZ << std::endl;
    std::cout << "  num_warps: " << num_warps << std::endl;
    std::cout << "  threads_per_warp: " << threads_per_warp << std::endl;
    std::cout << "  global range:[" << "x:"<< global_range_x << ", y:" << global_range_y << ", z:" << global_range_z << "]" << std::endl;
    std::cout << "  local range:[" << "x:"<< local_range_x << ", y:" << local_range_y << ", z:" << local_range_z << "]" << std::endl;
    std::cout << "  shared_memory: " << shared_memory << std::endl;

  {" ".join(f'zeKernelSetArgumentValue(kernel, {idx}, 4, &arg{idx});' for idx, item in enumerate([signature[i] for i in signature if i not in constants]))}

  if(0){{
    zeKernelSetArgumentValue(kernel, 0, 4, &arg0);
    zeKernelSetArgumentValue(kernel, 1, 4, &arg1);
    zeKernelSetArgumentValue(kernel, 2, 4, &arg2);
    zeKernelSetArgumentValue(kernel, 3, 4, &arg3);
  }}

  auto [l0_driver, l0_device, l0_context] = findXPUDevice();

  ze_command_list_handle_t command_list = nullptr;
  ze_command_list_desc_t commandListDesc = {{}};
  zeCommandListCreate(l0_context, l0_device, &commandListDesc, &command_list);
  ze_event_handle_t waitEvent = nullptr;
  uint32_t numWaitEvents = 0;
  ze_event_handle_t *phWaitEvents = nullptr;

  std::cout<<"zeCommandListAppendLaunchKernel"<<std::endl;
  zeCommandListAppendLaunchKernel(command_list, kernel,
                                    &launchArgs, waitEvent,
                                    numWaitEvents, phWaitEvents);

  ze_command_queue_desc_t command_queue_description = {{ ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0,
			ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL }};
  ze_command_queue_handle_t command_queue = nullptr;
  zeCommandQueueCreate(l0_context, l0_device, &command_queue_description, &command_queue);
  zeCommandQueueExecuteCommandLists(command_queue, 1, &command_list, nullptr);
  
  zeCommandQueueSynchronize(command_queue, UINT32_MAX);
}}

PYBIND11_MODULE(__triton_l0_launcher, m) {{
  m.doc() = "triton bindings to the C++ launcher API";
    m.def("launch", [](int grid_x,
                       int grid_y,
                       int grid_z,
                       int num_warps,
                       int num_ctas,
                       int clusterDimX,
                       int clusterDimY,
                       int clusterDimZ,
                       int shared_memory,
                       void* _stream,
                       void* _kernel,
                       py::object &launch_enter_hook,
                       py::object &launch_exit_hook,
                       py::object &compiled_kernel{', ' if signature.items() else ''}
                       {', '.join([f"{_extracted_type_pybind11(ty)} _arg{i}" for i, ty in signature.items()])}){{
      int threads_per_warp = 32;
      std::cout<<"launch"<<std::endl;
      //if(py::hasattr(compiled_kernel, "threads_per_warp"))
      //  threads_per_warp = compiled_kernel.attr("threads_per_warp").cast<int>();
      std::cout<<"ze_kernel_handle_t"<<std::endl;
      ze_kernel_handle_t* kernel_ptr = static_cast<ze_kernel_handle_t*>(_kernel);
      ze_kernel_handle_t kernel = *kernel_ptr;
      l0_kernel_launch(grid_x, grid_y, grid_z, num_warps, threads_per_warp, shared_memory, kernel{', ' if signature.items() else ''}
             {', '.join(f"getPointer(_arg{i},{i})" if ty[0] == "*" else f"_arg{i}" for i, ty in signature.items())});
    }});
}}

"""  # noqa: E501

def _build_xpu_ext(name, src, srcdir):

    TRITON_XPU_BUILD_LOGGING = os.getenv('TRITON_XPU_BUILD_LOGGING')
    if TRITON_XPU_BUILD_LOGGING is None or TRITON_XPU_BUILD_LOGGING == '0' or TRITON_XPU_BUILD_LOGGING.lower() == 'off':
        import logging
        logging.disable(logging.CRITICAL)

    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))  # noqa: E501

    # fallback on setuptools
    extra_compile_args = ['-w']
    # library_dirs = [cuda_lib_dir]
    # include_dirs = [srcdir, cu_include_dir]
    # library_dirs = []
    # include_dirs = [srcdir]
    libraries = ['ze_loader']
    # extra arguments
    # extra_link_args = []
    # create extension module
    # build extension module
    define_macros = [('TRITON_XPU_PROFILE', None)] if use_profile() else []

    # create extension module
    ext = SYCLExtension(name,
                        [src],
                        extra_compile_args=extra_compile_args,
                        libraries=libraries,
                        define_macros=define_macros)

    args = ['build_ext']
    args.append('--build-temp=' + srcdir)
    args.append('--build-lib=' + srcdir)
    args.append('-q')
    args = dict(
        name=name,
        ext_modules=[ext],
        cmdclass={
            'build_ext': SYCLBuildExtension},
        script_args=args,
    )
    # with quiet():
    setuptools.setup(**args)
    return so


#
# SYCL
#
class SYCLUtils(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(SYCLUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "utils", "sycl.cpp")).read_text()
        key = hashlib.md5(src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        fname = "sycl_utils.so"
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "main.cpp")
                with open(src_path, "w") as f:
                    f.write(src)
                so = _build_xpu_ext("sycl_utils", src_path, tmpdir)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location("sycl_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties


class SYCLDriver(DriverBase):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(SYCLDriver, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.utils = SYCLUtils()
        # self.backend = self.SYCL
        self.backend = "SYCL"

import inspect
#
# LEVEl_ZERO
#
class L0Utils(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(L0Utils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "utils", "level_zero.cpp")).read_text()
        key = hashlib.md5(src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        fname = "l0_utils.so"
        cache_path = cache.get_file(fname)
        print("cache_path:, ", cache_path)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "main.cpp")
                with open(src_path, "w") as f:
                    f.write(src)
                so = _build_xpu_ext("l0_utils", src_path, tmpdir)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location("l0_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.load_binary = mod.load_binary
        print("self.load_binary")
        print(self.load_binary)
        self.get_device_properties = mod.get_device_properties


class L0Driver(DriverBase):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(L0Driver, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.utils = L0Utils()
        # self.backend = self.L0
        self.backend = "L0"

class XPUBackend(BaseBackend):
    stub_so_path = ""

    def __init__(self, device_type: str) -> None:
        super(XPUBackend, self).__init__(device_type)
        # self.driver = SYCLDriver()
        self.driver = L0Driver()

    def add_stages(self, arch, extern_libs, stages, tma_infos):
        filter_in_stages = ["ast", "ttir"]
        filter_out_stages = []
        for key, _ in stages.items():
            if key not in filter_in_stages:
                filter_out_stages.append(key)
        for filter_out_key in filter_out_stages:
            stages.pop(filter_out_key)

        context = _triton.context()

        stages["ttgir"] = (lambda path: _triton.parse_mlir_module(Path(path).read_text(), context),
                           lambda src: optimize_ttgir(ttir_to_ttgir(src, arch["num_warps"]), arch["num_stages"], arch))
        # stages["spirv"] = (lambda path: Path(path).read_text(),
        #                    lambda src: ttgir_to_spirv(src, extern_libs, arch))
        # stages["spvbin"] = (lambda path: Path(path).read_bytes(),
        #                     lambda src: spirv_to_spvbin(src, arch))
        stages["llir_xpu"] = (lambda path: Path(path).read_text(),
                           lambda src: ttgir_to_llir(src, extern_libs, arch, tma_infos))

    def add_meta_info(self, ir, module, next_module, metadata, asm):
        if ir == "spirv":
            metadata["name"] = spirv_get_kernel_name(next_module)
            if "shared" not in metadata:
                metadata["shared"] = module.share_memory_size

        if ir == "spvbin":
            asm[ir] = next_module

    def get_driver(self):
        return self.driver

    def get_stream(self, idx=None):
        if idx is None:
            idx = self.get_current_device()
        return torch.xpu.current_stream(idx).sycl_queue

    @functools.lru_cache(None)
    def get_device_properties(self, device):
        return self.driver.utils.get_device_properties(torch.xpu.device(device).sycl_device)  # noqa: E501

    def get_current_device(self):
        return torch.xpu.current_device()

    def set_current_device(self, device):
        torch.xpu.set_device(device)

    def get_load_binary_fn(self):
        print("get_load_binary_fn")

        def _load_binary_fn(kernel_name, binary, shared_size, device):
            ret = self.driver.utils.load_binary(kernel_name, binary, shared_size, torch.xpu.device(device).sycl_device)  # noqa: E501
            print("_load_binary_fn")
            # print(ret)
            return ret

        return _load_binary_fn

    def get_kernel_bin(self):
        return "spvbin"

    def get_architecture_descriptor(self, **kwargs):
        arch = kwargs.get("arch", None)
        if arch is None:
            arch = self.get_device_properties(self.get_current_device())
        max_work_group_size = arch['max_work_group_size']
        max_num_sub_groups = arch['max_num_sub_groups']
        sub_group_sizes = arch['sub_group_sizes']
        # TODO: chose a reasonable subgroup size
        threads_per_warp = 32
        # assert threads_per_warp in sub_group_sizes, "Current platform does not support threads_per_warp to be 32"  # noqa: E501
        num_warps = max_work_group_size // threads_per_warp
        assert num_warps < max_num_sub_groups, \
            "invalid setting. max_work_group_size {}, max_num_subgroup {}, subgroup_sizes {}".format(  # noqa: E501
                max_work_group_size,
                max_num_sub_groups,
                max_num_sub_groups)
        capability = {"num_warps": num_warps, "threads_per_warp": threads_per_warp, "num_stages": 2}  # noqa: E501
        return capability

    def make_launcher_stub(self, name, signature, constants, ids):
        # name of files that are cached
        so_cache_key = make_so_cache_key(version_key(), signature, constants, ids)
        so_cache_manager = get_cache_manager(so_cache_key)
        so_name = f"{name}.so"
        # retrieve stub from cache if it exists
        cache_path = so_cache_manager.get_file(so_name)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                # src = generate_launcher(constants, signature)
                print("generate_launcher_l0")
                src = generate_launcher_l0(constants, signature)
                src_path = os.path.join(tmpdir, "main.cpp")
                with open(src_path, "w") as f:
                    f.write(src)
                so = _build_xpu_ext(name, src_path, tmpdir)
                with open(so, "rb") as f:
                    return so_cache_manager.put(f.read(), so_name, binary=True)
        else:
            return cache_path


register_backend("xpu", XPUBackend)
