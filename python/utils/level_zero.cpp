#include <cstdlib>
#include <iostream>
#include <level_zero/ze_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

namespace py = pybind11;
using namespace py::literals;

namespace {

bool getBoolEnv(const std::string &env) {
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return (str == "on" || str == "true" || str == "1");
}

} // namespace

namespace {

inline void checkResult(ze_result_t res, const char *func) {
  if (res != ZE_RESULT_SUCCESS)
    throw std::runtime_error(std::string(func) +
                             " failed: " + std::to_string(res));
}

#define CHECK_ZE_RESULT(expr) checkResult((expr), #expr)

} // namespace

#define L0_SAFE_CALL(call)                                                     \
  {                                                                            \
    auto status = (call);                                                      \
    if (status != 0) {                                                         \
      fprintf(stderr, "%s:%d: L0 error %d\n", __FILE__, __LINE__,              \
              (int)status);                                                    \
      exit(1);                                                                 \
    }                                                                          \
  }

#define EXPECT_EQ(value1, value2)                                              \
  {                                                                            \
    auto result = (value2);                                                    \
    if ((value1) != (result)) {                                                \
      std::string err_log("L0 API error code: ");                              \
      std::stringstream ss;                                                    \
      ss << std::hex << result << std::endl;                                   \
      throw std::runtime_error(err_log + ss.str());                            \
    }                                                                          \
  }

#define EXPECT_TRUE(value1) EXPECT_EQ(true, value1)

inline auto findDriverAndDevice()
{
    L0_SAFE_CALL(zeInit(ZE_INIT_FLAG_GPU_ONLY));

    // Discover all the driver instances
    uint32_t driverCount = 0;
    L0_SAFE_CALL(zeDriverGet(&driverCount, nullptr));
    fprintf(stderr, "driverCount = %d\n", (int)driverCount);

    ze_driver_handle_t *allDrivers =
        (ze_driver_handle_t *)malloc(driverCount * sizeof(ze_driver_handle_t));
    L0_SAFE_CALL(zeDriverGet(&driverCount, allDrivers));

    ze_driver_handle_t hDriver = nullptr;
    ze_device_handle_t hDevice = nullptr;
    for (uint32_t i = 0; i < driverCount; ++i) {
        uint32_t deviceCount = 0;
        hDriver = allDrivers[i];
        L0_SAFE_CALL(zeDeviceGet(hDriver, &deviceCount, nullptr));
        fprintf(stderr, "driver = %d: deviceCount= %d\n", (int)i, (int)deviceCount);
        ze_device_handle_t *allDevices =
            (ze_device_handle_t *)malloc(deviceCount * sizeof(ze_device_handle_t));
        L0_SAFE_CALL(zeDeviceGet(hDriver, &deviceCount, allDevices));
        for (uint32_t d = 0; d < deviceCount; ++d) {
            ze_device_properties_t device_properties;
            L0_SAFE_CALL(zeDeviceGetProperties(allDevices[d], &device_properties));
            if (ZE_DEVICE_TYPE_GPU == device_properties.type) {
                hDevice = allDevices[d];
                break;
            }
        }
        free(allDevices);
        if (nullptr != hDevice) {
            break;
        }
    }
    free(allDrivers);
    assert(hDriver);
    assert(hDevice);

    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    ze_context_handle_t hContext = nullptr;
    L0_SAFE_CALL(zeContextCreate(hDriver, &contextDesc, &hContext));

    return std::make_tuple(hDriver, hDevice, hContext);
}

ze_module_handle_t create_module(ze_context_handle_t context,
                                 ze_device_handle_t device,
                                 uint32_t *binary_ptr, size_t binary_size) {

  const char *build_flags = "";
  const ze_module_format_t format = ZE_MODULE_FORMAT_IL_SPIRV;

  ze_module_desc_t module_description = {};
  module_description.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  ze_module_constants_t module_constants = {};
  module_constants.numConstants = 0;
  module_constants.pConstantIds = 0;
  module_constants.pConstantValues = 0;

  module_description.pNext = nullptr;
  module_description.format = format;
  module_description.inputSize =
      static_cast<uint32_t>(binary_size * sizeof(uint32_t));
  module_description.pInputModule = (uint8_t *)binary_ptr;
  module_description.pBuildFlags = build_flags;
  module_description.pConstants = &module_constants;

  ze_module_build_log_handle_t buildlog;
  ze_module_handle_t module;
  auto context_initial = context;
  auto device_initial = device;
  auto error_no =
      zeModuleCreate(context, device, &module_description, &module, &buildlog);

  if (error_no != ZE_RESULT_SUCCESS) {
    size_t szLog = 0;
    EXPECT_EQ(ZE_RESULT_SUCCESS,
              zeModuleBuildLogGetString(buildlog, &szLog, nullptr));

    char *strLog = (char *)malloc(szLog);
    EXPECT_EQ(ZE_RESULT_SUCCESS,
              zeModuleBuildLogGetString(buildlog, &szLog, strLog));

    std::cerr << "L0 build module failed. Log:\n" << strLog << std::endl;
    free(strLog);
    EXPECT_EQ(ZE_RESULT_SUCCESS, zeModuleBuildLogDestroy(buildlog));
  }

  EXPECT_EQ(ZE_RESULT_SUCCESS, error_no);
  std::cout << "create_module: " << std::endl;
  return module;
}

void printModuleKernelName(ze_module_handle_t hModule) {
  uint32_t Count = 0;
  auto ret = zeModuleGetKernelNames(hModule, &Count, nullptr);
  assert(ret == ZE_RESULT_SUCCESS);
  std::unique_ptr<const char *[]> PNames(new const char *[Count]);
  ret = zeModuleGetKernelNames(hModule, &Count, PNames.get());
  assert(ret == ZE_RESULT_SUCCESS);
  if (1 || getBoolEnv("MLIR_ENABLE_DUMP")) {
    for (uint32_t i = 0; i < Count; ++i) {
      std::cout << std::string(PNames[i]) << std::endl;
    }
  }
}

ze_kernel_handle_t create_function(ze_module_handle_t module,
                                   ze_kernel_flags_t flag,
                                   std::string func_name) {
  ze_kernel_handle_t kernel;
  ze_kernel_desc_t kernel_description = {};
  kernel_description.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;

  kernel_description.pNext = nullptr;
  kernel_description.flags = flag;
  kernel_description.pKernelName = func_name.c_str();
  auto module_initial = module;
  if (1 || getBoolEnv("MLIR_ENABLE_DUMP")) {
    std::cout << "create kernel:" << func_name << std::endl;
  }
  EXPECT_EQ(ZE_RESULT_SUCCESS,
            zeKernelCreate(module, &kernel_description, &kernel));
  //  EXPECT_EQ(module, module_initial);
  std::cout << "create_function: " << std::endl;
  return kernel;
}


ze_kernel_handle_t create_function(ze_module_handle_t module,
                                   std::string func_name) {
  return create_function(module, 0, func_name);
}

py::tuple spirv_to_l0_kernel(uint32_t *binary_ptr,
                               size_t binary_size, std::string kernel_name) {
  int32_t n_regs = 0;
  int32_t n_spills = 0;
  auto [l0_driver, l0_device, l0_context] = findDriverAndDevice();

  ze_module_handle_t l0_module =
    create_module(l0_context, l0_device, binary_ptr, binary_size);
  printModuleKernelName(l0_module);

  ze_kernel_handle_t l0_kernel = create_function(l0_module, kernel_name);

  ze_kernel_handle_t *k = &l0_kernel;
  py::capsule kernel_capsulle(&l0_kernel, [](void *f) {

  });

  //ze_module_handle_t *kb = &l0_module;
  py::capsule module_capsulle(&l0_module, [](void *f) {

  });

  py::tuple tup =
      py::make_tuple(module_capsulle, kernel_capsulle, n_regs, n_spills);

  std::cout << "spirv_to_l0_kernel" << std::endl;

  return tup;
}


PYBIND11_MODULE(l0_utils, m) {
  m.doc() = "triton level_zero utils to load the spirv kernel";
  m.def(
      "get_device_properties",
      [](void *device_ptr) {
        auto [pDriver, pDevice, pContext] = findDriverAndDevice();

        ze_device_compute_properties_t compute_properties{};
        L0_SAFE_CALL(zeDeviceGetComputeProperties(pDevice, &compute_properties));
        auto max_shared_mem = compute_properties.maxSharedLocalMemory;
        bool support_fp64 = false;
        auto eu_count_per_ss = 8;
        auto threads_per_eu = 8;
        auto max_clock_frequency = 1600.0;
        auto max_work_group_size = 1024;
        auto max_num_sub_groups = 64; // Max sub-groups per work group
        auto sub_group_sizes = compute_properties.subGroupSizes;

        py::dict properties =
            py::dict("max_shared_mem"_a = max_shared_mem,
                     "support_fp64"_a = support_fp64,
                     "eu_count_per_ss"_a = eu_count_per_ss,
                     "threads_per_eu"_a = threads_per_eu,
                     "max_clock_frequency"_a = max_clock_frequency,
                     "max_work_group_size"_a = max_work_group_size,
                     "max_num_sub_groups"_a = max_num_sub_groups,
                     "sub_group_sizes"_a = sub_group_sizes);
        return properties;
      },
      "Get the properties for a given device",
      py::return_value_policy::take_ownership);
  m.def(
      "load_binary",
      [](std::string name, py::bytes bytes, int shared, void *device_ptr) {
        std::string binary(bytes);

        std::cout << "binary size in u32:" << binary.size() / sizeof(uint32_t)
                << std::endl;

        auto retTuple = spirv_to_l0_kernel((uint32_t *)binary.c_str(),
                                binary.size() / sizeof(uint32_t), name);
        std::cout << "After spirv_to_l0_kernel" << std::endl;
        // py::tuple tup =
        //     py::make_tuple(0, 0, 0, 0);
        // return tup;
        return retTuple;
      },
      "Load provided spirv to LEVEL ZERO kernel",
      py::return_value_policy::take_ownership);
}