#include "Utility.h"

void dbgInfo(const std::string message) {
    const char* env_var = std::getenv("ENABLE_TRITON_DEBUG");
    if(env_var != nullptr && std::string(env_var) == "1") {
        llvm::outs() << "\n" << message << "\n";
    }
}

void dbgInfo(const std::string message, mlir::Value value) {
    const char* env_var = std::getenv("ENABLE_TRITON_DEBUG");
    if(env_var != nullptr && std::string(env_var) == "1") {
        llvm::outs() << "\n" << message << ": " << value << "\n";
    }
}

void dbgInfo(const std::string message, mlir::Type type) {
    const char* env_var = std::getenv("ENABLE_TRITON_DEBUG");
    if(env_var != nullptr && std::string(env_var) == "1") {
        llvm::outs() << "\n" << message << ": " << type << "\n";
    }
}

void dbgInfo(const std::string message, int value) {
    const char* env_var = std::getenv("ENABLE_TRITON_DEBUG");
    if(env_var != nullptr && std::string(env_var) == "1") {
        llvm::outs() << "\n" << message << ": " << value << "\n";
    }
}

void dbgInfo(const std::string message, mlir::Operation op) {
    const char* env_var = std::getenv("ENABLE_TRITON_DEBUG");
    if(env_var != nullptr && std::string(env_var) == "1") {
        llvm::outs() << "\n" << message << ": " << op << "\n";
    }
}

void dbgInfo(const std::string message, mlir::Attribute attr) {
    const char* env_var = std::getenv("ENABLE_TRITON_DEBUG");
    if(env_var != nullptr && std::string(env_var) == "1") {
        llvm::outs() << "\n" << message << ": " << attr << "\n";
    }
}