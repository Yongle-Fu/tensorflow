# Description:
#   TensorFlow Lite minimal example.

load("//tensorflow/lite:build_def.bzl", "tflite_linkopts")
load("@emsdk//emscripten_toolchain:wasm_rules.bzl", "wasm_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

BASE_LINKOPTS = [
    "-O3", # optimize for speed
    # "-Os", # optimize for size
    "-g4", # Debug options
    "-s ASSERTIONS=1", 
    # "-s ALLOW_BLOCKING_ON_MAIN_THREAD=1", 
    "-s ALLOW_MEMORY_GROWTH=1",
    # "-s TOTAL_MEMORY=32MB",
    # "-s MAXIMUM_MEMORY=4GB",
    # "-s ENVIRONMENT=web,worker",
    "-s DEFAULT_LIBRARY_FUNCS_TO_INCLUDE=[]",
    # "-s DISABLE_EXCEPTION_CATCHING=1",
    # "-s SYSCALLS_REQUIRE_FILESYSTEM=1",
    "-s FILESYSTEM=1",
    "-s EXIT_RUNTIME=0",
    "-s EXPORTED_FUNCTIONS='[\"_malloc\", \"_free\"]'",
    "-s EXPORTED_RUNTIME_METHODS='[\"cwrap\"]'",
    "-s EXPORT_NAME='MinimalWasm'",
    "-s MODULARIZE=1",
    # "-s INLINING_LIMIT=1",
    "-s MALLOC=emmalloc",
    "--pre-js $(location :pre.js)",
    "--post-js $(location :post.js)",
]

cc_binary(
    name = "minimal",
    srcs = [
        "c_api_test.cc",
    ],
    additional_linker_inputs = [
        "pre.js",
        "post.js",
    ],
    linkopts = tflite_linkopts() + BASE_LINKOPTS + ["-s USE_PTHREADS=0"],
    deps = [
        "//tensorflow/lite/c:c_api",
        # "//tensorflow/lite/c:c_api_test",
        # "//tensorflow/lite:framework",
        # "//tensorflow/lite/core:cc_api_stable",
        # "//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_binary(
    name = "minimal-threaded-simd",
    srcs = [
        "c_api_test.cc",
    ],
    additional_linker_inputs = [
        "pre.js",
        "post.js",
    ],
    copts = [
        "-msimd128",
        "-pthread",
    ],
    linkopts = tflite_linkopts() + BASE_LINKOPTS + [
        "-s USE_PTHREADS=1",
        # Pre-create 8 webworkers (threads). The actual number of threads that
        # will be used by XNNPACK for creating the threadpool might be fewer. It
        # is by default set to the number of logical cores which in most cases
        # should be fewer than 8. It can also be set manually by calling the
        # `setThreadsCount` API.
        "-s PTHREAD_POOL_SIZE=8",
    ],
    deps = [
        "//tensorflow/lite/c:c_api",
    ],
)

# bazel build -c opt -s --action_env=SHARED_MEMORY=0 //tensorflow/lite/examples/minimal:wasm-minimal
wasm_cc_binary(
    name = "wasm-minimal",
    cc_target = ":minimal",
    threads = "off",
)

# bazel build -c opt //tensorflow/lite/examples/minimal:wasm-minimal-threaded-simd
wasm_cc_binary(
    name = "wasm-minimal-threaded-simd",
    cc_target = ":minimal-threaded-simd",
    simd = True,
    threads = "emscripten",
)