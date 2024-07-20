# Description:
#   TensorFlow Lite minimal example.

load("//tensorflow/lite:build_def.bzl", "tflite_linkopts")
load("@emsdk//emscripten_toolchain:wasm_rules.bzl", "wasm_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_binary(
    name = "minimal",
    srcs = [
        "minimal.cc",
    ],
    # copts = [
    #     "-msimd128",
    #     "-pthread",
    # ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        # "//tensorflow/lite/c:c_api",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
)

wasm_cc_binary(
    name = "wasm-minimal",
    cc_target = ":minimal",
    # simd = True,
    # threads = "emscripten", # uncommnet this build ok
)
# bazel build -c opt //tensorflow/lite/examples/minimal:wasm-minimal