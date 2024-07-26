#!/bin/sh
set -e
# cd ${0%/*}
SCRIPT_DIR=$(pwd)

NAME=wasm-minimal
# NAME=wasm-minimal-simd
# NAME=wasm-minimal-threaded-simd
OUTPUT_DIR=../../../../../bazel-bin/tensorflow/lite/examples/minimal/$NAME
echo $OUTPUT_DIR

# copy file to js_example directory
cp -rfv $OUTPUT_DIR/minimal.{js,wasm} .

# run node example
node example.js