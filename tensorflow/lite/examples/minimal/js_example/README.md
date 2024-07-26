# example

```shell
bazel build -c opt -s --action_env=SHARED_MEMORY=0 //tensorflow/lite/examples/minimal:wasm-minimal

cd js_example
copy_file.sh
node example.js
```
