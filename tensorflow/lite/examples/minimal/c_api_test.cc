#include <emscripten.h>
#include <cstdio>
// #include <cstdlib> // exit

#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"

void exit_with_message(const char* message, const char* file, int line) {
  printf("Assertion failed: %s in %s at line %d\n", message, file, line);
  // exit(1);
}

#define ASSERT_NE(val1, val2) \
    if ((val1) == (val2)) { \
        exit_with_message(#val1 " != " #val2, __FILE__, __LINE__); \
    }

#define ASSERT_EQ(val1, val2) \
    if ((val1) != (val2)) { \
        exit_with_message(#val1 " == " #val2, __FILE__, __LINE__); \
    }

#define ASSERT_STREQ(str1, str2) \
    do { \
        const char* s1 = (str1); \
        const char* s2 = (str2); \
        while (*s1 && (*s1 == *s2)) { \
            s1++; \
            s2++; \
        } \
        if (*s1 != *s2) { \
            exit_with_message(#str1 " == " #str2, __FILE__, __LINE__); \
        } \
    } while (0)

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    printf("Error at %s:%d\n", __FILE__, __LINE__);          \
  }

extern "C" {
  int EMSCRIPTEN_KEEPALIVE em_say_hello(int num) {
    printf("em_say_hello\n");
    return num * 2;
  }

  // void EMSCRIPTEN_KEEPALIVE em_invoke_test(const char* filename) {
  void EMSCRIPTEN_KEEPALIVE em_invoke_test(const void* model_data, size_t model_size) {
    printf("=== Run inference Test ===\n");

    // TfLiteModel* model = TfLiteModelCreateFromFile(filename);
    TfLiteModel* model = TfLiteModelCreate(model_data, model_size);
    printf("model: %p", model);
    ASSERT_NE(model, NULL);

    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    printf("options: %p", options);
    ASSERT_NE(options, NULL);

    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
    ASSERT_NE(interpreter, NULL);

    // The options can be deleted immediately after interpreter creation.
    TfLiteInterpreterOptionsDelete(options);

    ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
    ASSERT_EQ(TfLiteInterpreterGetInputTensorCount(interpreter), 1);
    ASSERT_EQ(TfLiteInterpreterGetOutputTensorCount(interpreter), 1);

    int input_dims[1] = {2};
    ASSERT_EQ(TfLiteInterpreterResizeInputTensor(interpreter, 0, input_dims, 1), kTfLiteOk);
    ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    ASSERT_NE(input_tensor, NULL);
    ASSERT_EQ(TfLiteTensorType(input_tensor), kTfLiteFloat32);
    ASSERT_EQ(TfLiteTensorNumDims(input_tensor), 1);
    ASSERT_EQ(TfLiteTensorDim(input_tensor, 0), 2);
    ASSERT_EQ(TfLiteTensorByteSize(input_tensor), sizeof(float) * 2);
    ASSERT_NE(TfLiteTensorData(input_tensor), NULL);
    ASSERT_STREQ(TfLiteTensorName(input_tensor), "input");

    TfLiteQuantizationParams input_params = TfLiteTensorQuantizationParams(input_tensor);
    ASSERT_EQ(input_params.scale, 0.f);
    ASSERT_EQ(input_params.zero_point, 0);

    float input[2] = {1.f, 3.f};
    ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor, input, 2 * sizeof(float)), kTfLiteOk);

    ASSERT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

    const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    ASSERT_NE(output_tensor, NULL);
    ASSERT_EQ(TfLiteTensorType(output_tensor), kTfLiteFloat32);
    ASSERT_EQ(TfLiteTensorNumDims(output_tensor), 1);
    ASSERT_EQ(TfLiteTensorDim(output_tensor, 0), 2);
    ASSERT_EQ(TfLiteTensorByteSize(output_tensor), sizeof(float) * 2);
    ASSERT_NE(TfLiteTensorData(output_tensor), NULL);
    ASSERT_STREQ(TfLiteTensorName(output_tensor), "output");

    TfLiteQuantizationParams output_params = TfLiteTensorQuantizationParams(output_tensor);
    ASSERT_EQ(output_params.scale, 0.f);
    ASSERT_EQ(output_params.zero_point, 0);

    float output[2];
    ASSERT_EQ(TfLiteTensorCopyToBuffer(output_tensor, output, 2 * sizeof(float)), kTfLiteOk);
    ASSERT_EQ(output[0], 3.f);
    ASSERT_EQ(output[1], 9.f);

    TfLiteInterpreterDelete(interpreter);

    // The model should only be deleted after destroying the interpreter.
    TfLiteModelDelete(model);
  }  

  int main() {
    printf("wasm-minimal main");
    return 0;
  }
} // extern "C"