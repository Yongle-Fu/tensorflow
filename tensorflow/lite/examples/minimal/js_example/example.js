// const loadWasm = require('./minimal-threaded-simd');
// const loadWasm = require('./minimal-simd');
const loadWasm = require('./minimal');

const fs = require('fs');
const path = require('path');

(async function main() {
    console.log('------------- Example Main Begin -------------');
    const tflite = await loadWasm();
    // console.log('tflite', tflite);

    let invoke_hello_func = tflite.cwrap('em_say_hello', 'number', ['number']);
    let invoke_test_func  = tflite.cwrap('em_invoke_test', 'void', ['number', 'number']); // model from data and length
    // let invoke_test_func = tflite.cwrap('em_invoke_test', 'void', ['string']); // model from filepath
    console.log('invoke_hello_func', typeof(invoke_hello_func));
    console.log('invoke_test_func', typeof(invoke_test_func));

    // say hello communication with wasm
    if (typeof(invoke_hello_func) == 'function') console.log('invoke_hello_func', invoke_hello_func(42));

    const currentDirectory = process.cwd();
    console.log('Current working directory:', currentDirectory);

    // load model from binary data
    var filePath = path.join(currentDirectory, 'add.bin');
    fs.readFile(filePath, (err, data) => {
        if (err) {
            console.error('Error reading file:', err);
        } else {
            console.log('File content length', data.length, data);
            // run model inference test
            invoke_test_func(data, data.length);
        }
    });

    // load model from file path
    // filePath = "add.bin"
    // filePath = "tensorflow/lite/testdata/add.bin"
    // invoke_test_func(filePath);

    console.log('------------- Example Main End -------------');
})();