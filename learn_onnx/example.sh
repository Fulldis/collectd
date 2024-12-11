#!/bin/bash

make || 1

# lot of schema error :(
echo "1" | ./inference square_predictor.onnx float_input variable 1
echo "2" | ./inference square_predictor.onnx float_input variable 1
echo "3" | ./inference square_predictor.onnx float_input variable 1
