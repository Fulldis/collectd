#ifndef ONNX_MODEL_H
#define ONNX_MODEL_H 1

typedef struct {
  char *model_path;
  int64_t *input_shapes;
  size_t inputs_len;
} OrtModelConfig;

struct ort_context_s;
typedef struct ort_context_s OrtContext;

int onnx_init(OrtModelConfig *config, OrtContext **ctx);
int onnx_destroy(OrtContext *ortContext);
int onnx_run(OrtContext *ortContext, float **inputs, float *outputs);

#endif