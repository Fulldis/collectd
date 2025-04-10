#include "utils/common/common.h"
#include "onnxruntime/onnxruntime_c_api.h"

#include "onnx_model.h"


#define ONNX_CHECK_ERROR(err)                                                            \
    {                                                                                    \
        if (ortApi == NULL) {                                                            \
          ERROR("ONNX_CHECK_ERROR was called with ortApi == NULL");                      \
          return 1;                                                                      \
        }                                                                                \
        if (err != NULL) {                                                               \
            ERROR("ONNX error occured: %s", ortApi->GetErrorMessage(err));               \
            ortApi->ReleaseStatus(err);                                                  \
            return 1;                                                                    \
        }                                                                                \
    }

typedef struct {
  OrtSession *session;

  size_t inputs_len;
  char **input_names;
  int64_t *input_shapes;
  OrtValue **input_tensors;

  size_t outputs_len;
  char **output_names;
  OrtValue **output_tensors;
} OrtModel;

struct ort_context_s {
  const OrtApi *api;
  OrtEnv *env;
  OrtAllocator *allocator;
  OrtModel *model;
};


OrtStatusPtr model_prepare_tensors(const OrtApi *ortApi, OrtModel *model, OrtAllocator *allocator) {
    OrtStatusPtr err = NULL;

    model->input_tensors = calloc(model->inputs_len, sizeof(*model->input_tensors));
    for (size_t i = 0; i < model->inputs_len; i++) {
      const int64_t shape[2] = {1, model->input_shapes[i]};
      err = ortApi->CreateTensorAsOrtValue(allocator, shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &model->input_tensors[i]);
      if (err != NULL) {
        return err;
      }
    }

    model->output_tensors = calloc(model->outputs_len, sizeof(*model->output_tensors));
    for (size_t i = 0; i < model->outputs_len; i++) {
      const int64_t shape[2] = {1, 1}; // outputs are always single-valued
      err = ortApi->CreateTensorAsOrtValue(allocator, shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &model->output_tensors[i]);
      if (err != NULL) {
        return err;
      }
    }

    return err;
}

OrtStatusPtr model_prepare_names(const OrtApi *ortApi, OrtModel *model, OrtAllocator *allocator) {
    OrtStatusPtr err = NULL;
    err = ortApi->SessionGetInputCount(model->session, &model->inputs_len);
    if (err != NULL) {
      return err;
    }

    model->input_names = malloc(sizeof(*model->input_names) * model->inputs_len);
    for (size_t i = 0; i < model->inputs_len; i++) {
      err = ortApi->SessionGetInputName(model->session, i, allocator, &model->input_names[i]);
      if (err != NULL) {
        return err;
      }
    }

    err = ortApi->SessionGetOutputCount(model->session, &model->outputs_len);
    if (err != NULL) {
      return err;
    }

    model->output_names = malloc(sizeof(*model->output_names) * model->outputs_len);
    for (size_t i = 0; i < model->outputs_len; i++) {
      err = ortApi->SessionGetOutputName(model->session, i, allocator, &model->output_names[i]);
      if (err != NULL) {
        return err;
      }
    }

    return err;
}

OrtStatusPtr model_create(const OrtApi *ortApi, OrtContext *ortContext, OrtModelConfig *cfg) {
    OrtStatusPtr err = NULL;
    OrtModel *model = calloc(1, sizeof(*model));

    OrtSessionOptions *sessionOpts;
    err = ortApi->CreateSessionOptions(&sessionOpts);
    if (err != NULL) return err;

    err = ortApi->CreateSession(ortContext->env, cfg->model_path, sessionOpts, &model->session);
    if (err != NULL) return err;

    ortApi->ReleaseSessionOptions(sessionOpts);
    if (err != NULL) return err;

    err = model_prepare_names(ortApi, model, ortContext->allocator);
    if (err != NULL) return err;

    if (model->inputs_len != cfg->inputs_len) {
      ERROR("model and config inputs do no match");
      abort();
    }
    model->input_shapes = calloc(cfg->inputs_len, sizeof(*model->input_shapes));
    for (size_t i = 0; i < cfg->inputs_len; i++) {
      model->input_shapes[i] = cfg->input_shapes[i];
    }

    err = model_prepare_tensors(ortApi, model, ortContext->allocator);
    if (err != NULL) return err;

    printf("Created model: %s\n", cfg->model_path);
    printf("Inputs:\n");
    for (size_t i = 0; i < model->inputs_len; i++) {
      printf("\t%ld:\t%s\n", i, model->input_names[i]);
    }
    printf("Outputs:\n");
    for (size_t i = 0; i < model->outputs_len; i++) {
      printf("\t%ld:\t%s\n", i, model->output_names[i]);
    }

    ortContext->model = model;
    return err;
}

int onnx_init(OrtModelConfig *cfg, OrtContext **ctx) {
    const OrtApi *ortApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    OrtContext *ortContext = calloc(1, sizeof(*ortContext));
    *ctx = ortContext;

    ortContext->api = ortApi;

    OrtStatusPtr err = ortApi->CreateEnv(ORT_LOGGING_LEVEL_INFO, "onnx_test", &ortContext->env);
    ONNX_CHECK_ERROR(err);

    err = ortApi->GetAllocatorWithDefaultOptions(&ortContext->allocator);
    ONNX_CHECK_ERROR(err);

    err = model_create(ortApi, ortContext, cfg);
    ONNX_CHECK_ERROR(err);

    return 0;
}

int onnx_destroy(OrtContext *ortContext) {
    const OrtApi *ortApi = ortContext->api;
    OrtModel *model = ortContext->model;

    for (size_t i = 0; i < model->inputs_len; i++) {
      ortApi->ReleaseValue(model->input_tensors[i]);
    }
    free(model->input_tensors);
    for (size_t i = 0; i < model->inputs_len; i++) {
      ortApi->ReleaseValue(model->output_tensors[i]);
    }
    free(model->output_tensors);
    for (size_t i = 0; i < model->inputs_len; i++) {
      OrtStatusPtr err = ortApi->AllocatorFree(ortContext->allocator, model->input_names[i]);
      ONNX_CHECK_ERROR(err);
    }
    for (size_t i = 0; i < model->outputs_len; i++) {
      OrtStatusPtr err = ortApi->AllocatorFree(ortContext->allocator, model->output_names[i]);
      ONNX_CHECK_ERROR(err);
    }
    free(model->input_names);
    free(model->output_names);

    ortApi->ReleaseEnv(ortContext->env);
    ortApi->ReleaseSession(model->session);
    ortApi->ReleaseAllocator(ortContext->allocator);

    free(model);
    free(ortContext);
    return 0;
}

int onnx_run(OrtContext *ortContext, float **inputs, float *outputs) {
    const OrtApi *ortApi = ortContext->api;
    OrtModel *model = ortContext->model;

    for (size_t i = 0; i < model->inputs_len; i++) {
      float *input_buffer = NULL;

      OrtStatusPtr err = ortApi->GetTensorMutableData(model->input_tensors[i], (void**)&input_buffer);
      ONNX_CHECK_ERROR(err);

      for (size_t j = 0; j < model->input_shapes[i]; j++) {
        input_buffer[i] = inputs[i][j];
        printf("%f ", inputs[i][j]);
      }
      printf("\n");
    }

    OrtStatusPtr err = ortApi->Run(
      model->session,
      NULL,
      (const char *const *) model->input_names,
      (const OrtValue *const *) model->input_tensors,
      model->inputs_len,
      (const char *const *) model->output_names,
      model->outputs_len,
      model->output_tensors
    );
    ONNX_CHECK_ERROR(err);

    for (size_t i = 0; i < model->outputs_len; i++) {
      float *output_buffer = NULL;
      OrtStatusPtr err = ortApi->GetTensorMutableData(model->output_tensors[i], (void**)&output_buffer);
      ONNX_CHECK_ERROR(err);
      outputs[i] = output_buffer[0];
      printf("%f ", output_buffer[0]);
    }
    printf("\n");
    return 0;
}