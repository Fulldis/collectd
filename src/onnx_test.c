#include "collectd.h"
#include "onnxruntime/onnxruntime_c_api.h"

#include "filter_chain.h"
#include "utils/common/common.h"
#include "utils_cache.h"

#define ONNX_CHECK_ERROR(expr)                                                                 \
    {                                                                                          \
        OrtStatus* err = (expr);                                                               \
        if (err != NULL) {                                                                     \
            plugin_log(LOG_ERR, "ONNX error occured: %s", ctx->ort->GetErrorMessage(err));     \
            ctx->ort->ReleaseStatus(err);                                                      \
            abort();                                                                           \
        }                                                                                      \
    }

// пока храню всё тут без особой структуры
typedef struct {
    const OrtApiBase *api_base;
    const OrtApi *ort;
    OrtEnv *env;
    OrtSessionOptions *session_opts;
    OrtSession *session;
    OrtMemoryInfo *mem_info;
    OrtValue *input_tensor;
    OrtValue *output_tensor;

    int64_t input_shape[2];
    size_t input_shape_len;
    size_t model_input_len;
    const char *input_names[1];
    const char *output_names[1];
} OrtContext;

void init(OrtContext *ctx) {
    ctx->api_base = OrtGetApiBase();
    ctx->ort = ctx->api_base->GetApi(ORT_API_VERSION);
    const OrtApi *ort = ctx->ort;

    ctx->env = NULL;
    ONNX_CHECK_ERROR(ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "onnxruntime", &ctx->env));

    ctx->session_opts = NULL;
    ONNX_CHECK_ERROR(ort->CreateSessionOptions(&ctx->session_opts));

    ctx->mem_info = NULL;
    ONNX_CHECK_ERROR(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeCPU, &ctx->mem_info));
}

void cleanup(OrtContext *ctx) {
    const OrtApi *ort = ctx->ort;

    ort->ReleaseValue(ctx->input_tensor);
    ort->ReleaseValue(ctx->output_tensor);
    ort->ReleaseEnv(ctx->env);
    ort->ReleaseSession(ctx->session);
    ort->ReleaseSessionOptions(ctx->session_opts);
    ort->ReleaseMemoryInfo(ctx->mem_info);
}

void start_session(OrtContext *ctx, char *model_path, char *input_name, char *output_name, int input_size) {
    const OrtApi *ort = ctx->ort;

    ctx->session = NULL;
    ONNX_CHECK_ERROR(ort->CreateSession(ctx->env, model_path, ctx->session_opts, &ctx->session));

    ctx->input_shape[0] = 1;
    ctx->input_shape[1] = input_size;
    ctx->input_shape_len = 2;
    ctx->model_input_len = input_size * sizeof(float);

    ctx->input_names[0] = input_name;
    ctx->output_names[0] = output_name;
}

void run_inference(OrtContext *ctx, float *input_buffer, float **output_buffer) {
    const OrtApi *ort = ctx->ort;

    ctx->input_tensor = NULL;
    ONNX_CHECK_ERROR(ort->CreateTensorWithDataAsOrtValue(
        ctx->mem_info,
        input_buffer,
        ctx->model_input_len,
        ctx->input_shape,
        ctx->input_shape_len,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &ctx->input_tensor
    ));

    ctx->output_tensor = NULL;
    ONNX_CHECK_ERROR(ort->Run(
        ctx->session,
        NULL,
        ctx->input_names,
        (const OrtValue* const*)&ctx->input_tensor,
        1,
        ctx->output_names,
        1,
        &ctx->output_tensor
    ));

    ONNX_CHECK_ERROR(ort->GetTensorMutableData(ctx->output_tensor, (void**)output_buffer));

}


static int tt_destroy(void **user_data) /* {{{ */
{
  OrtContext *ctx = (OrtContext *) *user_data;
  cleanup(ctx);
  return 0;
} /* }}} int tt_destroy */

static int tt_create(const oconfig_item_t *ci, void **user_data) /* {{{ */
{
  char *model_path = "square_predictor.onnx";
  char *input_name = "float_input";
  char *output_name = "variable";
  int input_size = 1;

  OrtContext *ctx = calloc(1, sizeof(ctx));
  init(ctx);
  start_session(ctx, model_path, input_name, output_name, input_size);

  *user_data = ctx;

  plugin_log(LOG_DEBUG, "Successfully started session using %s\n", model_path);

  return 0;
} /* }}} int tt_create */

static int tt_invoke(metric_family_t const *fam, notification_meta_t **meta, void **user_data) {
  if (fam->type != METRIC_TYPE_GAUGE) {
    return FC_TARGET_CONTINUE;
  }

  OrtContext *ctx = *user_data;
  label_pair_t to_match = {.name = "system.memory.state", .value = "free"};

  for (size_t i = 0; i < fam->metric.num; ++i) {
    metric_t *metric = &fam->metric.ptr[i];
    label_set_t labels = metric->label;

    for (size_t j = 0; j < labels.num; j++) {
      label_pair_t label = labels.ptr[j];
      if (strcmp(label.name, to_match.name) == 0 && strcmp(label.value, to_match.value) == 0) {
        printf("found metric with required key (%s : %s)\n", label.name, label.value);
        printf("extracted value: %lf\n", metric->value.gauge);
        float *output = NULL;
        float input = (float) metric->value.gauge;
        run_inference(ctx, &input, &output);
        metric->value.gauge = *output;
        printf("replaced with answer: %f\n", *output);
      }
    }
  }

  return FC_TARGET_CONTINUE;
} /* }}} int tt_invoke */


void module_register(void) {
  target_proc_t tproc = {0};

  tproc.create = tt_create;
  tproc.destroy = tt_destroy;
  tproc.invoke = tt_invoke;

  fc_register_target("onnx_test", tproc);
} /* module_register */
