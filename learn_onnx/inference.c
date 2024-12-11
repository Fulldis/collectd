#include <stdint.h>
#include <stdio.h>
#include <onnxruntime/onnxruntime_c_api.h>

#define ORT_ABORT_ON_ERROR(expr)                             \
    {                                                        \
        OrtStatus* err = (expr);                             \
        if (err != NULL) {                                   \
            printf("An error occured\n");                    \
            printf("%s", ctx->ort->GetErrorMessage(err));    \
            ctx->ort->ReleaseStatus(err);                    \
            abort();                                         \
        }                                                    \
    }

struct OrtContext {
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
};

typedef struct OrtContext OrtContext;

void init(OrtContext *ctx) {
    ctx->api_base = OrtGetApiBase();
    ctx->ort = ctx->api_base->GetApi(ORT_API_VERSION);
    const OrtApi *ort = ctx->ort;

    ctx->env = NULL;
    ORT_ABORT_ON_ERROR(ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "test", &ctx->env));

    ctx->session_opts = NULL;
    ORT_ABORT_ON_ERROR(ort->CreateSessionOptions(&ctx->session_opts));

    ctx->mem_info = NULL;
    ORT_ABORT_ON_ERROR(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeCPU, &ctx->mem_info));
}

void start_session(OrtContext *ctx, char *model_path, char *input_name, char *output_name, int input_size) {
    const OrtApi *ort = ctx->ort;

    ctx->session = NULL;
    ORT_ABORT_ON_ERROR(ort->CreateSession(ctx->env, model_path, ctx->session_opts, &ctx->session));

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
    ORT_ABORT_ON_ERROR(ort->CreateTensorWithDataAsOrtValue(
        ctx->mem_info,
        input_buffer,
        ctx->model_input_len,
        ctx->input_shape,
        ctx->input_shape_len,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &ctx->input_tensor
    ));

    ctx->output_tensor = NULL;
    ORT_ABORT_ON_ERROR(ort->Run(
        ctx->session,
        NULL,
        ctx->input_names,
        (const OrtValue* const*)&ctx->input_tensor,
        1,
        ctx->output_names,
        1,
        &ctx->output_tensor
    ));

    ORT_ABORT_ON_ERROR(ort->GetTensorMutableData(ctx->output_tensor, (void**)output_buffer));

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


int main(int argc, char **argv) {
    if (argc != 5) {
        printf("Wrong amount of args");
        return 1;
    }

    char *model_path = argv[1];
    char *input_name = argv[2];
    char *output_name = argv[3];
    int input_size = atoi(argv[4]);

    OrtContext ctx;
    init(&ctx);
    start_session(&ctx, model_path, input_name, output_name, input_size);

    float *input_buffer = malloc(ctx.model_input_len);
    for (int i = 0; i < input_size; i++) {
        scanf("%f", &input_buffer[i]);
    }

    float *output;

    run_inference(&ctx, input_buffer, &output);
    printf("%f\n", *output);

    cleanup(&ctx);
    free(input_buffer);

    return 0;
}
