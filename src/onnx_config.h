#include "collectd.h"

#include "filter_chain.h"
#include "onnx_model.h"

typedef struct {
  char *output_family_name;
  char **input_names;
  size_t inputs_len;
  int64_t *input_shapes;
  char **output_names;
  size_t outputs_len;
  OrtModelConfig *model_config;
} PluginConfig;


int config_init(const oconfig_item_t *ci, PluginConfig *cfg);