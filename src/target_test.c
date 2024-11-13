#include "collectd.h"

#include "filter_chain.h"
#include "utils/common/common.h"
#include "utils_cache.h"

static int tt_destroy(void **user_data) /* {{{ */
{
  return 0;
} /* }}} int tt_destroy */

static int tt_create(const oconfig_item_t *ci, void **user_data) /* {{{ */
{
  return 0;
} /* }}} int tt_create */

static int tt_invoke(metric_family_t const *fam, notification_meta_t **meta, void **user_data) {
  if (fam->type != METRIC_TYPE_GAUGE) {
    return FC_TARGET_CONTINUE;
  }

  printf("INVOKE: name = %s", fam->name);

  for (size_t i = 0; i < fam->resource.num; ++i) {
    printf(" %s = %s", fam->resource.ptr[i].name, fam->resource.ptr[i].value);
  }

  for (size_t i = 0; i < fam->metric.num; ++i) {
    gauge_t history[20];
    strbuf_t buf = STRBUF_CREATE;
    metric_identity(&buf, &fam->metric.ptr[i]);
    printf("\n%s = %lf, history = ", buf.ptr, fam->metric.ptr[i].value.gauge);
    uc_get_history_by_name(buf.ptr, history, 20);
    for (size_t i = 0; i < 20; ++i) {
      printf(" %lf", history[i]);
    }
    STRBUF_DESTROY(buf);
  } 
/*
  metric_identity(strbuf_t *buf, metric_t const *m);


  int ret = uc_get_history_by_name(fam->name, history, 10);
  if (ret == 0) {

  for (size_t i = 0; i < 10; ++i) {
    printf(" %lf", history[i]);
  }

  }
*/
  printf("\n");

  return FC_TARGET_CONTINUE;
} /* }}} int tt_invoke */

void module_register(void) {
  target_proc_t tproc = {0};

  tproc.create = tt_create;
  tproc.destroy = tt_destroy;
  tproc.invoke = tt_invoke;
  fc_register_target("test", tproc);
} /* module_register */
