/**
 * collectd - src/cpu_test.c
 * Copyright (C) 2024      Florian octo Forster
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; only version 2 of the License is applicable.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
 *
 * Authors:
 *   Florian octo Forster <octo at collectd.org>
 **/

#include "cpu.c" /* sic */
#include "testing.h"

DEF_TEST(usage_simple_rate) {
  usage_t usage = {0};

  cdtime_t t0 = TIME_T_TO_CDTIME_T(100);
  derive_t count_t0 = 3000;

  usage_init(&usage, t0);
  usage_record(&usage, 0, STATE_USER, count_t0);

  // Unable to calculate a rate with a single data point.
  EXPECT_EQ_DOUBLE(NAN, usage_rate(usage, 0, STATE_USER));

  cdtime_t t1 = t0 + TIME_T_TO_CDTIME_T(10);
  derive_t count_t1 = count_t0 + 100;
  gauge_t want_rate = 100.0 / 10.0;

  usage_init(&usage, t1);
  usage_record(&usage, 0, STATE_USER, count_t1);

  EXPECT_EQ_DOUBLE(want_rate, usage_rate(usage, 0, STATE_USER));

  // States that we have not set should be NAN
  for (state_t s = 0; s < STATE_MAX; s++) {
    if (s == STATE_USER) {
      continue;
    }
    EXPECT_EQ_DOUBLE(NAN, usage_rate(usage, 0, s));
  }

  usage_reset(&usage);
  return 0;
}

int main(void) {
  RUN_TEST(usage_simple_rate);

  END_TEST;
}
