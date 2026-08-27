#ifndef HETEROSIM_ACCEL_SIM_RAMULATOR2_BRIDGE_H
#define HETEROSIM_ACCEL_SIM_RAMULATOR2_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *heterosim_ramulator_handle;

heterosim_ramulator_handle heterosim_ramulator_create(
    const char *config_path, unsigned partition_id, unsigned partition_count);
void heterosim_ramulator_destroy(heterosim_ramulator_handle handle);
int heterosim_ramulator_send(heterosim_ramulator_handle handle,
                             uint64_t address, int is_write, void *payload);
void heterosim_ramulator_tick(heterosim_ramulator_handle handle);
void *heterosim_ramulator_pop_completed(heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_clock(heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_reads(heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_writes(heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_completed(heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_rejected(heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_outstanding(heterosim_ramulator_handle handle);
int heterosim_ramulator_is_finished(heterosim_ramulator_handle handle);
void heterosim_ramulator_finish(heterosim_ramulator_handle handle);

#ifdef __cplusplus
}
#endif

#endif
