#ifndef HETEROSIM_BOOKSIM2_CYCLE_ADAPTER_H
#define HETEROSIM_BOOKSIM2_CYCLE_ADAPTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HETEROSIM_BOOKSIM2_ABI_VERSION 1u

typedef struct heterosim_booksim2_completion_v1 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint64_t flow_id;
  uint32_t source_node;
  uint32_t destination_node;
  uint32_t flit_count;
  uint32_t reserved;
  uint64_t accepted_cycle;
  uint64_t completion_cycle;
} heterosim_booksim2_completion_v1;

typedef struct heterosim_booksim2_stats_v1 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint64_t cycles;
  uint64_t accepted_packets;
  uint64_t injected_packets;
  uint64_t delivered_packets;
  uint64_t accepted_flits;
  uint64_t delivered_flits;
  uint64_t outstanding_credits;
  uint64_t queued_packets;
  uint64_t completed_unpolled_packets;
  uint32_t network_drained;
  uint32_t zero_in_flight;
} heterosim_booksim2_stats_v1;

void* heterosim_booksim2_create(const char* config_path, uint32_t node_count);
void heterosim_booksim2_destroy(void* handle);
int heterosim_booksim2_submit(
    void* handle,
    uint64_t flow_id,
    uint32_t source_node,
    uint32_t destination_node,
    uint32_t flit_count);
int heterosim_booksim2_step(void* handle);
int heterosim_booksim2_pop_completion(
    void* handle,
    uint32_t destination_node,
    heterosim_booksim2_completion_v1* completion);
int heterosim_booksim2_get_stats(
    void* handle,
    heterosim_booksim2_stats_v1* stats);
const char* heterosim_booksim2_last_error(void);
const char* heterosim_booksim2_abi(void);

#ifdef __cplusplus
}
#endif

#endif
