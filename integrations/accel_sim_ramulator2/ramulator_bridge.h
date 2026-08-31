#ifndef HETEROSIM_ACCEL_SIM_RAMULATOR2_BRIDGE_H
#define HETEROSIM_ACCEL_SIM_RAMULATOR2_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *heterosim_ramulator_handle;

#define HETEROSIM_RAMULATOR_ABI_VERSION 2U
#define HETEROSIM_MAX_BYTE_MASK_WORDS 2U

enum heterosim_memory_operation {
  HETEROSIM_MEMORY_READ = 0,
  HETEROSIM_MEMORY_WRITE = 1,
};

enum heterosim_parent_request_flags {
  HETEROSIM_REQUEST_BYTE_MASK_VALID = 1U << 0,
  HETEROSIM_REQUEST_SECTOR_MASK_VALID = 1U << 1,
};

enum heterosim_send_result {
  HETEROSIM_SEND_INVALID = -1,
  HETEROSIM_SEND_RETRY = 0,
  HETEROSIM_SEND_ACCEPTED = 1,
};

enum heterosim_address_translation_result {
  HETEROSIM_ADDRESS_INVALID = -1,
  HETEROSIM_ADDRESS_IDENTITY = 0,
  HETEROSIM_ADDRESS_TRANSLATED = 1,
};

enum heterosim_memory_initiator {
  HETEROSIM_INITIATOR_GPU = 0,
  HETEROSIM_INITIATOR_ATLAS_LOGIC_DIE = 1,
};

typedef struct heterosim_parent_request_v2 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint64_t parent_id;
  uint64_t global_address;
  uint32_t size_bytes;
  uint32_t partition_id;
  uint32_t operation;
  uint32_t flags;
  uint64_t byte_mask[HETEROSIM_MAX_BYTE_MASK_WORDS];
  uint32_t byte_mask_word_count;
  uint32_t sector_mask;
  uint64_t ordering_domain;
  uint64_t sequence_number;
  uint32_t qos_class;
  uint32_t reserved;
  void *payload;
} heterosim_parent_request_v2;

typedef struct heterosim_parent_completion_v2 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint64_t parent_id;
  uint32_t partition_id;
  uint32_t operation;
  uint32_t total_children;
  uint32_t completed_children;
  uint32_t durable;
  uint32_t initiator;
  void *payload;
} heterosim_parent_completion_v2;

/*
 * Translate a captured GPU address before L1/L2 lookup.  With no binding file
 * configured this is an explicit identity operation.  A configured table is
 * strict: the whole request must lie in exactly one capture or already-global
 * range, otherwise HETEROSIM_ADDRESS_INVALID is returned.
 */
int heterosim_translate_gpu_address(uint64_t trace_address,
                                    uint32_t size_bytes,
                                    uint64_t *global_address);
uint64_t heterosim_address_translated_requests(void);
uint64_t heterosim_address_already_global_requests(void);
uint64_t heterosim_address_unmapped_requests(void);
uint64_t heterosim_address_binding_ranges(void);

heterosim_ramulator_handle heterosim_ramulator_create(
    const char *config_path, unsigned partition_id, unsigned partition_count);
heterosim_ramulator_handle heterosim_ramulator_retain(
    heterosim_ramulator_handle handle);
void heterosim_ramulator_destroy(heterosim_ramulator_handle handle);
int heterosim_ramulator_send(heterosim_ramulator_handle handle,
                             uint64_t address, int is_write, void *payload);
int heterosim_ramulator_send_v2(
    heterosim_ramulator_handle handle,
    const heterosim_parent_request_v2 *request);
/*
 * Submit an ATLAS Logic-Die request at the internal Hybrid-Bond gateway port.
 * The request bypasses the GPU-facing external link, but uses the same parent
 * table, child splitter, scheduler, address mapper, and Ramulator2 instance.
 */
int heterosim_ramulator_send_internal_v2(
    heterosim_ramulator_handle handle,
    const heterosim_parent_request_v2 *request);
void heterosim_ramulator_tick(heterosim_ramulator_handle handle);
/*
 * Advance up to max_gpu_cycles, stopping immediately after the first cycle
 * that produces any parent completion.  This lets an external event runtime
 * skip idle compute intervals without losing the exact completion cycle.
 * Returns the number of GPU clock cycles actually advanced.
 */
uint64_t heterosim_ramulator_advance_until_event(
    heterosim_ramulator_handle handle, uint64_t max_gpu_cycles);
void heterosim_ramulator_advance_gpu_cycle(void);
int heterosim_ramulator_external_runtime_active(void);
void *heterosim_ramulator_pop_completed(heterosim_ramulator_handle handle);
int heterosim_ramulator_pop_completed_v2(
    heterosim_ramulator_handle handle,
    heterosim_parent_completion_v2 *completion);
/*
 * Pop only a completion owned by one initiator.  This is required when
 * Accel-Sim and a live ATLAS Chip share a partition completion queue: neither
 * backend is allowed to consume and reinterpret the other backend's payload.
 */
int heterosim_ramulator_pop_completed_for_initiator_v2(
    heterosim_ramulator_handle handle, uint32_t initiator,
    heterosim_parent_completion_v2 *completion);
uint64_t heterosim_ramulator_clock(heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_reads(heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_writes(heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_completed(heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_rejected(heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_outstanding(heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_durable_completed(
    heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_children_sent(
    heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_children_completed(
    heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_logical_bytes(
    heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_internal_bytes(
    heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_request_payload_bytes(
    heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_response_payload_bytes(
    heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_request_wire_bytes(
    heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_response_wire_bytes(
    heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_link_cycles(
    heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_gpu_cycles(
    heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_gateway_cycles(
    heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_global_time_fs(
    heterosim_ramulator_handle handle);
uint64_t heterosim_ramulator_initiator_parents(
    heterosim_ramulator_handle handle, uint32_t initiator);
uint64_t heterosim_ramulator_initiator_completed(
    heterosim_ramulator_handle handle, uint32_t initiator);
uint64_t heterosim_ramulator_initiator_children(
    heterosim_ramulator_handle handle, uint32_t initiator);
int heterosim_ramulator_compare_atlas_addresses(
    heterosim_ramulator_handle handle, uint64_t lhs, uint64_t rhs);
int heterosim_ramulator_is_finished(heterosim_ramulator_handle handle);
void heterosim_ramulator_finish(heterosim_ramulator_handle handle);

#ifdef __cplusplus
}
#endif

#endif
