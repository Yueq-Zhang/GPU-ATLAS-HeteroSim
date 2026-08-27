#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hetero/memory/paged_kv.h"
#include "hetero/memory/runtime_memory_planner.h"
#include "hetero/runtime/global_event_runtime.h"
#include "hetero/runtime/scheduler.h"
#include "hetero/services/bounded_link.h"
#include "hetero/services/fixed_latency_memory.h"
#include "hetero/services/shared_3d_memory.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

std::uint64_t get_u64(
    const py::dict& object,
    const char* key,
    std::uint64_t default_value = 0) {
    if (!object.contains(key)) {
        return default_value;
    }
    return py::cast<std::uint64_t>(object[key]);
}

py::dict simulate(const py::list& request_objects, const py::dict& scheduler) {
    std::vector<heterosim::runtime::RequestInput> requests;
    requests.reserve(py::len(request_objects));
    for (const auto& item : request_objects) {
        const auto request = py::cast<py::dict>(item);
        requests.push_back(heterosim::runtime::RequestInput{
            py::cast<std::string>(request["request_id"]),
            get_u64(request, "arrival_time_fs"),
            get_u64(request, "prompt_length"),
            get_u64(request, "output_length"),
            request.contains("priority") ? py::cast<std::int64_t>(request["priority"]) : 0,
            request.contains("execution_scope") &&
                py::cast<std::string>(request["execution_scope"]) == "decode_step",
            get_u64(request, "initial_kv_length")});
    }
    const heterosim::runtime::SchedulerConfig config{
        get_u64(scheduler, "max_num_sequences"),
        get_u64(scheduler, "max_batched_tokens"),
        get_u64(scheduler, "prefill_chunk_tokens"),
        get_u64(scheduler, "max_prefill_wait_epochs", 8),
        get_u64(scheduler, "epoch_duration_fs")};
    const auto result = heterosim::runtime::simulate_token_barrier(requests, config);

    py::list epochs;
    for (const auto& epoch : result.epochs) {
        py::list selections;
        for (const auto& selection : epoch.selections) {
            selections.append(py::dict(
                "request_id"_a = selection.request_id,
                "phase"_a = selection.phase == heterosim::runtime::RequestPhase::kPrefill
                                  ? "prefill"
                                  : "decode",
                "token_begin"_a = selection.token_begin,
                "token_count"_a = selection.token_count));
        }
        epochs.append(py::dict(
            "epoch_id"_a = epoch.epoch_id,
            "boundary_time_fs"_a = epoch.boundary_time_fs,
            "completion_time_fs"_a = epoch.completion_time_fs,
            "selections"_a = selections));
    }
    py::list request_results;
    for (const auto& request : result.requests) {
        request_results.append(py::dict(
            "request_id"_a = request.request_id,
            "generated_length"_a = request.generated_length,
            "committed_kv_length"_a = request.committed_kv_length,
            "token_ready_time_fs"_a = request.token_ready_time_fs,
            "finish_time_fs"_a = request.finish_time_fs));
    }
    return py::dict(
        "schema_version"_a = "hetero-runtime-result/v1",
        "epochs"_a = epochs,
        "requests"_a = request_results);
}

py::dict allocate_kv(
    const py::list& request_objects,
    const py::dict& model,
    const py::dict& address,
    const std::string& memory_space_id) {
    const auto capacity = get_u64(
        address, "kv_capacity_bytes", std::numeric_limits<std::uint64_t>::max());
    heterosim::memory::PagedKvAllocator allocator(memory_space_id, capacity);
    const heterosim::memory::PagedKvGeometry geometry{
        get_u64(model, "num_layers"),
        get_u64(model, "num_kv_heads"),
        get_u64(model, "head_dim"),
        get_u64(model, "bytes_per_element", 2),
        get_u64(address, "page_tokens")};
    py::list allocations;
    for (const auto& item : request_objects) {
        const auto request = py::cast<py::dict>(item);
        const auto allocation = allocator.allocate(
            heterosim::memory::PagedKvRequest{
                py::cast<std::string>(request["request_id"]),
                get_u64(request, "prompt_length"),
                get_u64(request, "output_length")},
            geometry);
        allocations.append(py::dict(
            "request_id"_a = allocation.request_id,
            "memory_space_id"_a = allocation.physical_address.memory_space_id,
            "offset_bytes"_a = allocation.physical_address.offset_bytes,
            "allocation_epoch"_a = allocation.physical_address.allocation_epoch,
            "final_committed_tokens"_a = allocation.final_committed_tokens,
            "allocated_blocks"_a = allocation.allocated_blocks,
            "bytes_per_block"_a = allocation.bytes_per_block,
            "logical_bytes"_a = allocation.logical_bytes,
            "allocated_bytes"_a = allocation.allocated_bytes));
    }
    return py::dict(
        "schema_version"_a = "hetero-buffer-bindings/v1",
        "memory_space_id"_a = memory_space_id,
        "used_bytes"_a = allocator.used_bytes(),
        "allocations"_a = allocations);
}

py::dict run_task_dag(const py::list& task_objects) {
    std::vector<heterosim::runtime::RuntimeTask> tasks;
    tasks.reserve(py::len(task_objects));
    for (const auto& item : task_objects) {
        const auto task = py::cast<py::dict>(item);
        tasks.push_back(heterosim::runtime::RuntimeTask{
            py::cast<std::string>(task["task_id"]),
            py::cast<std::string>(task["resource_id"]),
            task.contains("dependencies")
                ? py::cast<std::vector<std::string>>(task["dependencies"])
                : std::vector<std::string>{},
            get_u64(task, "release_time_fs"),
            get_u64(task, "duration_fs")});
    }
    const auto result = heterosim::runtime::GlobalEventRuntime{}.run(tasks);
    py::list timings;
    for (const auto& timing : result.tasks) {
        timings.append(py::dict(
            "task_id"_a = timing.task_id,
            "resource_id"_a = timing.resource_id,
            "ready_time_fs"_a = timing.ready_time_fs,
            "start_time_fs"_a = timing.start_time_fs,
            "completion_time_fs"_a = timing.completion_time_fs));
    }
    return py::dict(
        "schema_version"_a = "hetero-runtime-dag-result/v1",
        "makespan_fs"_a = result.makespan_fs,
        "tasks"_a = timings);
}

py::dict plan_memory_lifecycle(
    const py::list& space_objects,
    const py::list& event_objects) {
    std::vector<heterosim::memory::MemorySpaceSpec> spaces;
    for (const auto& item : space_objects) {
        const auto space = py::cast<py::dict>(item);
        spaces.push_back(heterosim::memory::MemorySpaceSpec{
            py::cast<std::string>(space["memory_space_id"]),
            get_u64(space, "capacity_bytes"),
            get_u64(space, "base_alignment_bytes", 64)});
    }
    heterosim::memory::RuntimeMemoryPlanner planner(spaces);
    py::list events;
    for (const auto& item : event_objects) {
        const auto event = py::cast<py::dict>(item);
        const auto operation = py::cast<std::string>(event["operation"]);
        const auto allocation_id = py::cast<std::string>(event["allocation_id"]);
        heterosim::memory::AllocationRecord record;
        if (operation == "allocate") {
            record = planner.allocate(heterosim::memory::AllocationRequest{
                allocation_id,
                py::cast<std::string>(event["memory_space_id"]),
                get_u64(event, "size_bytes"),
                get_u64(event, "alignment_bytes", 64),
                event.contains("lifetime")
                    ? py::cast<std::string>(event["lifetime"])
                    : std::string("request")});
        } else if (operation == "release") {
            record = planner.release(allocation_id);
        } else {
            throw std::invalid_argument("unknown memory lifecycle operation: " + operation);
        }
        events.append(py::dict(
            "time_fs"_a = get_u64(event, "time_fs"),
            "operation"_a = operation,
            "allocation_id"_a = record.allocation_id,
            "memory_space_id"_a = record.physical_address.memory_space_id,
            "offset_bytes"_a = record.physical_address.offset_bytes,
            "allocation_epoch"_a = record.physical_address.allocation_epoch,
            "size_bytes"_a = record.size_bytes,
            "active"_a = record.active));
    }
    py::list memory_spaces;
    for (const auto& item : space_objects) {
        const auto space = py::cast<py::dict>(item);
        const auto id = py::cast<std::string>(space["memory_space_id"]);
        memory_spaces.append(py::dict(
            "memory_space_id"_a = id,
            "used_bytes"_a = planner.used_bytes(id),
            "peak_bytes"_a = planner.peak_bytes(id)));
    }
    return py::dict(
        "schema_version"_a = "hetero-memory-lifecycle/v1",
        "events"_a = events,
        "memory_spaces"_a = memory_spaces);
}

py::dict simulate_bounded_link(
    const py::dict& config,
    const py::list& transaction_objects) {
    const heterosim::services::BoundedLinkConfig link_config{
        py::cast<std::string>(config["route_id"]),
        get_u64(config, "wire_bandwidth_Bps"),
        get_u64(config, "latency_fs"),
        get_u64(config, "queue_depth_transactions"),
        get_u64(config, "credits"),
        config.contains("full_duplex") ? py::cast<bool>(config["full_duplex"]) : true};
    std::vector<heterosim::services::LinkTransaction> transactions;
    for (const auto& item : transaction_objects) {
        const auto transaction = py::cast<py::dict>(item);
        const auto payload = get_u64(transaction, "payload_bytes");
        const auto header = get_u64(transaction, "header_bytes");
        transactions.push_back(heterosim::services::LinkTransaction{
            get_u64(transaction, "transaction_id"),
            get_u64(transaction, "parent_task_id"),
            py::cast<std::string>(config["route_id"]),
            py::cast<std::string>(transaction["source_id"]),
            py::cast<std::string>(transaction["destination_id"]),
            payload,
            header,
            payload + header,
            get_u64(transaction, "issue_time_fs")});
    }
    const auto result = heterosim::services::BoundedLinkModel(link_config).run(transactions);
    py::list responses;
    for (const auto& response : result.responses) {
        responses.append(py::dict(
            "transaction_id"_a = response.transaction_id,
            "parent_task_id"_a = response.parent_task_id,
            "route_id"_a = response.route_id,
            "completion_time_fs"_a = response.completion_time_fs,
            "payload_bytes"_a = response.payload_bytes,
            "wire_bytes"_a = response.wire_bytes));
    }
    return py::dict(
        "schema_version"_a = "hetero-bounded-link-result/v1",
        "responses"_a = responses,
        "submitted_transactions"_a = result.submitted_transactions,
        "completed_transactions"_a = result.completed_transactions,
        "payload_bytes"_a = result.payload_bytes,
        "wire_bytes"_a = result.wire_bytes,
        "backpressure_events"_a = result.backpressure_events,
        "last_completion_fs"_a = result.last_completion_fs);
}

py::dict simulate_shared_3d_memory(
    const py::dict& config,
    const py::list& request_objects) {
    const auto initiators = py::cast<std::vector<std::string>>(config["initiator_order"]);
    const heterosim::services::Shared3DMemoryConfig memory_config{
        py::cast<std::string>(config["memory_space_id"]),
        initiators,
        get_u64(config, "channel_count"),
        get_u64(config, "banks_per_channel"),
        get_u64(config, "transaction_bytes", 64),
        get_u64(config, "queue_depth_per_initiator", 64),
        get_u64(config, "fixed_latency_fs"),
        get_u64(config, "channel_injection_interval_fs"),
        get_u64(config, "bank_busy_time_fs")};
    std::vector<heterosim::services::MemoryRequest> requests;
    for (const auto& item : request_objects) {
        const auto request = py::cast<py::dict>(item);
        const auto operation = request.contains("operation")
                                   ? py::cast<std::string>(request["operation"])
                                   : std::string("read");
        requests.push_back(heterosim::services::MemoryRequest{
            get_u64(request, "request_id"),
            get_u64(request, "parent_task_id"),
            py::cast<std::string>(request["initiator_id"]),
            heterosim::PhysicalAddress{
                py::cast<std::string>(config["memory_space_id"]),
                get_u64(request, "offset_bytes"),
                get_u64(request, "allocation_epoch", 1)},
            request.contains("value_id")
                ? py::cast<std::string>(request["value_id"])
                : std::string("anonymous"),
            get_u64(request, "value_version"),
            get_u64(request, "size_bytes"),
            operation == "write" ? heterosim::services::MemoryOperation::kWrite
                                 : heterosim::services::MemoryOperation::kRead,
            get_u64(request, "issue_time_fs"),
            get_u64(request, "ordering_domain"),
            get_u64(request, "sequence_number"),
            static_cast<std::uint32_t>(get_u64(request, "qos_class"))});
    }
    const auto result = heterosim::services::Shared3DMemoryModel(memory_config).run(requests);
    py::list parent_responses;
    for (const auto& response : result.parent_responses) {
        parent_responses.append(py::dict(
            "request_id"_a = response.request_id,
            "parent_task_id"_a = response.parent_task_id,
            "completion_time_fs"_a = response.completion_time_fs,
            "completed_bytes"_a = response.completed_bytes));
    }
    py::list child_records;
    for (const auto& child : result.child_records) {
        child_records.append(py::dict(
            "child_request_id"_a = child.child_request_id,
            "parent_request_id"_a = child.parent_request_id,
            "initiator_id"_a = child.initiator_id,
            "offset_bytes"_a = child.physical_address.offset_bytes,
            "channel"_a = child.dram_address.channel,
            "bank"_a = child.dram_address.bank,
            "row"_a = child.dram_address.row,
            "column"_a = child.dram_address.column,
            "size_bytes"_a = child.size_bytes,
            "issue_time_fs"_a = child.issue_time_fs,
            "service_start_fs"_a = child.service_start_fs,
            "completion_time_fs"_a = child.completion_time_fs));
    }
    return py::dict(
        "schema_version"_a = "hetero-shared-3d-memory-result/v1",
        "timing_owner"_a = "shared3d.memory_service",
        "parent_responses"_a = parent_responses,
        "child_records"_a = child_records,
        "parent_requests_submitted"_a = result.parent_requests_submitted,
        "parent_requests_completed"_a = result.parent_requests_completed,
        "child_requests_submitted"_a = result.child_requests_submitted,
        "child_requests_completed"_a = result.child_requests_completed,
        "submitted_bytes"_a = result.submitted_bytes,
        "completed_bytes"_a = result.completed_bytes,
        "backpressure_events"_a = result.backpressure_events,
        "requests_by_initiator"_a = result.requests_by_initiator,
        "requests_by_channel"_a = result.requests_by_channel,
        "last_completion_fs"_a = result.last_completion_fs);
}

}  // namespace

PYBIND11_MODULE(_heterosim_runtime, module) {
    module.doc() = "C++ dynamic runtime boundary for GPU-ATLAS-HeteroSim";
    module.def("simulate_token_barrier", &simulate);
    module.def("allocate_paged_kv", &allocate_kv);
    module.def("run_task_dag", &run_task_dag);
    module.def("plan_memory_lifecycle", &plan_memory_lifecycle);
    module.def("simulate_bounded_link", &simulate_bounded_link);
    module.def("simulate_shared_3d_memory", &simulate_shared_3d_memory);
    module.def(
        "ideal_link_completion_fs",
        &heterosim::services::ideal_link_completion_fs,
        "issue_time_fs"_a,
        "latency_fs"_a,
        "payload_bytes"_a,
        "header_bytes"_a,
        "wire_bandwidth_bytes_per_second"_a);
}
