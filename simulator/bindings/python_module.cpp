#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hetero/memory/paged_kv.h"
#include "hetero/runtime/global_event_runtime.h"
#include "hetero/runtime/scheduler.h"
#include "hetero/services/fixed_latency_memory.h"

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
            request.contains("priority") ? py::cast<std::int64_t>(request["priority"]) : 0});
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

}  // namespace

PYBIND11_MODULE(_heterosim_runtime, module) {
    module.doc() = "C++ dynamic runtime boundary for GPU-ATLAS-HeteroSim";
    module.def("simulate_token_barrier", &simulate);
    module.def("allocate_paged_kv", &allocate_kv);
    module.def("run_task_dag", &run_task_dag);
    module.def(
        "ideal_link_completion_fs",
        &heterosim::services::ideal_link_completion_fs,
        "issue_time_fs"_a,
        "latency_fs"_a,
        "payload_bytes"_a,
        "header_bytes"_a,
        "wire_bandwidth_bytes_per_second"_a);
}
