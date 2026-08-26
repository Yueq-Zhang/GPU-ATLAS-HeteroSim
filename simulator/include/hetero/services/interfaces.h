#pragma once

#include <cstdint>
#include <string>

#include "hetero/types.h"

namespace heterosim::services {

enum class MemoryOperation : std::uint8_t { kRead, kWrite };

struct MemoryRequest {
    std::uint64_t request_id{};
    std::uint64_t parent_task_id{};
    std::string initiator_id;
    heterosim::PhysicalAddress physical_address;
    std::string value_id;
    std::uint64_t value_version{};
    std::uint64_t size_bytes{};
    MemoryOperation operation{MemoryOperation::kRead};
    TimeFs issue_time_fs{};
    std::uint64_t ordering_domain{};
    std::uint64_t sequence_number{};
    std::uint32_t qos_class{};
};

struct LinkTransaction {
    std::uint64_t transaction_id{};
    std::uint64_t parent_task_id{};
    std::string route_id;
    std::string source_id;
    std::string destination_id;
    std::uint64_t payload_bytes{};
    std::uint64_t header_bytes{};
    std::uint64_t wire_bytes{};
    TimeFs issue_time_fs{};
};

enum class CompletionStatus : std::uint8_t { kSuccess, kCancelled, kError };
enum class SubmitDisposition : std::uint8_t { kAccepted, kRetryAt };

struct SubmitResult {
    SubmitDisposition disposition{SubmitDisposition::kAccepted};
    TimeFs retry_at_fs{};
};

struct MemoryResponse {
    std::uint64_t request_id{};
    std::uint64_t parent_task_id{};
    heterosim::PhysicalAddress physical_address;
    std::string value_id;
    std::uint64_t value_version{};
    TimeFs completion_time_fs{};
    CompletionStatus status{CompletionStatus::kSuccess};
    std::uint64_t completed_bytes{};
};

struct LinkResponse {
    std::uint64_t transaction_id{};
    std::uint64_t parent_task_id{};
    std::string route_id;
    TimeFs completion_time_fs{};
    CompletionStatus status{CompletionStatus::kSuccess};
    std::uint64_t payload_bytes{};
    std::uint64_t wire_bytes{};
};

struct TaskDescriptor {
    std::uint64_t task_id{};
    std::string backend_id;
    std::string task_signature;
};

struct ArtifactRef {
    std::string artifact_key;
    std::string path;
};

class RuntimeCallbacks {
public:
    virtual ~RuntimeCallbacks() = default;
    virtual void schedule_wakeup(
        const std::string& component_id,
        TimeFs time_fs,
        std::uint64_t token) = 0;
    virtual void schedule_task_completion(
        std::uint64_t task_id,
        TimeFs time_fs,
        CompletionStatus status) = 0;
    virtual void schedule_memory_response(const MemoryResponse& response) = 0;
    virtual void schedule_link_completion(const LinkResponse& response) = 0;
};

class IExecutionBackend {
public:
    virtual ~IExecutionBackend() = default;
    virtual void prepare(const TaskDescriptor& task, const ArtifactRef& artifact) = 0;
    virtual bool can_accept(const TaskDescriptor& task, TimeFs now_fs) const = 0;
    virtual SubmitResult submit(
        const TaskDescriptor& task,
        TimeFs now_fs,
        RuntimeCallbacks& callbacks) = 0;
    virtual void on_wakeup(
        std::uint64_t token,
        TimeFs now_fs,
        RuntimeCallbacks& callbacks) = 0;
    virtual void on_memory_response(
        const MemoryResponse& response,
        TimeFs now_fs,
        RuntimeCallbacks& callbacks) = 0;
    virtual bool quiescent() const noexcept = 0;
};

class IMemoryService {
public:
    virtual ~IMemoryService() = default;
    virtual void bind_runtime(RuntimeCallbacks& callbacks) = 0;
    virtual bool can_accept(const MemoryRequest& request, TimeFs now_fs) const = 0;
    virtual SubmitResult try_submit(const MemoryRequest& request, TimeFs now_fs) = 0;
    virtual void on_wakeup(
        std::uint64_t token,
        TimeFs now_fs,
        RuntimeCallbacks& callbacks) = 0;
    virtual void advance_to(TimeFs time_fs) = 0;
    virtual bool quiescent() const noexcept = 0;
};

class ILinkService {
public:
    virtual ~ILinkService() = default;
    virtual void bind_runtime(RuntimeCallbacks& callbacks) = 0;
    virtual bool can_accept(const LinkTransaction& transaction, TimeFs now_fs) const = 0;
    virtual SubmitResult try_submit(
        const LinkTransaction& transaction,
        TimeFs now_fs) = 0;
    virtual void on_wakeup(
        std::uint64_t token,
        TimeFs now_fs,
        RuntimeCallbacks& callbacks) = 0;
    virtual void advance_to(TimeFs time_fs) = 0;
    virtual bool quiescent() const noexcept = 0;
};

}  // namespace heterosim::services
