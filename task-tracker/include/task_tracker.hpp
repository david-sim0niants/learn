#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string_view>

namespace task_tracker {

class TaskTrackerBase {
  protected:
    TaskTrackerBase() = default;
    ~TaskTrackerBase();

    TaskTrackerBase(const TaskTrackerBase&) = delete;
    TaskTrackerBase& operator=(const TaskTrackerBase&) = delete;

    TaskTrackerBase(TaskTrackerBase&&);
    TaskTrackerBase& operator=(TaskTrackerBase&&);

    class Impl;
    std::unique_ptr<Impl> impl;
};

enum class TaskStatus {
    Todo,
    InProgress,
    Done,
};

inline std::optional<TaskStatus> toTaskStatus(std::string_view status)
{
    if (status == "todo")
        return TaskStatus::Todo;
    else if (status == "in_progress")
        return TaskStatus::InProgress;
    else if (status == "done")
        return TaskStatus::Done;
    else
        return std::nullopt;
}

struct Task {
    int64_t id;
    std::string title;
    std::optional<std::string> category;
    TaskStatus status;
};

struct TaskFilter {
    std::optional<std::string> category;
    std::optional<TaskStatus> status;
};

class TaskTrackerView : protected TaskTrackerBase {
  public:
    static TaskTrackerView& instance();

    using Callback = std::function<bool(const Task& task)>;

    void list(const Callback& cb, const TaskFilter& filter = TaskFilter());
};

class TaskTracker final : public TaskTrackerView {
  public:
    static TaskTracker& instance();

    int64_t add(std::string_view title);
};

inline TaskTrackerView& taskTrackerView()
{
    return TaskTrackerView::instance();
}

inline TaskTracker& taskTracker()
{
    return TaskTracker::instance();
}

} // namespace task_tracker
