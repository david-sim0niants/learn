#pragma once

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

class TaskTrackerView : protected TaskTrackerBase {
  public:
    static TaskTrackerView& instance();
};

class TaskTracker final : TaskTrackerView {
  public:
	static TaskTracker& instance();

	void add(std::string_view title);
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
