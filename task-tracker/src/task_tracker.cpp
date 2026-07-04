#include "task_tracker.hpp"
#include "platform.hpp"

#include <filesystem>
#include <iostream>

#include <sqlite3.h>

namespace task_tracker {

namespace {

class SQLiteException : public std::exception {
  public:
	explicit SQLiteException(int code) : code(code)
	{
	}

	const char* what() const noexcept override
	{
		return sqlite3_errstr(code);
	}

  private:
	int code;
};

sqlite3* openDB(int flags)
{
	const auto db_path = platform::getEnsuredDataDirectory() / "main.db";

	sqlite3* db = nullptr;
	int ret = sqlite3_open_v2(db_path.c_str(), &db, flags, nullptr);

	if (ret != SQLITE_OK)
		throw SQLiteException(ret);
	else
		return db;
}

void closeDB(sqlite3* db)
{
	sqlite3_close_v2(db);
}

} // namespace

class TaskTrackerBase::Impl {
  public:
	explicit Impl(int flags)
	{
		db.reset(openDB(flags));
	}

  private:
	std::unique_ptr<sqlite3, void (*)(sqlite3*)> db{nullptr, closeDB};
};

TaskTrackerBase::TaskTrackerBase(TaskTrackerBase&&) = default;
TaskTrackerBase& TaskTrackerBase::operator=(TaskTrackerBase&&) = default;
TaskTrackerBase::~TaskTrackerBase() = default;

TaskTrackerView& TaskTrackerView::instance()
{
	static TaskTrackerView instance;
	int flags = SQLITE_OPEN_READONLY;
	return instance.impl
			   ? instance
			   : (instance.impl = std::make_unique<Impl>(flags), instance);
}

void TaskTracker::add(std::string_view title)
{
	std::cout << "Adding task: " << title << std::endl;
}

TaskTracker& TaskTracker::instance()
{
	static TaskTracker instance;
	int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE;
	return instance.impl
			   ? instance
			   : (instance.impl = std::make_unique<Impl>(flags), instance);
}

} // namespace task_tracker
