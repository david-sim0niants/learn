#include "task_tracker.hpp"
#include "platform.hpp"
#include "sqlite_helpers.hpp"

#include <filesystem>
#include <iostream>

#include <sqlite3.h>

namespace task_tracker {

namespace {

SQLite openDB(int flags)
{
    const auto db_path = platform::getEnsuredDataDirectory() / "main.db";
    return SQLite(db_path.c_str(), flags);
}

void ensureTablesExist(SQLite& db)
{
    db.exec(
        "CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL);");
}

} // namespace

class TaskTrackerBase::Impl {
  public:
    explicit Impl(int flags) : db(openDB(flags))
    {
        ensureTablesExist(db);
    }

    void add(std::string_view title)
    {
        const char* sql = "INSERT INTO tasks (title) VALUES (?);";

        int ret = db.prepare(sql).bind(title).step();

        switch (ret) {
            case SQLITE_DONE:
                break;
            case SQLITE_ROW:
            default:
                throw SQLiteException(ret);
        }
    }

  private:
    SQLite db;
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

TaskTracker& TaskTracker::instance()
{
    static TaskTracker instance;
    int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE;
    return instance.impl
               ? instance
               : (instance.impl = std::make_unique<Impl>(flags), instance);
}

void TaskTracker::add(std::string_view title)
{
    impl->add(title);
}

} // namespace task_tracker
