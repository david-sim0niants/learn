#include "task_tracker.hpp"
#include "platform.hpp"
#include "sqlite_helpers.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

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
    auto init_db_path = platform::getSharedDataDirectory() / "init_db.sql";
    std::ifstream file(init_db_path);

    if (! file)
        throw std::runtime_error("Failed to open init_db.sql");

    std::stringstream ss;
    ss << file.rdbuf();

    db.exec(ss.str().c_str());
}

} // namespace

class TaskTrackerBase::Impl {
  public:
    explicit Impl(int flags) : db(openDB(flags))
    {
        ensureTablesExist(db);
    }

    int64_t add(std::string_view title)
    {
        constexpr char sql[] =
            "INSERT INTO tasks (title, status) VALUES (?, 'todo');";

        int ret = db.prepare(sql).bind(title).step();

        switch (ret) {
            case SQLITE_DONE:
                break;
            case SQLITE_ROW:
            default:
                throw SQLiteException(ret);
        }

        return db.getLastInsertRowId();
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

int64_t TaskTracker::add(std::string_view title)
{
    return impl->add(title);
}

} // namespace task_tracker
