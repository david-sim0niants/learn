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

    return db;
}

void closeDB(sqlite3* db)
{
    sqlite3_close_v2(db);
}

void ensureTablesExist(sqlite3* db)
{
    int ret = sqlite3_exec(
        db,
        "CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL);",
        nullptr, nullptr, nullptr);

    if (ret != SQLITE_OK)
        throw SQLiteException(ret);
}

auto prepareStatement(sqlite3* db, const char* sql)
{
    sqlite3_stmt* stmt = nullptr;
    int ret = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (ret != SQLITE_OK)
        throw SQLiteException(ret);

    return std::unique_ptr<sqlite3_stmt, int (*)(sqlite3_stmt*)>(
        stmt, sqlite3_finalize);
}

} // namespace

class TaskTrackerBase::Impl {
  public:
    explicit Impl(int flags)
    {
        db.reset(openDB(flags));
        ensureTablesExist(db.get());
    }

    void add(std::string_view title)
    {
        const char* sql = "INSERT INTO tasks (title) VALUES (?);";

        auto stmt = prepareStatement(db.get(), sql);

        int ret =
            sqlite3_bind_text(stmt.get(), 1, title.data(),
                              static_cast<int>(title.size()), SQLITE_TRANSIENT);
        if (ret != SQLITE_OK)
            throw SQLiteException(ret);

        ret = sqlite3_step(stmt.get());
        switch (ret) {
            case SQLITE_DONE:
                break;
            case SQLITE_ROW:
                throw SQLiteException(ret);
            default:
                throw SQLiteException(ret);
        }
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
