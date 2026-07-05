#include "task_tracker.hpp"
#include "platform.hpp"
#include "sqlite_helpers.hpp"

#include <cassert>
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

static const char* toString(TaskStatus status)
{
    switch (status) {
        case TaskStatus::Todo:
            return "todo";
        case TaskStatus::InProgress:
            return "in_progress";
        case TaskStatus::Done:
            return "done";
        default:
            throw std::invalid_argument("Invalid task status");
    }
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

    void list(const TaskTrackerView::Callback& cb, const TaskFilter& filter)
    {
        std::string sql = R""""(
            SELECT t.id, t.title, c.name, t.status
            FROM tasks t
            LEFT JOIN categories c
            ON t.category = c.id
    )"""";

        sql += " WHERE 1=1";
        if (filter.category.has_value())
            sql += " AND c.name = ?";
        if (filter.status.has_value())
            sql += " AND t.status = ?";

        sql += ';';

        auto stmt = db.prepare(sql.c_str());

        if (filter.category.has_value())
            stmt.bind(*filter.category);

        if (filter.status.has_value())
            stmt.bind(toString(*filter.status));

        while (stmt.step() == SQLITE_ROW) {
            assert(stmt.columnType(0) == SQLITE_INTEGER);
            assert(stmt.columnType(1) == SQLITE_TEXT);
            assert(stmt.columnType(2) == SQLITE_TEXT ||
                   stmt.columnType(2) == SQLITE_NULL);
            assert(stmt.columnType(3) == SQLITE_TEXT);

            Task task = {
                .id = stmt.column<int64_t>(0),
                .title = stmt.column<std::string>(1),
                .category = stmt.column<std::optional<std::string>>(2),
                .status = toTaskStatus(stmt.column<std::string>(3)).value(),
            };
            cb(task);
        }
    }

  private:
    std::string fetchCategory(int64_t category_id)
    {
        constexpr char sql[] = "SELECT name FROM categories WHERE id = ?;";

        auto stmt = db.prepare(sql).bind(category_id);

        if (stmt.step() != SQLITE_ROW)
            return "";

        return stmt.column<std::string>(0);
    }

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

void TaskTrackerView::list(const Callback& cb, const TaskFilter& filter)
{
    return impl->list(cb, filter);
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
