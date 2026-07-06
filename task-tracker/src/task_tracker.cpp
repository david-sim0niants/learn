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
            "INSERT OR IGNORE INTO tasks (title, status) VALUES (?, 'todo');";

        if (db.prepare(sql).bind(title).step() != SQLITE_DONE)
            throw std::runtime_error("Unexpected error while adding task");
        else
            return db.getChanges() > 0 ? db.getLastInsertRowId() : -1;
    }

    std::optional<Task> get(int64_t id)
    {
        constexpr char sql[] = R""""(
            SELECT t.id, t.title, c.name, t.status
            FROM tasks t
            LEFT JOIN categories c
            ON t.category = c.id
            WHERE t.id = ?;
        )"""";

        auto stmt = db.prepare(sql).bind(id);
        if (stmt.step() == SQLITE_ROW)
            return toTask(stmt);
        else
            throw std::runtime_error("Unexpected error while retrieving task");
    }

    void list(const TaskTrackerView::Callback& cb, const TaskFilter& filter)
    {
        std::string sql = R""""(
            SELECT t.id, t.title, c.name, t.status
            FROM tasks t
            LEFT JOIN categories c
            ON t.category = c.id
            WHERE 1=1
        )"""";

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

        while (stmt.step() == SQLITE_ROW)
            cb(toTask(stmt));
    }

    int64_t update(int64_t id, const TaskUpdate& update)
    {
        std::string sql = "UPDATE tasks SET ";

        if (update.title.has_value())
            sql += "title = :title, ";

        if (update.category.has_value()) {
            ensureCategoryExists(*update.category);
            sql +=
                "category = (SELECT id FROM categories WHERE name = :category), ";
        }

        if (update.status.has_value())
            sql += "status = :status, ";

        if (sql.ends_with(", "))
            sql.erase(sql.size() - 2); // Remove trailing comma and space
        else
            return id; // nothing to be updated

        sql += " WHERE id = :id;";

        auto stmt = db.prepare(sql.c_str());

        if (update.title.has_value())
            stmt.bind(":title", *update.title);
        if (update.category.has_value())
            stmt.bind(":category", *update.category);
        if (update.status.has_value())
            stmt.bind(":status", toString(*update.status));

        stmt.bind(":id", id);

        if (stmt.step() != SQLITE_DONE)
            throw std::runtime_error("Unexpected error while updating task");

        if (db.getChanges() == 0)
            return -1; // No rows updated, task not found
        else
            return id;
    }

    int64_t remove(int64_t id)
    {
        constexpr char sql[] =
            "DELETE FROM tasks WHERE id = ? RETURNING category;";

        auto stmt = db.prepare(sql).bind(id);
        int ret = stmt.step();

        if (ret == SQLITE_ROW) {
            auto category_id = stmt.column<std::optional<int64_t>>(0);
            if (category_id)
                removeCategoryIfUnused(*category_id);
        }

        return db.getChanges() > 0 ? id : -1;
    }

  private:
    static Task toTask(SQLiteStatement& stmt)
    {
        assert(stmt.columnType(0) == SQLITE_INTEGER);
        assert(stmt.columnType(1) == SQLITE_TEXT);
        assert(stmt.columnType(2) == SQLITE_TEXT ||
               stmt.columnType(2) == SQLITE_NULL);
        assert(stmt.columnType(3) == SQLITE_TEXT);

        return Task{
            .id = stmt.column<int64_t>(0),
            .title = stmt.column<std::string>(1),
            .category = stmt.column<std::optional<std::string>>(2),
            .status = toTaskStatus(stmt.column<std::string>(3)).value(),
        };
    }

    void ensureCategoryExists(std::string_view category)
    {
        constexpr char sql[] =
            "INSERT OR IGNORE INTO categories (name) VALUES (?);";

        if (db.prepare(sql).bind(category).step() != SQLITE_DONE)
            throw std::runtime_error(
                "Unexpected error while creating or checking for category");
    }

    void removeCategoryIfUnused(int64_t category_id)
    {
        constexpr char sql[] =
            "DELETE FROM categories WHERE id = :id AND NOT EXISTS (SELECT 1 FROM tasks WHERE category = :id);";
        if (db.prepare(sql).bind(":id", category_id).step() != SQLITE_DONE)
            throw std::runtime_error(
                "Unexpected error while removing unused category");
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

std::optional<Task> TaskTrackerView::get(int64_t id)
{
    return impl->get(id);
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

int64_t TaskTracker::update(int64_t id, const TaskUpdate& update)
{
    return impl->update(id, update);
}

int64_t TaskTracker::remove(int64_t id)
{
    return impl->remove(id);
}

} // namespace task_tracker
