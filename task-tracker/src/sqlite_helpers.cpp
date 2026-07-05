#include "sqlite_helpers.hpp"

#include <cassert>
#include <limits>
#include <optional>

namespace task_tracker {

namespace {

inline void throwIfError(int ret)
{
    if (ret != SQLITE_OK)
        throw SQLiteException(ret);
}

inline void throwIfError(sqlite3* db, int ret)
{
    if (ret != SQLITE_OK)
        throw SQLiteException(db, ret);
}

} // namespace

sqlite3* openSQLite(const char* db_path, int flags)
{
    sqlite3* db = nullptr;
    throwIfError(sqlite3_open_v2(db_path, &db, flags, nullptr));
    sqlite3_extended_result_codes(db, 1);
    return db;
}

void closeSQLite(sqlite3* db)
{
    sqlite3_close_v2(db);
}

sqlite3_stmt* prepareSQLiteStatement(sqlite3* db, const char* sql)
{
    sqlite3_stmt* stmt = nullptr;
    throwIfError(db, sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr));
    return stmt;
}

void finalizeSQLiteStatement(sqlite3_stmt* stmt)
{
    sqlite3_finalize(stmt);
}

int SQLiteStatement::step()
{
    int ret = sqlite3_step(stmt.get());
    if (ret != SQLITE_DONE && ret != SQLITE_ROW)
        throw SQLiteException(db, ret);
    else
        return ret;
}

template<>
int64_t SQLiteStatement::column(int col)
{
    return sqlite3_column_int64(stmt.get(), col);
}

template<>
std::string SQLiteStatement::column(int col)
{
    auto text = sqlite3_column_text(stmt.get(), col);
    if (text)
        return reinterpret_cast<const char*>(text);
    else
        return "";
}

template<>
std::optional<std::string> SQLiteStatement::column(int col)
{
    auto text = sqlite3_column_text(stmt.get(), col);
    if (text)
        return reinterpret_cast<const char*>(text);
    else
        return std::nullopt;
}

int SQLiteStatement::columnType(int col)
{
    return sqlite3_column_type(stmt.get(), col);
}

void SQLiteStatement::bindArg(std::string_view text)
{
    assert(text.size() <= (size_t)std::numeric_limits<int>::max());
    throwIfError(db, sqlite3_bind_text(stmt.get(), arg_idx, text.data(),
                                       static_cast<int>(text.size()),
                                       SQLITE_TRANSIENT));
}

void SQLiteStatement::bindArg(int64_t num)
{
    throwIfError(db, sqlite3_bind_int64(stmt.get(), arg_idx, num));
}

void SQLite::exec(const char* sql)
{
    int ret = sqlite3_exec(db.get(), sql, nullptr, nullptr, nullptr);
    throwIfError(db.get(), ret);
}

} // namespace task_tracker
