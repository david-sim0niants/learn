#include "sqlite_helpers.hpp"

#include <iostream>

namespace task_tracker {

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

void SQLiteStatement::bindArg(std::string_view text)
{
    throwIfError(db, sqlite3_bind_text(stmt.get(), 1, text.data(),
                                       static_cast<int>(text.size()),
                                       SQLITE_TRANSIENT));
}

void SQLite::exec(const char* sql)
{
    int ret = sqlite3_exec(db.get(), sql, nullptr, nullptr, nullptr);
    throwIfError(db.get(), ret);
}

} // namespace task_tracker
