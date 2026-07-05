#pragma once

#include <exception>
#include <memory>

#include <sqlite3.h>

namespace task_tracker {

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

sqlite3* openSQLite(const char* db_path, int flags);
void closeSQLite(sqlite3* db);

sqlite3_stmt* prepareSQLiteStatement(sqlite3* db, const char* sql);
void finalizeSQLiteStatement(sqlite3_stmt* stmt);

class SQLiteStatement {
  public:
    explicit SQLiteStatement(sqlite3_stmt* stmt)
    {
        this->stmt.reset(stmt);
    }

    explicit SQLiteStatement(sqlite3* db, const char* sql)
    {
        this->stmt.reset(prepareSQLiteStatement(db, sql));
    }

    template<typename Self, typename T>
    decltype(auto) bind(this Self&& self, T&& arg)
    {
        self.bindArg(std::forward<T>(arg));
        return std::forward<Self>(self);
    }
    // other bind overloads will be added when needed

    /* Bind multiple arguments in one call. */
    template<typename... Args>
    decltype(auto) bind(this auto&& self, Args&&... args)
    {
        return (self.bind(std::forward<Args>(args)), ...);
    }

    int step() noexcept
    {
        return sqlite3_step(stmt.get());
    }

    template<typename T>
    T column(int col);

    int columnType(int col);

  private:
    void bindArg(std::string_view text);

    std::unique_ptr<sqlite3_stmt, void (*)(sqlite3_stmt*)> stmt{
        nullptr, finalizeSQLiteStatement};
};

class SQLite {
  public:
    explicit SQLite(sqlite3* db)
    {
        this->db.reset(db);
    }

    explicit SQLite(const char* path, int flags)
    {
        db.reset(openSQLite(path, flags));
    }

    SQLiteStatement prepare(const char* sql)
    {
        return SQLiteStatement(db.get(), sql);
    }

    void exec(const char* sql);

  private:
    std::unique_ptr<sqlite3, void (*)(sqlite3*)> db{nullptr, closeSQLite};
};

} // namespace task_tracker
