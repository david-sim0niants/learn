#pragma once

#include <exception>
#include <memory>
#include <stdexcept>

#include <sqlite3.h>

namespace task_tracker {

class SQLiteException : public std::exception {
  public:
    explicit SQLiteException(int code) : code_(code)
    {
    }

    explicit SQLiteException(sqlite3* db, int code)
        : code_(code), context_error(sqlite3_errmsg(db)),
          full_error(std::string(message()) + ": " + contextError())
    {
    }

    const char* what() const noexcept override
    {
        return fullError();
    }

    inline int code() const noexcept
    {
        return code_;
    }

    inline const char* message() const noexcept
    {
        return sqlite3_errstr(code_);
    }

    inline const char* contextError() const noexcept
    {
        return context_error.c_str();
    }

    inline const char* fullError() const noexcept
    {
        if (full_error.empty())
            return message();
        else
            return full_error.c_str();
    }

  private:
    int code_;
    std::string context_error;
    std::string full_error;
};

sqlite3* openSQLite(const char* db_path, int flags);
void closeSQLite(sqlite3* db);

sqlite3_stmt* prepareSQLiteStatement(sqlite3* db, const char* sql);
void finalizeSQLiteStatement(sqlite3_stmt* stmt);

class SQLiteStatement {
  public:
    explicit SQLiteStatement(sqlite3* db, sqlite3_stmt* stmt) : db(db)
    {
        this->stmt.reset(stmt);
    }

    explicit SQLiteStatement(sqlite3* db, const char* sql) : db(db)
    {
        this->stmt.reset(prepareSQLiteStatement(db, sql));
    }

    /* Bind argument at the current index. */
    template<typename Self, typename T>
    decltype(auto) bind(this Self&& self, T&& arg)
    {
        self.bindArg(std::forward<T>(arg));
        ++self.arg_idx;
        return std::forward<Self>(self);
    }

    /* Bind argument at the given index. */
    template<typename Self, typename T>
    decltype(auto) bind(this Self&& self, unsigned int idx, T&& arg)
    {
        self.arg_idx = idx;
        self.bindArg(arg);
        ++self.arg_idx;
        return std::forward<Self>(self);
    }

    /* Bind named parameter. */
    template<typename Self, typename T>
    decltype(auto) bind(this Self&& self, const char* name, T&& arg)
    {
        int index = sqlite3_bind_parameter_index(self.stmt.get(), name);
        if (0 == index)
            throw std::invalid_argument(
                std::string("No such parameter in SQL statement: ") + name);
        int arg_idx = self.arg_idx;
        self.arg_idx = index;
        self.bindArg(arg);
        self.arg_idx = std::max(arg_idx, index) + 1;
    }

    /* Bind multiple arguments in one call starting from the current index. */
    template<typename... Args>
    decltype(auto) bind(this auto&& self, Args&&... args)
    {
        return (self.bind(std::forward<Args>(args)), ...);
    }

    /* Bind multiple arguments in one call starting from the given index. */
    template<typename... Args>
    decltype(auto) bind(this auto&& self, unsigned int idx, Args&&... args)
    {
        self.arg_idx = idx;
        return (self.bind(std::forward<Args>(args)), ...);
    }

    /* Will either return SQLITE_DONE, SQLITE_ERROR, or will throw an error. */
    int step();

    /* Supported types: std::string, std::optional<std::string>, int64_t */
    template<typename T>
    T column(int col);

    int columnType(int col);

  private:
    // other bind overloads will be added on demand

    void bindArg(std::string_view text);
    void bindArg(int64_t num);

    std::unique_ptr<sqlite3_stmt, void (*)(sqlite3_stmt*)> stmt{
        nullptr, finalizeSQLiteStatement};
    sqlite3* db;
    unsigned int arg_idx = 1;
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

    inline int getChanges()
    {
        return sqlite3_changes(db.get());
    }

    inline int64_t getLastInsertRowId() const noexcept
    {
        return sqlite3_last_insert_rowid(db.get());
    }

  private:
    std::unique_ptr<sqlite3, void (*)(sqlite3*)> db{nullptr, closeSQLite};
};

} // namespace task_tracker
