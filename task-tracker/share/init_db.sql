CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL UNIQUE,
    category INTEGER REFERENCES categories(id),
    status TEXT CHECK(status IN ('todo', 'in_progress', 'done')) NOT NULL
);

CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);
