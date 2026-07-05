CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL UNIQUE,
    status TEXT CHECK(status IN ('todo', 'in_progress', 'done')) NOT NULL,
    category_id INTEGER REFERENCES categories(id)
);

CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);
