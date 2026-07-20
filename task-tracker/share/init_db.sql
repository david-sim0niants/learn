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

CREATE TABLE IF NOT EXISTS task_history (
    id INTEGER PRIMARY KEY,
    task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    old_status TEXT CHECK(old_status IN ('todo', 'in_progress', 'done')) NOT NULL,
    new_status TEXT CHECK(new_status IN ('todo', 'in_progress', 'done')) NOT NULL,
    change_time TEXT DEFAULT(datetime('now')) NOT NULL
);

CREATE TRIGGER IF NOT EXISTS log_status_change
AFTER UPDATE OF status ON tasks
WHEN OLD.status != NEW.status
BEGIN
    INSERT INTO task_history (task_id, old_status, new_status, change_time)
    VALUES (OLD.id, OLD.status, NEW.status, datetime('now'));
END;
