#include <iostream>
#include <print>
#include <tuple>

#include "task_tracker.hpp"

#include <CLI/CLI.hpp>

using task_tracker::TaskTracker;
using task_tracker::taskTrackerView;
using task_tracker::taskTracker;

namespace {

struct SubCmdCtx {
    CLI::App& parser;
    int argc;
    char** argv;
};

using Subcommand =
    std::tuple<const char*, const char*, int (*)(SubCmdCtx&& ctx)>;

int runSubcommand(const Subcommand& sub_cmd, int argc, char* argv[])
{
    auto& [name, description, callback] = sub_cmd;

    CLI::App parser(description);

    struct Defer {
        char** argv;
        char* sub_cmd_arg;

        ~Defer()
        {
            argv[1] = sub_cmd_arg;
        }
    } defer{argv, argv[1]};

    std::string full_cmd = std::format("{} {}", argv[0], name);
    argv[1] = full_cmd.data();

    return callback(SubCmdCtx{parser, argc - 1, &argv[1]});
}

bool printTask(const task_tracker::Task& task)
{
    const char* status_map[] = {"Todo       ", "In Progress", "Done       "};
    const char* status_str = status_map[std::to_underlying(task.status)];
    const char* category =
        task.category.has_value() ? task.category->c_str() : "(none)";

    std::println(std::cout, "{}. {} | {} / {}", task.id, status_str, category,
                 task.title);
    return true;
}

bool printHistoryEntry(const task_tracker::TaskHistoryEntry& entry)
{
    const char* status_map[] = {"Todo       ", "In Progress", "Done       "};
    std::println("{}. Task ID: {}, Old Status: {} | "
                 "New Status: {} | Change Time: {:%Y-%m-%d %H:%M:%S}",
                 entry.id, entry.task_id,
                 status_map[std::to_underlying(entry.old_status)],
                 status_map[std::to_underlying(entry.new_status)],
                 entry.change_time);
    return true;
}

bool parseStatus(std::optional<task_tracker::TaskStatus>& status,
                 std::string_view status_str)
{
    status = task_tracker::toTaskStatus(status_str);
    if (! status.has_value() && ! status_str.empty()) {
        std::cerr << "Invalid status: " << status_str << std::endl;
        return false;
    }
    return true;
}

namespace subcommands {

int add(SubCmdCtx&& ctx)
{
    auto& parser = ctx.parser;

    std::string title;
    parser.add_option("title", title, "Title of the task")->required();

    CLI11_PARSE(parser, ctx.argc, ctx.argv);

    int64_t task_id = taskTracker().add(title);
    if (task_id < 0) {
        std::cerr << "Task with this title already exists: " << title
                  << std::endl;
        return 1;
    }

    auto task = taskTrackerView().get(task_id);
    if (! task) {
        std::println(std::cerr,
                     "Error: task with id {} was not found after adding",
                     task_id);
        return 1;
    }

    std::println("Added task: ");
    printTask(*task);
    return 0;
}

int list(SubCmdCtx&& ctx)
{
    auto& parser = ctx.parser;

    task_tracker::TaskFilter filter;
    parser.add_option("--category", filter.category,
                      "Filter tasks by category");

    std::string status;
    parser.add_option("--status", status,
                      "Filter tasks by status (todo, in_progress, done)");

    CLI11_PARSE(parser, ctx.argc, ctx.argv);

    if (! parseStatus(filter.status, status))
        return 1;

    taskTrackerView().list(printTask, filter);
    return 0;
}

int update(SubCmdCtx&& ctx)
{
    auto& parser = ctx.parser;

    int64_t task_id;
    parser.add_option("id", task_id, "ID of the task to update")->required();

    task_tracker::TaskUpdate update;
    parser.add_option("--title", update.title, "New title of the task");
    parser.add_option("--category", update.category,
                      "New category of the task");

    std::string status;
    parser.add_option("--status", status,
                      "New status of the task (todo, in_progress, done)");

    CLI11_PARSE(parser, ctx.argc, ctx.argv);

    if (! parseStatus(update.status, status))
        return 1;

    task_id = taskTracker().update(task_id, update);

    if (task_id < 0) {
        std::cerr << "Task not found: " << task_id << std::endl;
        return 1;
    }

    auto task = taskTrackerView().get(task_id);
    if (! task) {
        std::println(std::cerr,
                     "Error: task with id {} was not found after updating",
                     task_id);
        return 1;
    }

    std::println("Updated task: ");
    printTask(*task);
    return 0;
}

int remove(SubCmdCtx&& ctx)
{
    auto& parser = ctx.parser;

    int64_t id;
    parser.add_option("id", id, "ID of the task to delete")->required();

    CLI11_PARSE(parser, ctx.argc, ctx.argv);

    if (taskTracker().remove(id) < 0) {
        std::println("No task found with ID: {}", id);
        return 1;
    } else {
        std::println("Deleted task with ID: {}", id);
        return 0;
    }
}

int report(SubCmdCtx&& ctx)
{
    auto& parser = ctx.parser;

    auto* by_category =
        parser.add_flag("--by-category", "Generate report by category");

    CLI11_PARSE(parser, ctx.argc, ctx.argv);

    if (*by_category)
        std::cout << "Generating report..." << std::endl;
    return 0;
}

int overdue(SubCmdCtx&& ctx)
{
    auto& parser = ctx.parser;

    CLI11_PARSE(parser, ctx.argc, ctx.argv);

    std::cout << "Listing overdue tasks..." << std::endl;
    return 0;
}

int search(SubCmdCtx&& ctx)
{
    auto& parser = ctx.parser;

    std::string keyword;
    parser.add_option("keyword", keyword, "Keyword to search for")->required();

    CLI11_PARSE(parser, ctx.argc, ctx.argv);

    std::cout << "Searching tasks for keyword: " << keyword << std::endl;
    return 0;
}

int tag(SubCmdCtx&& ctx)
{
    auto& parser = ctx.parser;

    int id;
    parser.add_option("id", id, "ID of the task to tag")->required();

    std::string tag;
    parser.add_option("tag", tag, "Tag to add to the task")->required();

    CLI11_PARSE(parser, ctx.argc, ctx.argv);

    std::cout << "Tagging task ID: " << id << " with tag: " << tag << std::endl;
    return 0;
}

int history(SubCmdCtx&& ctx)
{
    auto& parser = ctx.parser;

    int id;
    parser.add_option("id", id, "ID of the task to view history")->required();

    CLI11_PARSE(parser, ctx.argc, ctx.argv);

    taskTrackerView().history(id, printHistoryEntry);
    return 0;
}

} // namespace subcommands
} // namespace

int main(int argc, char* argv[])
{
    using namespace subcommands;

    constexpr Subcommand sub_cmds[] = {
        {"add", "Add a new task", add},
        {"list", "List tasks", list},
        {"update", "Update a task", update},
        {"delete", "Delete a task", remove},
        {"report", "Generate a report of tasks", report},
        {"overdue", "List overdue tasks", overdue},
        {"search", "Search tasks by keyword", search},
        {"tag", "Tag a task", tag},
        {"history", "View task history", history},
    };

    CLI::App app{"Task Tracker"};
    int ret = 0;

    for (auto&& sub_cmd : sub_cmds) {
        auto& [name, description, callback] = sub_cmd;
        auto sub_cmd_parser = app.add_subcommand(name, description);
        if (callback)
            sub_cmd_parser->callback(
                [&] { ret = runSubcommand(sub_cmd, argc, argv); });
    }

    if (argc < 2) {
        std::cerr << app.help() << std::endl;
        return 0;
    } else {
        CLI11_PARSE(app, 2, argv);
        return ret;
    }
}
