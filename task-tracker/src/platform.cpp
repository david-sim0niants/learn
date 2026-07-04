#include "platform.hpp"

namespace task_tracker::platform {

std::filesystem::path getDataDirectory()
{
#ifdef _WIN32
	return std::filesystem::path(std::getenv("APPDATA")) / "task_tracker";
#elif __APPLE__
	return std::filesystem::path(std::getenv("HOME")) /
		   "Library/Application Support/task_tracker";
#else
	if (const char* xdg = std::getenv("XDG_DATA_HOME"))
		return std::filesystem::path(xdg) / "task_tracker";
	return std::filesystem::path(std::getenv("HOME")) /
		   ".local/share/task_tracker";
#endif
}

std::filesystem::path getEnsuredDataDirectory()
{
    auto path = getDataDirectory();
    std::filesystem::create_directory(path);
    return path;
}

} // namespace task_tracker::platform
