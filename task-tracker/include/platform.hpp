#pragma once

#include <cstdlib>
#include <filesystem>

namespace task_tracker::platform {

std::filesystem::path getDataDirectory();
std::filesystem::path getEnsuredDataDirectory();

constexpr std::filesystem::path getSharedDataDirectory()
{
    return TASK_TRACKER_SHARED_DATA_DIR;
}

} // namespace task_tracker::platform
