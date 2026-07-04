#pragma once

#include <cstdlib>
#include <filesystem>

namespace task_tracker::platform {

std::filesystem::path getDataDirectory();
std::filesystem::path getEnsuredDataDirectory();

} // namespace task_tracker::platform
