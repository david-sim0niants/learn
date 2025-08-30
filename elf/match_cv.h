#pragma once

#include <type_traits>

template<typename From, typename To>
using MatchConst = std::conditional_t<std::is_const_v<From>, std::add_const_t<To>, To>;

template<typename From, typename To>
using MatchVolatile = std::conditional_t<std::is_volatile_v<From>, std::add_volatile_t<To>, To>;

template <typename From, typename To>
using MatchCV = MatchConst<From, MatchVolatile<From, To>>;
