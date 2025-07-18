#pragma once

#include <istream>
#include <array>

#include <openssl/sha.h>

using CheckSum = std::array<unsigned char, SHA256_DIGEST_LENGTH>;

bool compute_checksum(std::istream& is, CheckSum& checksum);
bool compute_checksum(std::istream& is, std::streampos endpos, CheckSum& checksum);
