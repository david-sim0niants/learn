#pragma once

#include "move_only.h"

#include <openssl/evp.h>

class UniqueEVP_PKEY {
public:
    UniqueEVP_PKEY() = default;

    explicit UniqueEVP_PKEY(EVP_PKEY* pkey) : pkey(pkey)
    {
    }

    UniqueEVP_PKEY(UniqueEVP_PKEY&&) = default;
    UniqueEVP_PKEY& operator=(UniqueEVP_PKEY&&) = default;

    ~UniqueEVP_PKEY()
    {
        EVP_PKEY_free(pkey);
    }

    inline EVP_PKEY* get() const noexcept
    {
        return pkey;
    }

    inline operator bool() const noexcept
    {
        return bool(pkey);
    }

private:
    MoveOnly<EVP_PKEY*> pkey;
};

UniqueEVP_PKEY generate_rsa_key(int nr_bits);

UniqueEVP_PKEY load_private_key(const char* fn);
UniqueEVP_PKEY load_private_key(FILE* f);

UniqueEVP_PKEY load_public_key(const char* fn);
UniqueEVP_PKEY load_public_key(FILE* f);
