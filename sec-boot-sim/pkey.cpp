#include "pkey.h"

#include <cstdio>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <openssl/rsa.h>

UniqueEVP_PKEY generate_rsa_key(int nr_bits)
{
    EVP_PKEY* pkey = nullptr;

    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_RSA, NULL);
    if (! ctx ||
        EVP_PKEY_keygen_init(ctx) <= 0 ||
        EVP_PKEY_CTX_set_rsa_keygen_bits(ctx, nr_bits) <= 0 ||
        EVP_PKEY_keygen(ctx, &pkey) <= 0
    ) {
        ERR_print_errors_fp(stderr);
        pkey = nullptr;
    }

    if (ctx)
        EVP_PKEY_CTX_free(ctx);

    return UniqueEVP_PKEY(pkey);
}

UniqueEVP_PKEY load_private_key(const char* fn)
{
    FILE* f = fopen(fn, "r");
    if (! f)
        return UniqueEVP_PKEY(nullptr);
    UniqueEVP_PKEY key = load_private_key(f);
    fclose(f);
    return std::move(key);
}

UniqueEVP_PKEY load_private_key(FILE* f)
{
    return UniqueEVP_PKEY(PEM_read_PrivateKey(f, nullptr, nullptr, nullptr));
}

UniqueEVP_PKEY load_public_key(const char* fn)
{
    FILE* f = fopen(fn, "r");
    if (! f)
        return UniqueEVP_PKEY(nullptr);
    UniqueEVP_PKEY key = load_public_key(f);
    fclose(f);
    return std::move(key);
}

UniqueEVP_PKEY load_public_key(FILE* f)
{
    return UniqueEVP_PKEY(PEM_read_PUBKEY(f, nullptr, nullptr, nullptr));
}
