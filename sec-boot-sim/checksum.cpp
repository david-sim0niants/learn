#include "checksum.h"

#include "openssl/evp.h"

#include <cstdio>

bool compute_checksum(std::istream& is, CheckSum& checksum)
{
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (! ctx)
        return false;

    bool ok_so_far = true;
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr)) {
        char buf[BUFSIZ];

        while (! is.eof()) {
            is.read(buf, BUFSIZ);
            if (is.fail() && ! is.eof() || ! EVP_DigestUpdate(ctx, buf, is.gcount())) {
                ok_so_far = false;
                break;
            }
        }

        ok_so_far = ok_so_far && EVP_DigestFinal_ex(ctx, checksum.data(), nullptr);

        if (ok_so_far)
            is.clear();
    }

    EVP_MD_CTX_free(ctx);
    return ok_so_far;
}
