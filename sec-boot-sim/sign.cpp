#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string_view>

#include <openssl/evp.h>

#include "checksum.h"
#include "pkey.h"
#include "signature.h"

const char* self = nullptr;

inline void usage()
{
    std::cerr << "Usage: " << self << " <IMAGE-FN> <PRIVATE-KEY-FN>\n";
}

inline void fail(std::string_view msg, int code=EXIT_FAILURE)
{
    std::cerr << "Error: " << msg << std::endl;
    std::exit(code);
}

void sign(const char* image_fn, const char* priv_key_fn);

int main(int argc, char* argv[])
{
    self = argv[0];

    switch (argc) {
    case 0: case 1:
        usage();
        fail("image file was not provided");
    case 2:
        usage();
        fail("private key file was not provided");
    case 3:
        sign(argv[1], argv[2]);
        break;
    default:
        usage();
        fail("excess arguments");
    }

    return EXIT_SUCCESS;
}

void sign(std::iostream& image, EVP_PKEY* priv_key);

void sign(const char* image_fn, const char* priv_key_fn)
{
    std::fstream image {image_fn, std::ios::in | std::ios::out | std::ios::binary};
    if (! image.is_open())
        fail("could not open the image file");

    UniqueEVP_PKEY priv_key = load_private_key(priv_key_fn);
    if (! priv_key)
        fail("failed loading private key");

    sign(image, priv_key.get());
}

Signature make_signature(const CheckSum& checksum, EVP_PKEY* priv_key);
void put_sign(std::ostream& image, const Signature& sig);

void sign(std::iostream& image, EVP_PKEY* priv_key)
{
    CheckSum checksum;
    if (! compute_checksum(image, checksum))
        fail("failed computing checksum");

    Signature sig = make_signature(checksum, priv_key);
    if (sig.empty())
        fail("failed making signature");

    put_sign(image, sig);
    image.flush();
}

Signature make_signature(const CheckSum& checksum, EVP_PKEY* priv_key)
{
    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new(priv_key, nullptr);
    if (! ctx)
        return {};

    size_t siglen;
    Signature sig;

    if (EVP_PKEY_sign_init(ctx) <= 0 ||

        EVP_PKEY_sign(ctx, nullptr, &siglen, checksum.data(), checksum.size()) <= 0 ||

        (sig.resize(siglen),
         EVP_PKEY_sign(ctx, sig.data(), &siglen, checksum.data(), checksum.size()) <= 0)
    )
        sig = {};

    EVP_PKEY_CTX_free(ctx);
    return sig;
}

void put_sign(std::ostream& image, const Signature& sig)
{
    image.seekp(0, std::ios::end);
    image.write(reinterpret_cast<const char*>(sig.data()), sig.size());

    uint32_t siglen = sig.size();

    char footer[] = {
        char((siglen >> 0) & 0xFF),
        char((siglen >> 8) & 0xFF),
        char((siglen >> 16) & 0xFF),
        char((siglen >> 24) & 0xFF),
        'S', 'I', 'G', 'N'
    };

    image.write(footer, sizeof(footer) / sizeof(footer[0]));
}
