#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string_view>
#include <vector>

#include <openssl/evp.h>

#include "checksum.h"
#include "pkey.h"

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
    std::fstream image {image_fn};
    if (! image)
        fail("could not open the image file");

    UniqueEVP_PKEY priv_key = load_private_key(priv_key_fn);
    if (! priv_key)
        fail("failed loading private key");

    sign(image, priv_key.get());
}

using Signature = std::vector<uint8_t>;

Signature make_signature(const CheckSum& checksum, EVP_PKEY* priv_key);

void sign(std::iostream& image, EVP_PKEY* priv_key)
{
    CheckSum checksum;
    if (! compute_checksum(image, checksum))
        fail("failed computing checksum");

    Signature signature = make_signature(checksum, priv_key);
    // TODO
}

Signature make_signature(const CheckSum& checksum, EVP_PKEY* priv_key)
{
    // TODO
    return {};
}
