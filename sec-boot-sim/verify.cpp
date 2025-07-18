#include <iostream>
#include <fstream>
#include <algorithm>

#include <openssl/evp.h>
#include <openssl/err.h>

#include "checksum.h"
#include "pkey.h"
#include "signature.h"

const char *self = nullptr;

inline void usage()
{
    std::cerr << "Usage: " << self << " <IMAGE-FN> <PUBLIC-KEY-FN>\n";
}

inline void fail(std::string_view msg, int code=EXIT_FAILURE)
{
    std::cerr << "Error: " << msg << std::endl;
    std::exit(code);
}

enum class VerifyResult {
    Verified,
    Unverified,
    NoSignature,
    MalformedSignature,
};

void print_verify_result(VerifyResult result);
VerifyResult verify(const char* image_fn, const char* pubkey_fn);

int main(int argc, char* argv[])
{
    self = argv[0];

    switch (argc) {
    case 0: case 1:
        usage();
        fail("image file was not provided");
    case 2:
        usage();
        fail("public key file was not provided");
    case 3:
        print_verify_result(verify(argv[1], argv[2]));
        break;
    default:
        usage();
        fail("excess arguments");
    }

    return EXIT_SUCCESS;
}

void print_verify_result(VerifyResult result)
{
    switch (result) {
    case VerifyResult::Verified:
        std::cout << "Verified";
        break;
    case VerifyResult::Unverified:
        std::cout << "Unverified";
        break;
    case VerifyResult::NoSignature:
        std::cout << "No signature found";
        break;
    case VerifyResult::MalformedSignature:
        std::cout << "Malformed signature found";
        break;
    }
    std::cout << std::endl;
}

VerifyResult verify(std::ifstream& image, EVP_PKEY* pub_key);

VerifyResult verify(const char* image_fn, const char* pubkey_fn)
{
    std::ifstream image {image_fn, std::ios::in | std::ios::binary};
    if (! image.is_open())
        fail("could not open the image file");

    UniqueEVP_PKEY pub_key = load_public_key(pubkey_fn);
    if (! pub_key)
        fail("failed loading public key");

    return verify(image, pub_key.get());
}

enum class RetrieveSignatureResult {
    Found,
    Missing,
    Malformed
};

RetrieveSignatureResult retrieve_signature(std::istream& image, Signature& signature);

bool verify(const CheckSum& checksum, const Signature& signature, EVP_PKEY* pub_key);

VerifyResult verify(std::ifstream& image, EVP_PKEY* pub_key)
{
    Signature signature;
    switch (retrieve_signature(image, signature)) {
    case RetrieveSignatureResult::Missing:
        return VerifyResult::NoSignature;
    case RetrieveSignatureResult::Malformed:
        return VerifyResult::MalformedSignature;
    default:
        break;
    }

    image.seekg(-static_cast<std::streamoff>(signature.size()) - std::streamoff(8), std::ios::end);
    std::streampos endpos = image.tellg();
    image.seekg(0);
    CheckSum checksum;
    if (! compute_checksum(image, endpos, checksum))
        fail("failed computing checksum");

    return verify(checksum, signature, pub_key) ? VerifyResult::Verified : VerifyResult::Unverified;
}

RetrieveSignatureResult retrieve_signature(std::istream& image, Signature& signature)
{
    image.seekg(-8, std::ios::end);
    if (image.fail())
        return RetrieveSignatureResult::Missing;

    char footer[8];
    image.read(footer, sizeof(footer) / sizeof(footer[0]));

    if (image.gcount() < 8 || ! std::equal(footer + 4, footer + 8, "SIGN"))
        return RetrieveSignatureResult::Missing;

    uint32_t siglen = (footer[0] << 0) | (footer[1] << 8) | (footer[2] << 16) | (footer[3] << 24);
    if (image.tellg() < 8 + siglen || siglen > MAX_SIGLEN)
        return RetrieveSignatureResult::Malformed;

    image.seekg(image.tellg() - static_cast<std::streamoff>(siglen) - std::streamoff(8));

    signature.resize(siglen);
    image.read(reinterpret_cast<char*>(signature.data()), signature.size());

    return RetrieveSignatureResult::Found;
}

bool verify(const CheckSum& checksum, const Signature& signature, EVP_PKEY* pub_key)
{
    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new(pub_key, nullptr);
    if (! ctx)
        return false;

    if (EVP_PKEY_verify_init(ctx) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        return false;
    }

    int ret = EVP_PKEY_verify(ctx, signature.data(), signature.size(),
                                   checksum.data(), checksum.size());
    if (ret < 0) {
        char err_buf[256];
        ERR_error_string_n(ERR_get_error(), err_buf, sizeof(err_buf));
        fail(err_buf);
    }

    EVP_PKEY_CTX_free(ctx);

    return ret != 0;
}
