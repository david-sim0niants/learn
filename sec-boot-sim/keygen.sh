#!/bin/bash

DEFAULT_PUBKEY=public.pem
DEFAULT_PRIKEY=private.pem

function keygen()
{
    local PUBKEY="${1:-$DEFAULT_PUBKEY}"
    local PRIKEY="${2:-$DEFAULT_PRIKEY}"

    openssl genpkey -algorithm RSA -out $PRIKEY -pkeyopt rsa_keygen_bits:2048
    openssl rsa -pubout -in $PRIKEY -out $PUBKEY
}

keygen $@
