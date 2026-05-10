#!/bin/bash

if [ -d "./out" ]; then
    echo "Cleaning ./out directory..."
    rm -rf ./out
fi
mkdir -p ./out

if [ -d "./tests" ]; then
    echo "Cleaning ./tests directory..."
    rm -rf ./tests
fi
mkdir -p ./tests
