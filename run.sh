set -e # exits if a command fails

cmake -B build -G Ninja
cmake --build build/

./build/bin/main