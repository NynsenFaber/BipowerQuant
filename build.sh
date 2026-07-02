#!/bin/bash
set -e

echo "Starting build process..."
rm -f *.so
rm -f python/*.so  # Also clean the python directory
rm -rf build/
mkdir -p build
cd build

cmake .. 
make

cp compile_commands.json ..
# Output the compiled module directly into your python folder
cp bipower_core*.so ../python/
cd ..

echo "Build complete. Testing Python import..."
# Test it from within the python folder
cd python
python -c "import bipower_core; print('✅ bipower_core successfully built and imported!')"
cd ..