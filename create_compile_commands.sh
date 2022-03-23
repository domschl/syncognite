# This create a compile_commands.json file in the build directory. There is already a link
# to this file in the root directory.

BUILD_DIR=build

if [[ -d "$BUILD_DIR" ]]; then
    cd $BUILD_DIR
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=YES ..
    cp compile_commands.json ..
    echo "Compile commands exported to $BUILD_DIR/compile_commands.json and copied to root directory"
else
    echo "No build directory <$BUILD_DIR> found"
fi
