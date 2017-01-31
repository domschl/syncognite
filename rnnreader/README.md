# RNNs for text generation
## Requirements
Check the [main page](../../..) for build requirements.
## Build
After configuration of CMake, `rnnreader` can be built directly by:
```
make rnnreader
```
## Dataset
Use any (UTF-8) text-file, or `tiny-shakespeare.txt` from `datasets`.

## Training
From build directory:
```
rnnreader/rnnreader ../datasets/tiny-shakespeare.txt
```

## References
* [tensor-poet](https://github.com/domschl/tensor-poet) is an implementation of the same basic idea using Tensorflow.