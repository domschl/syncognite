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
* Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn)
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [tensor-poet](https://github.com/domschl/tensor-poet) is an implementation of the same basic idea using Tensorflow.
