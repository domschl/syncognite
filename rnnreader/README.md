# LSTMs for text generation
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
## Output
After about 250 epochs (three layers of LSTMs with 512 neurons, 90 time-steps), about 20hrs on a recent desktop CPU (8 threads), with `tiny-shakespeare.txt` from [`datasets`](../datasets/) as training text:
```
PETRUCHIO:
Procee! Come ither;
Our heart-stoods a chirt you there's merry,
Distlestimagity,
Though this his magegether forth's a point
That gove him reture the tribunes from my veawing
His was abold heares,
Sirray, Comprot thee, far
Thas crown his offtimu, doth from me destenther flous.

MARCIUS:
So, my resolute! Join, giencely, taking.

PETRUCHIO:
Why hath Shall I propes at to fear
The wellien and my people:' and bots, while you made
he duty entructions are of the roor
what my commands
thy groed,
Our antentictie!
And say has peeson kiesd, and freth
as you trumpeed? Look Porceith sortes?

HAROLIO:
So dear the king;
And let us heaving? to pluck werd I happy
From my compalinn; by the constage
And fight to visit as three on: bus certain
The lukening plaines mind othermine of my wore.
```
## References
* Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn)
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [tensor-poet](https://github.com/domschl/tensor-poet) is an implementation of the same basic idea using Tensorflow.
