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
After about 250 epochs (three layers of LSTMs with 512 neurons, 90 time-steps):
```
output: hou wilt hore.

PRONTERO:
That when senveimadly.

Citizen:
Ay Will heaven, doly!

OXFORD:
Come, manar! law my holy gewere I, by your, ghat,
Well ligit of.

LADY ANNE: I tomble reword,
Than gore toghtser, simeshard! To wolk my love,
And leney all colven.
I'll seturnden ares.

LEOM:
Senat stay me agherl?

GLOUCESTER:
Undes, amad! most tall,
Whou yot I can your warsh, and down both thyseef the words,
End With withesp ground: where, in all.
Thyolat
Marr muret His man me, every more throwed.

DUCHESS OF YORK:
A, Goo. Dratheng Romeo pue thy preyon,
Father nouge little-fool'd thee away'd.

LEONTES:
Sther lasiles!
Whichors no all time. Son lows, this mercious, my lord,
Cleaged from my fave: let's cape!
O, in Pompot of wise.

CORIOLANUS:
Go, hels if I know your king,
And pastly; believed no? if that foel arms? I love cewourning it becomfory!
Some scurng, Lord Fatth: let'el pant thou llasted,
Bries on mintle goast! Come, Encall, to it?

Third Gentleman:
I crow their briogsed with strung-inday.
```
## References
* Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn)
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [tensor-poet](https://github.com/domschl/tensor-poet) is an implementation of the same basic idea using Tensorflow.
