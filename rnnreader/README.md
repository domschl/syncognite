# RNN and LSTM for text generation

## Requirements

Check the [main page](../../..) for build requirements.

## Build

After configuration of CMake, `rnnreader` can be built directly by:

Note: M1 users should use `ccmake` to configure `USE_SYSTEM_BLAS` to `ON`. This uses the M1 hardware accelerators for matrix multiplication and leads to 3x-6x speedup.

```bash
make rnnreader
```

## Dataset

`rnnreader` can be trained with any UTF-8 encoded text file. A sample (about a 1/4 of William Shakespeares works) is
available at `datasets/tiny-shakespeare.txt`.

Additionally, longer training text can be downloaded:

### Shakespeare's complete works

To download the Complete Collected Works of William Shakespeare (5x the size of `tiny_shakespeare.txt`) from Project Gutenberg, use:

In directory `datasets`:

```bash
pip install -U ml-indie-tools
python download_shakespeare.py
```

The resulting file `shakespeare.txt`, containing the complete collected works, can be use instead of `tiny-shakespeare.txt`.

This uses [`ml-indie-tools`](https://github.com/domschl/ml-indie-tools) to download the Complete Works from Project Gutenberg.
The library can be used to download arbitrary book-collections from Project Gutenberg, see [Documentation](https://github.com/domschl/ml-indie-tools#gutenberg_dataset).

### Women-writers

To download a collecting of writing by Emilie Brontë, Jane Austen, and Virginia Woolf from Project Gutenberg, use:

In directory `datasets`:

```bash
pip install -U ml-indie-tools
python download_women_writers.py
```

Note: have a look at the download script, it can be easily modified for other authors, subjects or collections.

The resulting file `women_writers.txt`, which contains all book texts (about 12MB) concatenated, can be use instead of `tiny-shakespeare.txt`.

## Training

From build directory:

```bash
rnnreader/rnnreader ../datasets/tiny-shakespeare.txt
```

or, if you have downloaded the full Works:

```bash
rnnreader/rnnreader ../datasets/shakespeare.txt
```

or,

```bash
rnnreader/rnnreader ../datasets/women_writers.txt
```
![lstm training](https://github.com/domschl/syncognite/blob/main/doc/images/rnnreader-graph.png)

_6-layer LSTM with 2x 384 units trained on Complete Shakespeare_

![lstm training](https://github.com/domschl/syncognite/blob/main/doc/images/rnnreader-graph-2.png)

_4-layer LSTM with 2x 256 units trained on women_writers

## Output

### LSTM network, Shakespeare Complete Works

Training of a 6-layer LSTM network with 2x384 hidden units. The generator uses different 'temperatures' during generation:

```markdown
ROMEO:
Give me love the earth.
The king another joy, for myself and privoti
---- Temperature: 1.4000 ------------------
on.

Second Servingman:
Why, what, a pristes on my daughtess, I have found
The world frown me with my half of this compon himself
And chem their great ear former of your office,
If he had not say amends must be salds and wears
Hath right ado so loves them be for a minr.
What is the decirest thon say
---- Temperature: 1.7000 ------------------
 I sent thee fea.

ANGELO:
How now the gardiressh are sonrow or twoihing, are you to go,
If any take my less shall be sof an longer.

All:
It is all this answer man on his hasting.

LEONTES:
How?

MENENIUS:
If fore, my liege, I have no more:
Yet a mortal purity, and thy better shouldst have four
Twa

Training net: data-size: 15443, chunks: 242, batch_size: 64, threads: 8 (bz*ch): 15488
Ep: 181, Time: 128s, (3s test) loss:0.8329 err(val):0.4693 acc(val):0.5307
Ep: 182, Time: 127s, (2s test) loss:0.8303 err(val):0.4516 acc(val):0.5484
Ep: 183, Time: 128s, (3s test) loss:0.7876 err(val):0.4559 acc(val):0.5441
Ep: 184, Time: 129s, (2s test) loss:0.7416 err(val):0.4484 acc(val):0.5516
Ep: 185, Time: 128s, (2s test) loss:0.7247 err(val):0.4488 acc(val):0.5512
---- Temperature: 0.8000 ------------------
d, for which right for it should be, none; I call home,
To delives to hear a betigod: whoselger a show it shearerc,
But satisfets unsroibs when rikosaasly weary,
Thou last more rough closet. The kindress from themelt
At on thy bib time us both,
That will were mets on a suin a princiosine,
Lest them
---- Temperature: 1.0000 ------------------
fuilt or us of Froslaim I siy. Come, I spad with him that would unceem with rageous sould.

ANGELO:
I do lord, that within with us but boots,
She did not a plot.
Make! I will be mind or his mother's royaltie.
Ar, both yoursae-aulled quarrel would acqulnelding of my lats.

KING RICHARD III:
But grest
---- Temperature: 1.2000 ------------------
 who, this sensh, of a feold duke,
The god-made out of the princes.

NORTHUMBF:
As he is fair! O heaven, the neckss of his
but of that rall Gancaster,
Voanta-us supple in the pals, that was but friend:
'lfown them to long; when it with a teel with me all to
go: what a chawbering of our cans,
It blao
---- Temperature: 1.4000 ------------------
ns and bed on him that makes murders
Of that pain our hatr'd ruin to will no more of his
un doch with such pistl-fair, or care,
That will bring thee, if about his swings should about the thorn
That murdering it the mother of the nuss,
An old hanging of safe. Whos, it was worth they will.

TRANIO:
Of
---- Temperature: 1.7000 ------------------
 Mistress in thine! this more than word 'tis in hopes
and a hus-lain of the faces of heart,
That made a gold of his nature, be not a back of doors!
Whose her bring much in the curse of love
As we of such rings are slifford.

LUCENTIO:
Here in our thunds.

ANTONIO:
What should call their bastest part

Training net: data-size: 15443, chunks: 242, batch_size: 64, threads: 8 (bz*ch): 15488
Ep: 186, Time: 127s, (3s test) loss:0.7238 err(val):0.4476 acc(val):0.5524
Ep: 187, Time: 127s, (3s test) loss:0.7325 err(val):0.4558 acc(val):0.5442
Ep: 188, Time: 126s, (2s test) loss:0.7373 err(val):0.4584 acc(val):0.5416
Ep: 189, Time: 126s, (2s test) loss:0.7296 err(val):0.4511 acc(val):0.5489
Ep: 190, Time: 126s, (3s test) loss:0.7187 err(val):0.4579 acc(val):0.5421
---- Temperature: 0.8000 ------------------
 with me;
For like men: they are never joy
shall have behold ot with us.

BRUTUS:
Let's what.

MENENIUS:
Very be says, hear her;' and is is quietied.

KATHARINA:
No most right will.

COMINIUS:
On Secrevent.

JULIET:
Give me poor in Ebwilk, thou uncir'd with
When it stanpcul, on our slack; in your
ni
---- Temperature: 1.0000 ------------------
ventation of the king with our tenden
Wedt now jomininion.

FRIAR LAURENCE:
This councelticks to many action:
For she is in secret sederater.

MINTAP:
He is every pity. I would thou will not not
ut, if thou so oft non enough to to
whos shamon cheetiness. Ty rocurees is
comence me as chill. O trouble
---- Temperature: 1.2000 ------------------
s before the house,
Discent, how I am Lort Romerlo, welcome, we have scopt the
sporiched to reason their truth,
To sword in selming to this well-armed.
Buth the queen, do proppuring him.
But send is conscasition: come.

HENRY BOLINGBROKE:
Well, let me see thy weakon whather?

MENENIUS:
Live me a. Rn
---- Temperature: 1.4000 ------------------
lis to his brother?

GLOUCESTER:
The duke? Hav, it in thy dispatch,
No quarrellal and to accused in me, and my poor obey;
His present pastion water abong.
well, no more will come again.

First Servingman:
He will not yet live, O most wear a
foreword's death, he saw wont
Against my children: he comes
---- Temperature: 1.7000 ------------------
 above himself
as I as as a man' mine eyes
To grace it ence, in this rine, sir,
He pot the samipy of him o'er his son,
That will not be our countenances: to standly,
Straigor's pinest of his tiith of him.

FRIAR LAURENCE:
Stoppes the poting.

ANGELO:
Now, when She be remember you to send in wont.
```

### Output, women_writers

```markdown
Training net: data-size: 90000, chunks: 1407, batch_size: 64, threads: 8 (bz*ch): 90048
Ep: 166, Time: 223s, (7s test) loss:1.0310 err(val):0.3929 acc(val):0.6071
Ep: 167, Time: 226s, (7s test) loss:1.0282 err(val):0.3945 acc(val):0.6055
Ep: 168, Time: 227s, (7s test) loss:1.0265 err(val):0.3935 acc(val):0.6065
Ep: 169, Time: 226s, (7s test) loss:1.0246 err(val):0.3942 acc(val):0.6058
Ep: 170, Time: 226s, (7s test) loss:1.0233 err(val):0.3938 acc(val):0.6062
---- Temperature: 0.8000 ------------------
iam.”

On, ‘_Smiless, who owerwregd her; only that she is got but the regret support or quittyed women, 
on his way and then supplantation before one's getting Eleanor's spirits
Hall, veise engaged; Katharine neverther intelless for one accus to make her quite so
earnestness of who so asknesses
---- Temperature: 1.0000 ------------------
failed for several young friend, and then made her stupidity as they were but spoken of luss to the competicurras between 
them could myself in faults in Margaret’s requestic stabbelists. But this was one voices, connected with sake and freedy 
one of these owner, and fond of Mr. Rushworth and Susan,
---- Temperature: 1.2000 ------------------
 she would not make up her mind to believe life or the mail.

Rachel devoors reported a sunstitute and adviser, he did not think how I’ve had it to her more good thing 
of the old Shartomber-would be kind. Why I wanted me, borked with what might be supported.

They were as varied by the guardin
---- Temperature: 1.4000 ------------------
g fitter to speak from the fortnight the door of the house and value by themselves of a quarter of an hour 
than before the book in a pleasing compliment to Mansfield, with no way that she was evidently angry, 
and she still sunk. She had an invitation of her chance which she had never been welcome, a
---- Temperature: 1.7000 ------------------
nd in her sister’s begs ten minutes, and as they were as she disposed to
bring an anxiety of the room, and a great deal of one of her son and anxious with an anger, the lady 
superintended her to see her to make her spirits, had been at all to animald, he would not deserve him of a 
prints agitation w

Training net: data-size: 90000, chunks: 1407, batch_size: 64, threads: 8 (bz*ch): 90048
Ep: 171, Time: 223s, (7s test) loss:1.0225 err(val):0.3929 acc(val):0.6071
Ep: 172, Time: 225s, (7s test) loss:1.0223 err(val):0.3943 acc(val):0.6057
Ep: 173, Time: 226s, (7s test) loss:1.0222 err(val):0.3949 acc(val):0.6051
Ep: 174, Time: 226s, (7s test) loss:1.0226 err(val):0.3950 acc(val):0.6050
Ep: 175, Time: 227s, (7s test) loss:1.0252 err(val):0.3956 acc(val):0.6044
---- Temperature: 0.8000 ------------------
er of having recombing well satisfied, for on separating spirits easily; he percepting it; 
and trught completing adopped over Emma's art slowly; amongst attending their work to him now; 
at that subject, the sweetness of his; and whether the ancient walk was evidently
before bread, softed back once;
---- Temperature: 1.0000 ------------------
and they could speak with the flowing mania with the corner which he had thus remained that Ralph and Emma, 
on a charm, which, her great houses feared she went on accord to be on the fingers’ morpoper; 
it would be so, but her staying was comfortably refused to be secured, gazing a little purth again
---- Temperature: 1.2000 ------------------
. He had a first company. Everybody can so please.  No long sentencting a cross creature she is excellent. 
We were excellent in the Temple away, but you looked ill.

She should be carried to Catherine began so near them soon that we see the difference, 
and does not, but she is losing me!"

"Th
---- Temperature: 1.4000 ------------------
en you are engaged, in the argument of dancing when the worst of the fire down they may come 
in them where nothing to tell you that she would not expect a moment to comply a sort of warmth 
than before, and could tell him the whole of his mother, whom they should have life was still a table. 
Her fris
---- Temperature: 1.7000 ------------------
h sometimes thought her felicity in the world when the streets of the words which Mrs. Norris did not 
perfectly do no spride-son and three years ago, and what was in front day.

But the spots of the street as a really way was gone, and which might say, and she was gone, 
when the extent to the whole

Training net: data-size: 90000, chunks: 1407, batch_size: 64, threads: 8 (bz*ch): 90048
Ep: 176, Time: 223s, (7s test) loss:1.0276 err(val):0.3958 acc(val):0.6042
Ep: 177, Time: 226s, (7s test) loss:1.0323 err(val):0.3964 acc(val):0.6036
```

## References

* Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn)
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [tensor-poet](https://github.com/domschl/tensor-poet) is an implementation of the same basic idea using Tensorflow.
