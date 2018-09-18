A Deep Learning way of resolving coreference

implementation of paper 1606.01323v2 (stanford)

According to the paper, I generate each pair of mentions into a 5000+ dimentional vector and feed
the data into a simple feed-forward-structured neural network, which first encodes the input vectors
into 500-d vectors, then scores them indicating how likely it's the best coreference.

I modify (or simplify) the paper to fit our goal: only find coreferences between pronouns and entites, in other word,
only score pairs of (entity, pronoun) instead of clusterring all mentions which are coreferent, 
in other word, ignoring pairs like (entity, entity),
e.g Kay ate an apple in Melbourne, he felt full. he-->Kay

The training data with correct coreferent words are manually labelled, I actually implemented a web page
to improve efficiency of labelling, see the next repository. Also, data stucture maybe not quite
understandable by just looking at my code or txt files, but anyway, guess no one would directly
use this.I might be doing a more flexible version to serve as api for general usage. By then,
others would be able to get the best coreferent word by simply inputing the sentence and pronoun.

Btw, I also changed the source code of Polyglot to do a bit of tricks, it's not included here in the repository.

Tested for Chinese.English shud be working but not tested.


still improving the loss function, (since I changed the model)

Feature engineering & initial overfitting neural network by Jingyuan Zhang, optimized neural network (prevent overfitting) by seanxu, at Kalamodo, Suzhou

(And this is probably not counted as deep learning...more of a feature eng project + a simple neural network)
