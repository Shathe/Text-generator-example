# Text-generator-example

This code is based on a tutorial posted by Jason Brownlee [here](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/).

This text generator is based on the idea of learning the next (the most probable) character from a sequence of (the previous) characters.


### Example: 


Given the sequence of "Hi, My name is Julian, What's your nam"

What's the most probable character? **'e'**, right?



There are other approaches in text generation like trying to learn the next word from a sequence of words (insted of characters). This codification and problem is Word2Vec, which is a more uitlized approach. [This is an explaining blog/tutorial of Word2Vec](http://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/).
In stead of one-hot encoding the text, see (this tensorflow )example[https://www.tensorflow.org/tutorials/keras/basic_text_classification] where they use a embedding layer.

### Run code
In order to train the model with a specific [text](./text.txt) just run:
```
python train.py
```

In order to generate some text, just run:
```
python test.py
```
