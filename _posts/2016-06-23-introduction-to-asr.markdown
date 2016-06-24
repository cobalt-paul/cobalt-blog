---
layout: post
title:  "Introduction to Automatic Speech Recognition"
date:   2016-06-23 15:46:32 +0200
categories: jekyll update
---

The earliest example of a computer performing speech to text was a 1952 system built by Bell Labs which could recognize the numbers one to ten, for a single speaker. [0] Since then, we've come a long way: well more than a billion people on the planet have a high-quality large vocabulary speech recognition system in their pockets. Simple speech recognition systems power a variety of modern applications, notably phone support systems, but has been inaccessible to companies who couldn't staff and maintain a large speech research lab. But over the last five years, advances in speech recognition have made a staggering variety of novel application commercially feasible. The goal of this blog post is to introduce you to the vocabulary, concepts and challenges of speech recognition, and provide you with pointers on how to  find out more.



# Automatic Speech Recognition
1. [How the Problem is Framed](#problemstatement)
2. [Decomposition into Language and Acoustic Model](#amlm)
3. [Pronunciation and the Lexicon](#lex)
4. [How Acoustic Models work](#am)
5. [How Language Models work](#lm)
6. [Finite State Transducers, Hidden Markov Models, and Decoding](#fsthmm)
7. [Connectionist Temporal Classification, LSTM-RNNs](#ctclstm)
8. [Other Problems](#other)
9. [Further Reading](#furtherreading)

## <a name='problemstatement'></a>How the Problem is Framed

At its core, speech recognition takes audio as input and produces text as output. As a problem in statistical modeling, we frame the problem as: $$ argmax_{w} P(W \| A) $$, or "find the sentence of words (w) with the highest probability of having been spoken, given the audio that we heard." If this sentence is confusing, you may want to review some of the basics of probability, e.g., [here](https://www.khanacademy.org/math/probability/independent-dependent-probability).

The rest of this blog post relies on a basic familiarity with the concepts of machine learning. Specifically, you should know what 'features' are, and understand the basics of classification ( a model that takes features as input and predicts a label as output). If you are interested in a deeper introduction to machine learning, you may wish to look [here](TODO).

### 'Audio'

Digital audio is most commonly a series of scalar integer values called 'samples'. A typical microphone will measure the intensity of air pressure against the microphone N times per second, where 'N' is called the sample rate; if the sample rate is 16 kilohertz, then you will have 16,000 samples per second. Image 1 shows a typical 'waveform' of digital audio. Image 2 shows that as you zoom in, you see that the audio looks like a wave. In Image 3, we've zoomed in even further, and can see that in fact the audio is a series of 'points' which we then interpolate between, to give the illusion of a continuous measurement.

![Image One]({{ site.url }}/images/audio_one.png)
![Image Two]({{ site.url }}/images/audio_two.png)
![Image Three]({{ site.url }}/images/audio_three.png)

### 'Text'


### Terms

It's worth briefly noting the important specialized terms that we will be using later on. These definitions are simplified and sometimes contain outright lies (I will attempt to note when I have simplified something); they are intended to provide a conceptual handle on what's being done.


* acoustic model
A model which takes audio as input, and gives a probability distribution over phonemes as output.
* phoneme
A phoneme is a fundamental unit of speech production. When you say "cat", it is pronounced (in Arpabet) K AE T. K, AE and T are phonemes. 
* phone
An actual example of someone saying a phoneme.
* lexicon
Speech relies on a (mostly) hand-curated list of pronunciations called a "lexicon". The lexicon will contain hundreds of thousands of words and their phonetic transcription.
* language model
A model which takes a sentence as input and gives a likelihood of observing that sentence. The sentence "the cat in the hat" is much more likely than "cat the in hat the", and a good language model will assign the first sentence a (much) higher likelihood.




## <a name='amlm'></a>Decomposition into Language and Acoustic Model

The framing $$ argmax_{w} P(W \| A) $$ is nice, but challenging to model properly. It also doesn't allow us to use written text (e.g., from books or the internet) to improve our accuracy. However, by reframing the problem as $$ argmax_{w} P(W) P(A \| W) $$, or "find the sentence for which the audio is most likely". In speech recognition terms, we have now decomposed the problem into a language model ( $$ P(W) $$ )  and an acoustic model. This is great for a number of reasons. The statistical modeling problem $$ P(A \| W) $$ is much less sparse, so you can train more accurate models with less data. The language model can be trained on huge volumes of raw text (e.g., all books ever written, the internet, or data from the specific domain that you care about), allowing us to improve our accuracy by adding a statistical prior based on our knowledge of what words are commonly said in what order and what we expect you to be talking about.

This explanation is heavily simplified, and diving into the details will make many of the assertions make much more sense.


## <a name='am'></a>How Acoustic Models work

An acoustic model works roughly like this:
* the audio is cut up into 10ms chunks called frames
* the frame is transformed, using signal processing, into a condensed set of features (a typical feature set might be 40 mel-frequency cepstral coefficients, with 7 frames of context on each side)
* the features are then fed into the acoustic model (typically a deep neural network), which outputs the likelihoods of each phoneme.

There are some important simplifications in this explanation. First, there are other approaches, most notably "connectionist temporal classification" approaches, which are currently transition from research into industry. A later section will discuss this approach. Second, the output targets of a typical feed-forward deep neural network acoustic model are actually context-dependent targets in a phonetic tree, but conceptually it's reasonable to just thing of the model as producing phone likelihoods.


### Acoustic Model training

Acoustic model training, conceptually, is straight-forward to explain.

Given a training set of transcribed audio, you use a bootstrap acoustic model to run "forced-alignment". Where you get the bootstrap model is a topic for another blog post. Aligning your audio will result in a phone label for every 10 ms frame in your training set. These labeled frames are then fed to your neural network for training. The specifics of this training process involve an extraordinary amount of dark art, complexity, and careful tuning.

### Concerns, Trade-Offs, and Caveats

The data used to train your acoustic model has an enormous impact on how your model performs in different conditions. If you train your model exclusively on male speakers in their early twenties, you build models which perform poorly on female speakers and young or old speakers. If you train your model on a noise cancelling close-talk microphone, you will perform poorly on far-field microphones, cell phone microphones, or regular close-talk microphones. If you train your model on data collected in a quiet environment by a sitting speaker, your model will strugle in noisy environments or speakers who are running. 

Acoustic models are big, and they are computationally expensive. An acoustic model that will run on a large server can be more accurate than a model which needs to be small enough to run on your cell phone without killing the battery.

Training acoustic models can take a long time. Depending on how much data you have, this can be anywhere from hours to weeks.

Training acoustic models requires significant amounts of data. Hundreds of hours are typically enough to train an acceptable but not great model from scratch. Dozens of hours are typically enough to adapt a good acoustic model to a new domain (new noise conditions, new speaker characteristics, new microphone type).


## <a name='lm'></a>How Language Models work

Language models are often the biggest factor in determining whether a particular application works welll or terribly. If you want to build a system to talk about music, and you use a language model trained on eighteenth century romance novels, the output will be funny but totally useless.

Modern language models take a large volume of text and train a generative model. For every sentence, the model will produce a likelihood score. This score can then be used to say that one sentence is  more likely than another. A simple approach is N-gram language models. In 3-gram language modeling, for example, we count every 3 word sequence in our training data. The probability of any three word sequence is then the frequency of that sequence divided by the number of sequences in our training data. (This is an extreme simplification; more depth will be provided in a future blog post.) The probability of a candidate sentence is then the product of the probability of every 3 word sequence in our sentence. If the sentence is "the cat in the hat", the probability is: P("the" \| start-of-sentence")*P("cat" \| "the", start-of-sentence)*P("in"\|"cat", "the")*P("the"\|"in", "cat")*P("hat"\|"the", "in").

This explanation leaves out a number of important topics: handling of "unseen" word sequences; langauge model smoothing; interpolating multiple language models; more complicated language model approaches like recurrent neural network language models, and many other details.
## <a name='lex'></a>Pronunciation and the Lexicon

The lexicon forms an essential but awkward part of a speech recognizer. Speech recognition systems can only recognize words for which the pronunciation is known. These pronunciations are collected in the lexicon, a map from word to pronunciation. The pronunciations are sequences of phonemes. For example, when you say "cat", it is pronounced (in Arpabet) K AE T.

The lexicon bridges the gap from the acoustic model to the language model. We could, hypothetically, build an acoustic model trained to output words, but it would require significantly more data (because you'd need many examples of every word in your vocabulary) and adding words to the vocabulary would require training new acoustic models. By using a lexicon, our acoustic model can produce phonemes and our lexicon can turn sequences of phonemes into words.

## <a name='fsthmm'></a>Decoding and Searching

The "score" of a candidate sentence is: the language model probability of that sentence times the acoustic probability of that sentence. (Going from a sequence of per-frame phone likelihoods to a single acoustic probability of the sentence is outside the scope of this blog post.) 

A naive way to find the most likely sentence is to calculate a score for every possible sentence. If we supposed that the largest possible sentence is ten words, and our vocabulary is 100,000 words, this would require evaluating 1e50 possible sentences (assuming each word has only one pronunciation). Clearly this is not going to be fast enough to be run in real-time (or finish before the sun burns out), so we need to design a clever approach to searching for the most likely sentence.

TODO: insert a discussion of WFSTs, HMMs, viterbi search, beam pruning


## <a name='ctclstm'></a>Connectionist Temporal Classification, LSTM-RNNs
Do I really need to cover this?
## <a name='other'></a>Other Problems
## <a name='furtherreading'></a>Further Reading


[0] http://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/354_LALI-ASRHistory-final-10-8.pdf
