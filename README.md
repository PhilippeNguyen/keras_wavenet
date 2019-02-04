# keras_wavenet
Keras implementation of Wavenet (https://arxiv.org/abs/1609.03499).

Also includes an implementation of Fast/Queued Wavenet (https://arxiv.org/abs/1611.09482)

And an implementation of Parallel Wavenet (https://arxiv.org/abs/1711.10433).

BIG NOTE: The original Wavenet implementation is written in pure Keras (backend-agnostic); however, Fast/Queued Wavenet requires Tensorflow and Parallel Wavenet requires tensorflow/tensorflow_probability. (See Requirements section)

The Wavenet and Fast Wavenet implementations are based off the Nsynth implementations (https://github.com/tensorflow/magenta/tree/master/magenta/models/nsynth), but now are written using Keras layers instead of using pure tensorflow. I hope you find that this implementation is more flexible, easier to read, and easier to modify. I have some small modifications to the original model, which I'll list in a section below.

My Parallel Wavenet implementation reads in a trained Keras Wavenet model and uses this to train a Parallel/Student Wavenet. The Parallel Wavenet paper leaves out some details which I've filled in with some educated guesses, though there's no guarantees it's correct.

I've included the weight normalization optimizers, pulled directly from (https://github.com/openai/weightnorm/blob/master/keras/weightnorm.py)

Please let me know if you have questions or find mistakes.

#### Requirements
For original Wavenet:
 * keras
 * numpy
 * scipy
 * librosa

In addition to the above, for Fast/Queued Wavenet:
 * tensorflow

In addition to the above, for Parallel Wavenet:
 * tensorflow_probability

#### Usage

For training the original wavenet, use the build_wavenet.py script.
Hopefully I've organized the code well enough to make it simple to modify. The build_model function in the build_wavenet.py script will build the complete model.
 * keras_wavenet/layers/wavenet provides specific keras layers used in the Wavenet model.
 * keras_wavenet/model/wavenet provides some of the larger structures used in the wavenet model. For example, res_block builds the residual block core of the wavenet model. It's these functions/structures which you will want to look at/modify if you want to create a wavenet-like model for a different domain.

Once you have a trained wavenet, you can generate samples from it using the Fast/Queued Wavenet algorithm. Use the run_wavenet.py script
