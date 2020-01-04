# Rectified-diffGrad-tf

This is a mix of the diffGrad optimizer from the paper [diffGrad: An Optimization Method for 
Convolutional Neural Networks](https://arxiv.org/abs/1909.11015) and the rectified adam optimizer from the paper 
[On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265) in Tensorflow.

This uses version one of diffGrad from their paper. If you'd like version two, simply remove `tf.math.abs` from line 98 (making it
`1.0 / (1.0 + tf.math.exp(-(prev_g - grad)))`). I haven't implemented versions 3-5 but please feel free to contribute.
