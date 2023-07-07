# Conditional GAN Tutorial

This is a PyTorch implementation of the Conditional GAN tutorial. The code is based on [Youtube video](https://www.youtube.com/watch?v=Hp-jWm2SzR8&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=28).

Make the model supervised instead of unsupervised. Generate something conditioned on the label. For example, generate a digit conditioned on the label.

The model is based on WGan-GP. The generator and discriminator are both MLPs. The generator takes a random noise and a label as input and generates a fake image. The discriminator takes a real image and a label or a fake image and a label as input and outputs a score. The label is a one-hot vector.
