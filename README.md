# Semantic-editing-using-StyleGAN
A Style-Based Generator Architecture for Generative Adversarial Networks for editing hair attributes on real faces.

The StyleGAN is a continuation of the progressive, developing GAN that is a proposition for training generator models to synthesize enormous high-quality photographs via the incremental development of both discriminator and generator models from minute to extensive pictures.

We can edit one attribute of the human face by finding the hyperplane boundary in the latent space, which can generate amazing yet not perfect results. So far, we can set one attribute as a conditional attribute along with the primary attribute, as discussed in InterfaceGAN. Additionally, when using one attribute to edit our face, some other attributes may also be changed because of their correlations. We believe using a better classifier can control more than two conditions at the same time and make the boundary more explicit.



