# Semantic-editing-using-StyleGAN
A Style-Based Generator Architecture for Generative Adversarial Networks for editing hair attributes on real faces.

The StyleGAN is a continuation of the progressive, developing GAN that is a proposition for training generator models to synthesize enormous high-quality photographs via the incremental development of both discriminator and generator models from minute to extensive pictures.

We can edit one attribute of the human face by finding the hyperplane boundary in the latent space, which can generate amazing yet not perfect results. So far, we can set one attribute as a conditional attribute along with the primary attribute, as discussed in InterfaceGAN. Additionally, when using one attribute to edit our face, some other attributes may also be changed because of their correlations. We believe using a better classifier can control more than two conditions at the same time and make the boundary more explicit.

The features can be divided into three types:

-    Coarse - resolution of up to 82 - affects pose, general hair style, face shape, etc
-    Middle - resolution of 162 to 322 - affects finer facial features, hair style, eyes open/closed, etc.
-    Fine - resolution of 642 to 10242 - affects color scheme (eye, hair and skin) and micro features.

The Mapping Network’s goal is to encode the input vector into an intermediate vector whose different elements control different visual features.

There are many aspects in people’s faces that are small and can be seen as stochastic, such as freckles, exact placement of hairs, wrinkles, features which make the image more realistic and increase the variety of outputs. The common method to insert these small features into GAN images is adding random noise to the input vector. However, in many cases it’s tricky to control the noise effect due to the features entanglement phenomenon that was described above, which leads to other features of the image being affected.

StyleGAN generator uses the intermediate vector in each level of the synthesis network, which might cause the network to learn that levels are correlated. To reduce the correlation, the model randomly selects two input vectors and generates the intermediate vector ⱳ for them. It then trains some of the levels with the first and switches (in a random point) to the other to train the rest of the levels. The random switch ensures that the network won’t learn and rely on a correlation between levels.

# Implementation:

Set up your Google Collab environment and select a runtime environment to run the model. 
a. To run the model, click Runtime from the top and select Run all cells. 
b. Once the model begins to run, scroll to Get images section, and click on the visual input you see on screen to capture an image. 
c. Check the contents of the image in the next section and double click to edit the captured image. 
d. Wait for the resultant image to be generated.

1. Install the required dependencies: 

The first cell contains the following code:
!pip install 'h5py<3.0.0'  	# h5py is used to store and manipulate Numeric data
!pip install --upgrade tqdm 	# tqdm is used for creating progress bars

The above dependencies must be manually installed in the collab environment since the default versions of h5py and tqdm are not supported by StyleGAN.

Similarly, TensorFlow version 1.14.x / 1.15.x must strictly be used in the environment.   

2. Clone Git repo and create directories for importing images

Clone the repository to use the sample dataset and also the latent representations, latent space markings and the adaptive loss functions.

!git clone https://github.com/Azmarie/stylegan-encoder.git
mkdir aligned_images raw_images

## References:

https://www.analyticsvidhya.com/blog/2021/05/stylegan-explained-in-less-than-five-minutes/
https://github.com/NVlabs/stylegan
https://github.com/Azmarie/stylegan-encoder
