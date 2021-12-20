# Semantic-editing-using-StyleGAN
A Style-Based Generator Architecture for Generative Adversarial Networks for editing hair attributes on real faces.

![Alt Text](https://github.com/Dantusaikamal/Semantic-editing-using-StyleGAN/blob/main/img/styleGAN%20animation.gif)

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

a. To run the model, click *Runtime* from the top and select *Run all cells*.

b. Once the model begins to run, scroll to *Get images* section, and click on the visual input you see on screen to capture an image.

c. Check the contents of the image in the next section and double click to edit the captured image.

d. Wait for the resultant image to be generated.

**1\. Install the required dependencies:**

The first cell contains the following code:

```!pip install 'h5py<3.0.0'  *# h5py is used to store & manipulate Numeric data*

   !pip install --upgrade tqdm *# tqdm is used for creating progress bars*
```
The above dependencies must be manually installed in the collab environment since the default versions of h5py and tqdm are not supported by StyleGAN.

Similarly, TensorFlow version ***1.14.x / 1.15.x*** must strictly be used in the environment.   

**2\. Clone Git repo and create directories for importing images**

Clone the repository to use the sample dataset and also the latent representations, latent space markings and the adaptive loss functions.

``` 
!git clone https:*//github.com/Azmarie/stylegan-encoder.git*

mkdir aligned_images raw_images
```
**3. Get images**

To capture image from live video input, we are using multiple modules in Python.

Firstly, IPython.display is used to execute HTML code. We need it in order to further execute JavaScript code inside HTML script tags. The live video that is streamed inside the browser is developed using JavaScript methods.

The MediaDevices.getUserMedia() method prompts the user for permission to use a media input which produces a MediaStream with tracks containing the requested types of media.   The JS listens for a click on the button, then calls navigator.mediaDevices.getUserMedia() asking for the video.

Next, to display the image, pillow library is using an image class within it. The image module inside pillow package contains some important inbuilt functions like, load images or create new images, etc.

Once the camera starts capturing, we are using var canvas = document.createElement('canvas')

To create a canvas of the size we require from the live video capture. This canvas is stored in the variable data which is then used in take_photo method to create a snapshot of the canvas.

HTMLCanvasElement.toDataURL() method returns a data URI containing a representation of the image in the format specified by the type parameter. Here, our desired type is jpeg

We are also assigning the timestamp to every image that is capture using the following block:

```
timestampStr = datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

filename = 'raw_images/photo_%s.jpeg' %timestampStr
```
 Example of an image captured: photo_17-Dec-2021 (19:10:21.422345).jpeg

The take_photo method takes quality and resolution as parameters and returns a snapshot of the canvas. And the snapshot is displayed in the browser.  

**4\. Check the contents of our image folder before we start:**

Before we proceed further, let us check whether or not the image captured earlier is saved to the 'raw_images' directory we created in the beginning. If the captured image is present, we can go ahead.

**5\. Auto-Align faces**

The next step is to crop the image and eliminate unnecessary portion of the image. In the canvas, we are going to crop the image and align the face into center of the image. This will result into an aligned image of size 1024*1024.

- Look for faces in the images

- Crop out the faces from the images

- Align the faces (center the nose and make the eyes horizontal)

- Rescale the resulting images and save them in "aligned_images" folder

This is done by executing align_images.py file which uses Keras, argparse and multiprocessing to detect the landmarks and features in the face and align the image accordingly. The LandmarksDetector class is  pre-trained trained on similar dataset, and is used to extract features in the image. Then, the aligned image is printed.  We say two models are aligned if they share the same architecture, and one of them (the child) is obtained from the other (the parent) via fine-tuning to another domain, a common practice in transfer learning.

**6\. Encoding faces into StyleGAN latent space:**

a. Download a pretrained ResNet encoder

```
!gdown https://drive.google.com/uc?id=1aT59NFy9-bNyXjDuZOTMl0qX0jmZc6Zb

!mkdir data

!mv finetuned_resnet.h5 data
```

The pretrained StyleGAN network from NVIDIA trained on faces and a pretrained VGG-16 network,        trained on ImageNet will be downloaded. After guessing the initial latent codes using the pretrained  ResNet, it will run gradient descent to optimize the latent faces

b. Generate encoded images using TensorFlow

```
!python encode_images.py --optimizer=lbfgs --face_mask=False --iterations=50 --use_lpips_loss=0 --use_discriminator_loss=0 --output_video=True aligned_images/ generated_images/ latent_representations/

print("\n************ Latent code optimization finished! ***************")
```
 The slow version takes additional parameters including decay rate, decay steps, early stopping  threshold, average best loss, and has 8 times more number of iterations resulting in a high accuracy  encoded image of the aligned image.

```
!python encode_images.py --optimizer=adam --lr=0.02 --decay_rate=0.95 --decay_steps=6 --use_l1_penalty=0.3 --face_mask=True --iterations=400 --early_stopping=True --early_stopping_threshold=0.05 --average_best_loss=0.5 --use_lpips_loss=0 --use_discriminator_loss=0 --output_video=True aligned_images/ generated_images/ latent_representations/
```

**7\. Generating output from Encoded image:**

 Dnnlib is a standalone library used by Nvidia, hence there is no public documentation available. But  this lib was made for parsing and easy configuration, and also participates in creating and managing  TensorFlow sessions

 The StyleGAN model is loaded into TensorFlow session and is executed using tflib.init_tf().

Lastly, added noise is included in the network to generate more stochastic details in the images. This noise is just a single-channel picture consisting of uncorrelated Gaussian noise. Before each AdaIN operation, noise is supplied at a specific convolutional layer. Additionally, there is a scaling part for the noise, which is decided per feature.

A separate sample of noise for each block is evaluated based on the scaling factors of that layer.

```
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=1)
```

Kwargs, Or keyword arguments let our synthesis function take an arbitrary number of keyword arguments.   The tflib is a low-level library used to convert images to an 8-bit unsigned integer while the output transform argument is a helper argument.

The remaining keyword arguments are optional and can be used to further modify the operation. The output is a batch of images, whose format is dictated by the output_transform argument

Then, the generator.run method is used to generate the synthesized images and are plotted using matplotlib library.

**8\. Final steps** 

We can compare the original image with encoded image and the generated images. Also, store the same images to the disk or mount GDrive and store the images to a folder on Google Drive.

## References:

 https://www.analyticsvidhya.com/blog/2021/05/stylegan-explained-in-less-than-five-minutes/
 https://towardsdatascience.com/gan-papers-to-read-in-2020-2c708af5c0a4
 https://github.com/Dantusaikamal/Telecom-users-churn-analysis 
 https://www.sfu.ca/~ysa195/projects/CMPT743Project/
 https://machinelearningmastery.com/introduction-to-style-generative-adversarial-network-stylegan/

