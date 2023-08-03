## Affine Registration in 12 lines of code

Yes, you've read that correctly! 
Affine registration can be done in 12 lines of Python using PyTorch.
This is **good news for (neuro-)imaging people** like me and also a **fun toy problem** to understand what PyTorch is doing.
So, if you are interested in neuroimaging and/or deep learning this post will tickle your whistle!

## What is Affine Registration?

Affine registration is a method which **aligns images** (e.g. 3D brain-images).

Affine registration applies a combination of **translation**, **rotation**, **zooming** and **shearing** transformations.

Let's apply it to two (pointcloudy) fishes to get a visual understanding.

![affine_fish](https://upload.wikimedia.org/wikipedia/commons/f/fe/Cpd_fish_affine.gif)

In this example the blue fish - **moving image** - was aligned (registered) to the red fish - **static image** - such that each blue point matches the position of its corresponding red point.

Unfortunately, there does not exist a formular that takes two images and returns the desired transformation.
But since we can quantify how good the two images are aligned by measuring the **mean distances of the respective points** we can **solve it iteratively - step-by-step**.
Applied to this problem, an **iterative approach does something like this**:
- Apply some transformation to moving image
- Calculate distance between points
- If distance > x, apply another transformation to moving image
- Calculate distance between points 
- If distance > x, apply another transformation to moving image
- Calculate distance between points
- ...

The most naive version of the iterative approach would always apply a random transformation and take a long time until by chance it found a transformation which meets the requirement "distance < x".
As you can see in the GIF, there is a smarter way which smoothly reduces the distance until it meets the requirement.
We will get there at the end of the post!

The **transformation can be fully described by an affine matrix** $$\mathbf{A}$$ (2D: 3x3 matrix, 3D: 4x4 matrix).
The A matrix encodes:
- translation in its last column
- scale (zoom) on its diagonal
- rotation/shear on all non-diagnoal values of the first two rows/columns

![affine_ops](https://neutrium.net/images/mathematics/affine-transformation-all.png)

You might ask "Why encode the transformations in this weird matrix?".
Because we can now **transform** each coordinate point $$\vec{p}$$ **by simply multiplying it with this matrix** $$\mathbf{A}$$.

$$

\vec{p}_{\text{moved}} = \mathbf{A} \cdot \vec{p} = 
\begin{bmatrix}
a & b & c\\
d & e & f\\
0 & 0 & 1\\
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
1 \\
\end{bmatrix}
= 
\begin{bmatrix}
a \cdot x + b \cdot y + c \\
d \cdot x + e \cdot y + f \\
1 \\
\end{bmatrix}

$$

As you can see, we needed to add an extra dimension after $$y$$ with a $$\mathrm{1}$$ to the 2D point to make it work. 
In 3D you would have to do the same after the $$z$$. 
Similary, the last row of the affine matrix will always be $$0$$ $$0$$ ... and one $$1$$ at the end in 2D and 3D. 

Nice, we now introduced all needed concepts to do affine registration in a (naive) iterative fashion!
We could write a (slow) program that would randomly stumble towards better aligning affine transformations. 
Using this starting point we will now march, mumbling the programmers **"Make it work, make it pretty, make it fast!"-mantra**.
Thankfully, PyTorch works fast and is sufficiently pretty!

## Why PyTorch?
Though PyTorch is primarily used for deep learning, it also can be thought of as a **faster NumPy** since:
- It uses **NumPy semantics** (pretty)

{%highlight python%}
import torch
import numpy as np

# Using numpy
zeros  = np.zeros((4, 4))
ones   = np.ones((4, 4))
random = np.random.random((4, 4))
# Using torch
zeros  = torch.zeros(4, 4)
ones   = torch.ones(4, 4)
random = torch.rand(4, 4)
{%endhighlight%}
- Besides CPU ("normal" processor) it also **supports GPU** (graphics card) computation, which can be **10-100x faster**
{%highlight python%}
import torch

zeros  = torch.zeros(4, 4)
zeros_gpu = zeros.cuda()  # zeros_gpu stored in (NVIDIA) GPU
{%endhighlight%}

For our problem (affine registration) it has **two more powerful features**.

1. There exist **utility functions** which allow you to apply **affine transformations to 2D and 3D images**!
2. PyTorch offers **automatic differentiation** which is very useful since iterative optimization is much faster when a derivative can be calculated!

To understand the second point we have to answer the following question: **What is a gradient?**

If you can, try to remember what a derivative is (you probably had to know it during middle school math class).

- The derivative $$\frac{df}{dx}$$ of a function $$f(x)$$ is its rate of change w.r.t. (with respect to) $$x$$. 

Let's make it more concrete and say that $$x$$ **is an affine matrix** and the function $$f$$ **is the distance of points** between two fish images.
Wow, what a useful concept for our problem: Now the **derivative expresses how much the distance changes w.r.t. the affine matrix**.
Since $$x$$ holds 16 elements (all values of the 4x4 affine matrix) the derivative also contains 16 elements - each simply being the derivative of the distance w.r.t. the respective matrix element.
Bravo, this multivariable derivative is the gradient we wanted to understand!

- The gradient $$\nabla$$ of a function $$f(x_1, x_2,...)$$ is its rate of change w.r.t $$x_1$$, $$x_2$$,...

$$
\nabla =
\begin{bmatrix}
\frac{df}{dx_1} \\
\frac{df}{dx_2} \\
... \\
\end{bmatrix}
$$

The beauty about the **gradient** is that it **always points in the direction (here, affine matrix change) of the maximum increase of the function (here, maximum increase of distance)**.
So, if we want to minimize the distance we just have to change the affine matrix in the opposite direction.
Following this direction during our iterative approach, will result in smooth improvement as shown in the GIF.

- Apply some affine to moving image
- Calculate distance between points
- Calculate the gradient of the distance w.r.t affine
- Change affine in descending gradient direction and apply to moving image
- Calculate distance between points
- Calculate the gradient of the distance w.r.t affine
- Change affine in descending gradient direction and apply to moving image
- ...

## The 12 lines

Now that we know what Affine Registration is and what PyTorch offers us, lets look at the code.

{%highlight python%}
import torch
import torch.nn.functional as F

def affine_registration(moving, static, n_iterations=200, learning_rate=1e-3):
    affine = torch.eye(4)[None, :3]
    affine = torch.nn.Parameter(affine)
    optimizer = torch.optim.Adam([affine], learning_rate)
    for i in range(n_iterations):
        optimizer.zero_grad()
        affine_grid = F.affine_grid(affine, [1, 3, *static.shape])
        moved = F.grid_sample(moving[None, None], affine_grid)
        loss = - dice_score(static[None, None], moved)
        loss.backward()
        optimizer.step()
    return affine.detach()
{%endhighlight%}

For people who have used PyTorch for deep learning, the `affine_registration` function should look very familiar.
It looks like [code which trains a neural net](https://github.com/pytorch/examples/blob/main/mnist/main.py)!

Let's run through it, line by line:
1. The function takes a `static` (image), a `moving` (image), `n_iterations` (number of iterations) and a `learning_rate`
2. `affine` (matrix) is initialized using `torch.eye` (4x4 matrix filled with zeros + ones on diagonal -> affine with no effect)
3. `affine` is made a `torch.nn.Parameter` which **will be optimized** if passed to an optimizer
4. `optimizer` is initialized using the SGD (Stochastiv Gradient Descent) optimizer with the given `learning_rate`
5. Starting a for-loop which will repeat/iterate lines 6-11 for `n_iterations` times
6. `optimizer.zero_grad()` initializes all derivatives (stored in the background) to zero
7. `affine` is transformed into an `affine_grid`...
8. ...which is used to apply the affine transformation to the `moving` image
9. A `loss` value is calculated by measuring the negative Dice score (similarity metric) between `static` and `moved` image
10. `loss.backward()` calculates the **derivative/gradient of the loss w.r.t the parameters** using the chain rule (`loss` -> `moved` -> `affine_grid` -> `affine_grid` -> `affine`)
11. `optimizer.step()` **changes the** `affine` in the opposite of the gradient direction (gradient **descent**) to minimize the loss
12. The optimized `affine` parameter is converted back to a tensor `.detach()` and returned

The code deals with (3D) **images instead of points** now, which is why **lines 7-9 need some extra explanation**:

An **image** can be thought of as a **grid of pixels/points**. 
Applying an affine transformation to each of these pixels - i.e. multiplying its coordinates with the affine matrix, happening in **F.affine_grid** - works just fine BUT:
You end up with **new pixel coordinates** which are **not placed perfectly on a rectangular grid anymore**.
So in 2D each "old" pixel typically ends up somewhere in a 2x2 pixel area of the new image.
The standard approach to deal with this is **interpolation** which is what **F.grid_sample is doing for us in the background**.

![grid_affine](https://discuss.pytorch.org/uploads/default/original/3X/1/d/1d5046f3be18f55e5145a59bde922eef0d3bf09a.jpeg)

Finally, the `dice_score` needs explanation.

{%highlight python%}
def dice_score(x1, x2):
    inter = torch.sum(x1 * x2, dim=[2, 3, 4])
    union = torch.sum(x1 + x2, dim=[2, 3, 4])
    return (2. * inter / union).mean()
{%endhighlight%}

The **Dice score** is doing what the distance between the corresponding red and blue fish points was doing earlier in the post: It **measures image alignment**.
As shown below, the Dice score is **0 for non-overlapping** and **1 for perfectly overlapping image areas**.
PyTorch always tries to minimize loss functions -> We use `-dice_score` and hope that it approaches -1 😉

![dice](https://miro.medium.com/v2/resize:fit:1400/1*tSqwQ9tvLmeO9raDqg3i-w.png)

## Application to brain images

After this theoretical fugazi you might think "Talk is cheap, just show me a demo!" so here we go.
There is also a Colab notebook where you can rerun the demo!

Let's download two brain images

{%highlight python%}
import nibabel as nib
import urllib.request

def download(url, filepath):
    urllib.request.urlretrieve(url, filepath)


moving_fpath, static_fpath = 'moving.nii.gz', 'static.nii.gz'
download('https://openneuro.org/crn/datasets/ds003835/snapshots/1.0.0/files/sub-10:anat:sub-10_T1w.nii.gz', moving_fpath)
download('https://openneuro.org/crn/datasets/ds003835/snapshots/1.0.0/files/sub-20:anat:sub-20_T1w.nii.gz', static_fpath)
# Load niftis
moving_nii = nib.load(moving_fpath)
static_nii = nib.load(static_fpath)

{%endhighlight%}

and plot them (using `.orthoview()`) to see how misaligned the brains are

{%highlight python%}
moving_nii.orthoview()
{%endhighlight%}



{%highlight python%}
static_nii.orthoview()
{%endhighlight%}



The `moving_nii`-brain is not as accurately aligned with the crosshairs as the `static_nii`-brain.
Let's fix that by registering `moving_nii` to `static_nii`!

First, we have to convert these .nii-files (called "Nifti") into PyTorch tensors.

{%highlight python%}
# Get numpy arrays out of niftis
moving = nib.as_closest_canonical(moving_nii).get_fdata()
static = nib.as_closest_canonical(static_nii).get_fdata()
# Convert numpy arrays to torch tensors
moving = torch.from_numpy(moving).float()
static = torch.from_numpy(static).float()

{%endhighlight%}

And then we can apply the beautiful "affine_registration" function.
{%highlight python%}
# Get masks (1 if voxel is brain tissue else 0)
moving_mask = (moving > 0).float()
static_mask = (static > 0).float()
# Reduce resolution to 32³
size = (32, 32, 32)
moving_mask = F.interpolate(moving_mask[None, None], size)[0, 0]
static_mask = F.interpolate(static_mask[None, None], size)[0, 0]
# Do affine registration
optimal_affine = affine_registration(moving_mask, static_mask)

{%endhighlight%}

Oops, I sneaked two more operations in there:

1. Using `(x > 0).float()` to create brainmasks, since the Dice score is build to deal with masks
2. Reducing the resolution of those masks because `affine_registration` will be much faster this way

Finally, we want can use the `optimal_affine` to align the `moving` tensor
{%highlight python%}
affine_grid = F.affine_grid(optimal_affine, [1, 3, *static.shape])
moved = F.grid_sample(moving[None, None], affine_grid)[0, 0]
{%endhighlight%}

Did we do everything right? Let's **visualize** to check!  
{%highlight python%}
moved.orthoview()
{%endhighlight%}

Nice, it worked fine!

Optionally, we can convert the tensor back to a Nifti which can be **saved**: 
{%highlight python%}
moved_nii = nib.Nifti1Image(moved.cpu().numpy(), static_nii.affine)
moved_nii.to_filename('moved.nii.gz')
{%endhighlight%}

## torchreg: Lightweight image registration library using PyTorch

Welcome to the advertising bit of this post!

You did not really think I would go through the trouble of explaining you all this stuff, just to share my excitement about short code, did ya?!

I have added a few tweaks and tricks which are missing to make it a "mature" registration tool and ended up with **~100 lines** I call **torchreg**.

torchreg can be installed via pip

{%highlight bash%}
pip install torchreg
{%endhighlight%}

and provides you with the AffineRegistration class
{%highlight python%}
from torchreg import AffineRegistration
{%endhighlight%}
which supports

- **choosing** which **operations** (translation, rotation, zoom, shear) to change during optimization

{%highlight python%}
reg = AffineRegistration(with_translation=True, with_rotation=True, with_zoom=True, with_shear=False)
{%endhighlight%}
- start optimization with **initial parameters**

{%highlight python%}
reg = AffineRegistration(zoom=torch.Tensor([1.5, 2., 1.]))
{%endhighlight%}

- using a **multiresolution approach** to save compute (per default it runs with 1/4 and then 1/2 of the original resolution for 500 + 100 iterations)

{%highlight python%}
reg = AffineRegistration(scales=(4, 2), iterations=(500, 100))
{%endhighlight%}
- and using **custom similarity functions, optimizers and learning_rates**

{%highlight python%}
reg = AffineRegistration()
{%endhighlight%}

After initializing, you can **run the Affine Registration** with
{%highlight python%}
aligned_brain_mask = reg(brain_mask[None, None], template_mask[None, None])
{%endhighlight%}
and it will return the registered moving image!

Input has to be a torch Tensor (following the [Batch, Channel, Height, Width, Depth] convention, hence the [None, None] adding [Batch, Channel] dimensions).

With 
{%highlight python%}
aligned_brain_mask = reg(brain_mask[None, None].cuda(), template_mask[None, None].cuda())
{%endhighlight%}
you can **leverage your GPU** (if you have a NVIDIA GPU) and speed up the registration.

You can easily **access the affine**
{%highlight python%}
affine = reg.get_affine()
{%endhighlight%}
and the four parameters
{%highlight python%}
translation = reg.parameters[0]
rotation = reg.parameters[1]
zoom = reg.parameters[2]
shear = reg.parameters[3]
{%endhighlight%}

Thats it with the advertising bit!

## Conclusion

I think your conclusions out of this blog post highly depend on your background. 
Hopefully they look something like this: 

1. You are a **neuroimaging person**
- I finally really understand this "Affine Registration" my toolboxes uses
- PyTorch is some fast neural network stuff. I didn't understand that part!
- The next time I preprocess or write a toolbox for preprocessing **I'll use torchreg**!

2. You are a **deep learning person**
- I finally kinda understand what PyTorch does in the background of the training loops I always use!
- I will copy&paste the ~10 lines and will plot/print the affine, the moved image and the gradients in the loop to get a feeling for what is happening in the background!
- The word gradient does not scare me anymore!
- If I ever want to apply Affine Registration **I'll use torchreg**!

3. You are a **nerd**
- Interesting read, nice comprehensive + short code!
- **I'll give the repo a star** because this post entertained me!

4. You are a **normie**
- What is this weird guy talking about?!

One last closing remark: The code I showed you relied on accurate brainmasks. 
If you are interested in how to get these **accurate masks**, take a look into my **next blog post** (soon)!
