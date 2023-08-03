## Blog Post Title From First Header

Due to a plugin called `jekyll-titles-from-headings` which is supported by GitHub Pages by default. The above header (in the markdown file) will be automatically used as the pages title.

If the file does not start with a header, then the post title will be derived from the filename.

This is a sample blog post. You can talk about all sorts of fun things here.

---

### This is a header

#### Some T-SQL Code

```tsql
SELECT This, [Is], A, Code, Block -- Using SSMS style syntax highlighting
    , REVERSE('abc')
FROM dbo.SomeTable s
    CROSS JOIN dbo.OtherTable o;
```

#### Some PowerShell Code

```powershell
Write-Host "This is a powershell Code block";

# There are many other languages you can use, but the style has to be loaded first

ForEach ($thing in $things) {
    Write-Output "It highlights it using the GitHub style"
}
```

# Affine Registration in ~10 lines of code using PyTorch

Yes, you've read that correctly! 
Affine registration can be done in ~10 lines of Python using PyTorch.
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

The **transformation can be fully described by an affine matrix A** (2D: 3x3 matrix, 3D: 4x4 matrix).
The A matrix encodes:
- translation in its last column
- scale (zoom) on its diagonal
- rotation/shear on all non-diagnoal values of the first two rows/columns

![affine_ops](https://neutrium.net/images/mathematics/affine-transformation-all.png)

You might ask "Why encode the transformations in this weird matrix?".
Because we can now transform each coordinate point $p$ by simply multiplying it with this matrix $A$.

$$p_{moved} = A \cdot p = () \cdot $$

Nice, we now introduced all needed concepts to do affine registration in a (naive) iterative fashion!
We could write a (slow) program that would work. But we do not stop here. We want the program to be perfect. 
The mantra of perfection in programming is: **Make it work, make it pretty, make it fast!**
Therefore, we need PyTorch!


## Why PyTorch?
Though PyTorch is primarily used for deep learning, it also can be thought of as a **faster NumPy** since:
- It uses **NumPy semantics** 

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
zeros_gpu = zeros.cuda()  # zeros_gpu is stored in the (NVIDIA) GPU which enables fast computation
{%endhighlight%}

For our problem (affine registration) it has **two more powerful features**.

1. There exist **utility functions** which allow you to apply **affine transformations to 2D and 3D images**!
2. PyTorch offers **automatic differentiation** which is very useful since iterative optimization is much faster when a derivative can be calculated!

To understand the second point we have to answer the following question: **What is a gradient?**

If you can, try to remember what a derivative is (you probably had to know it during middle school math class).

- The derivative of a function f(x) is its rate of change w.r.t (with respect to) x. 

Let's make it more concrete and say that **x is an affine matrix** and the function **f is the distance of points** between two fish images.
Wow, what a useful concept for our problem: Now the **derivative expresses how much the distance changes w.r.t the affine matrix**.
Since x holds 16 elements (all values of the 4x4 affine matrix) the derivative also contains 16 elements - each simply being the derivative of the distance w.r.t. the respective matrix element.
Bravo, this multivariable derivative is the gradient we wanted to understand!

- The derivative of a function f(x_1, x_2) is its rate of change w.r.t x_1, x_2...

The beauty about the **gradient** is that it **always points in the direction (here, affine matrix change) of the maximum increase of the function (here, maximum increase of distance)**.
So, if we want to minimize the distance we just have to change the affine matrix in the opposite direction.
This explains what "some transformation" in [1] is and how it results in a smooth reduction of the distance as shown in the GIF.

## The ~10 lines

Now that we know what Affine Registration is and what PyTorch offers us, lets look at the code.
