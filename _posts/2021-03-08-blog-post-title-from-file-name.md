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

This does not really show 
