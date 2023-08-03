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

Trying out stuff:

$$latex_{baby}$$

Save latex, baby!

![affine_fish](https://upload.wikimedia.org/wikipedia/commons/f/fe/Cpd_fish_affine.gif)

Unfortunately, there does not exist a formular that takes two images and returns the desired transformation.
But since we can quantify how good the two images are aligned by measuring the distances of the respective points we can solve it **iteratively - step-by-step**.
Applied to this problem, an iterative approach does something like this:
- Apply some transformation to moving image
- Calculate distance between points
- If distance > x, apply another transformation to moving image
- Calculate distance between points 
- If distance > x, apply another transformation to moving image
- Calculate distance between points
- ...

Now we have to load these .nii-files (called "nifti") and convert them into PyTorch tensors.

{%highlight python%}
import numpy as np

size = (32, 32, 32)
moving_mask = F.interpolate(moving_mask[None, None], size)[0, 0]
static_mask = F.interpolate(static_mask[None, None], size)[0, 0]
come_on = 0

{%endhighlight%}

First let's download some brain images:

```python
import requests

def download(url, filepath):
    return 0

download()
download()
download()

```
Bla

## Application to brain images

bla bla

This does not really show 
