---
theme: scottdoy
highlight-style: atelier-dune-light
transition: fade
slide-level: 2

author: Scott Doyle
contact: scottdoy@buffalo.edu
title: Building an AI School for Pathology
subtitle: Part 2&#58; The Backpropocalypse
date: 2020-01-22
---

# 

## Training AI Students

Petabyte Pedagogy

## Why Is It Called a "Black Box"?

<iframe frameborder="0" seamless='seamless' scrolling=yes src="https://playground.tensorflow.org/"></iframe>

## Teaching a School of AI Students

**Solution:** Interrogate the system to see what it's "thinking"!

<ul>
<li class="fragment">**Classification Output** (What is this?)</li>
<li class="fragment">**Intermediate Representation** (How did you get your answer?)</li>
<li class="fragment">**Active Learning** (What are you confused about?)</li>
<li class="fragment">**Error Analysis** (What did you get wrong?)</li>
<li class="fragment">**Generative Models** (Draw a picture of what you think this is.)</li>
</ul>

# 

## Classification Output

*What Is This?*

## Deep Learning for Pathology Segmentation

<div class="l-double">
<div>
Can we replace manual annotations with deep learning?
<p class="fragment">**Patient Data:** Small, use traditional ML</p>
<p class="fragment">**Pixel Data:** Dense, use deep learning</p>
</div>
<div>
![](img/044a_original_boundaries.png){ width=80% }
</div>
</div>

## Deep Learning for Pathology Segmentation

![Unet Architecture](img/Unet.svg){width=100%}


## Deep Learning Segmentation

<div class="l-double">
<div>
![](img/he_tissue_sample.jpg){ width=80% }
</div>
<div>
![](img/ground_truth.png){ width=80% }
</div>
</div>

## Deep Learning Segmentation

<div class="l-double">
<div>
![](img/he_tissue_sample.jpg){ width=80% }
</div>
<div>
![](img/segmentation.jpg){ width=80% }
</div>
</div>


## Deep Learning Segmentation

<video controls>
  <source src="img/montage.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>

# 
## Intermediate Representations

**How did you get your answer?**

## Semantic Segmentation Filters

![](img/Unet.svg){ width=45% }

<div class="l-multiple" style="grid-template-columns: 1fr 1f; row-gap:0; col-gap:0;">
<div style="grid-row: 1;">
![](img/he_filter01.png){ width=100% }
</div>
<div style="grid-row: 1;">
![](img/he_filter02.png){ width=100% }
</div>
<div style="grid-row: 1;">
![](img/he_filter03.png){ width=100%}
</div>
<div style="grid-row: 1;">
![](img/he_filter04.png){ width=100% }
</div>
</div>

## Improving Training Efficiency: Active Learning

<div class="txt-left">
**Random Learning (RL):** Annotate everything!

<p class="fragment">**Active Learning (AL):** Only annotate the good stuff!</p>
<ul>
<li class="fragment">Achieve higher performance using the same number of
samples</li>
<li class="fragment">Use fewer annotated samples to hit a target
performance</li>
</ul>
</div>

## Active Learning Pipeline

![](img/active_learning.png){ width=80% }

## Active Learning Pipeline

![](img/active_learning_interface.png){ width=80% }

## Progression of Random Training

<div class="l-multiple" style="grid-template-columns: 1fr 1fr 1fr 1fr;
row-gap:0;">
<div style="grid-row: 1;">
![Original](img/al_original.png){ width=100% }
</div>
<div style="grid-row: 1;">
![Bootstrap](img/al_bootstrap.png){ width=100% }
</div>
<div style="grid-row: 1;">
![&nbsp;](img/al_rl01.png){ width=100% }
</div>
<div style="grid-row: 1;">
![Random Learning](img/al_rl02.png){ width=100% }
</div>
</div>

## Progression of Active Training

<div class="l-multiple" style="grid-template-columns: 1fr 1fr 1fr 1fr;
row-gap:0;">
<div style="grid-row: 1;">
![Original](img/al_original.png){ width=100% }
</div>
<div style="grid-row: 1;">
![Bootstrap](img/al_bootstrap.png){ width=100% }
</div>
<div style="grid-row: 1;">
![&nbsp;](img/al_al01.png){ width=100% }
</div>
<div style="grid-row: 1;">
![Active Learning](img/al_al02.png){ width=100% }
</div>
</div>
# 
## Double-Checking AI

Fragile Neural Networks

## Fragile Neural Networks

Image **normalization** is critical for neural networks.

<p class="fragment">**Small changes** in image quality can drastically affect
deep learning performance.</p>
<p class="fragment">Serial sections help us explore fragility with a
minimum of biological or technical variability.</p>
<p class="fragment">**Excellent test set for quality assurance!**</p>

## Fragile Neural Networks

<div class="l-double">
<div>
![](img/fragile_orig_01.png){width=70%}
</div>
<div>
![](img/fragile_orig_02.png){width=70%}
</div>
</div>

## Fragile Neural Networks

<div class="l-double">
<div>
![](img/fragile_seg_01.png){width=70%}
</div>
<div>
![](img/fragile_seg_02.png){width=70%}
</div>
</div>

## Fragile Neural Networks

![](img/fragile_combined.png){width=35%}









## Importance of QA and Human-in-the-Loop

<div class="l-double">
<div>
Obviously, normalization and background lighting correction can solve this specific problem...
<p class="fragment">But what about the “generalization” of neural networks?</p>
<p class="fragment">**What other variation exists that we aren’t aware of a
priori? **</p>
</div>
<div>
![](img/fragile_combined.png){width=70%}
</div>
</div>

## Generative Models: Draw Me A Picture

<div class="l-double">
<div>
<p class="fragment">**Variational Autoencoders** and **Generative Adversarial Networks** can recreate images from a learned “latent space” of possible images in the domain.</p>
<p class="fragment">Can we train a system to recognize medical image structures?</p>
</div>
<div>
![](img/gan_faces.png){width=70%}
</div>
</div>

## Nuclei GAN: Which is Real, Which is Fake?

<div class="l-double">
<div>
![](img/gan_fake.png){width=100%}
</div>
<div>
![](img/gan_real.png){width=100%}
</div>
</div>

## Nuclei GAN: Learning to See

<video style="width: 40%;" controls>
  <source src="img/nuclei_montage.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>

# 
## Concluding Remarks

## AI is the Student, Not the Master

You can interrogate AI to understand it better.
<p class="fragment">In training AI, you can develop a deeper understanding of
your data by thorough review and classification.</p>
<p class="fragment">By reviewing and re-training, you understand the AI "thought
process" in a lot of detail, **even if you don't know the details of how the
system is programmed.**</p>
<p class="fragment">A lot like students!</p>

## This is Just the Beginning

**Interpretable AI** is a huge field, the surface of which I have not scratched. 

<p class="fragment">As we develop more efficient teaching methods, it helps us
understand our students -- human or AI -- a little better.</p>
<p class="fragment">It also leads us to more thorough understanding of the
subject matter as well.</p>

# 

## Thank You!

