---
theme: scottdoy
highlight-style: atelier-dune-light
transition: fade
slide-level: 2

author: Scott Doyle
contact: scottdoy@bufalo.edu
title: Traditional and Deep Learning
subtitle: A Layman's Guide
date: 2020-01-16
---

# 

## Machine Learning

A Brief Introduction

## Machine Learning Definitions

**Machine Learning** (ML) uses **collected data** to do something useful.

<div class="txt-left">
<ul>
<li class="fragment">Find underlying patterns (**knowledge discovery**)</li>
<li class="fragment">Simplify a complex phenomenon (**model building**)</li>
<li class="fragment">Place data into categories (**classification**)</li>
<li class="fragment">Predict future data (**regression**)</li>
</ul>
</div>

## Machine Learning Definitions

The job of the ML expert is to:

<div class="txt-left">
<ul>
<li class="fragment">Understand and identify the **goal**</li>
<li class="fragment">Collect **data**</li>
<li class="fragment">Select an appropriate **model** or **algorithm**</li>
<li class="fragment">Evaluate the system in terms of **costs**</li>
</ul>
</div>

## Types of Machine Learning

<div class="l-double">
<div>
**Supervised Learning**

<p class="fragment">Use **labeled datasets** to classify new, unseen data</p>
</div>
<div> 
**Unsupervised Learning**

<p class="fragment">Use **unlabeled data** to identify natural groups</p>
</div>
</div>

# 

## Example: Cancer Diagnosis

## Example: Biomedical Image Analysis

<div class="l-double">
<div>
![](img/fna_92_5311_benign.png){ width=80% }
</div>
<div> 
![](img/fna_91_5691_malignant.png){ width=80% }
</div>
</div>

## Fine Needle Aspirates

<div class="l-double">
<div>
![Benign FNA Image](img/fna_92_5311_benign.png){ width=80% }
</div>
<div> 
![Malignant FNA Image](img/fna_91_5691_malignant.png){ width=80% }
</div>
</div>

**Problem Statement:** Predict whether a patient's tumor is benign or malignant, given an FNA image

## Data Definitions

The starting point for all ML algorithms is **data**.

<p class="fragment">So... what do we mean by "data"?</p>

## Data Comes in Many Forms

![Complex, Multi-Modal Data](img/data_formats.png){ width=70% }

## Quantitative Structure:<br/> Expression of Disease State

<p class="fragment">Biological structure is **primary data**. </p>

<p class="fragment">We can quantify **biological structure**.</p>

<p class="fragment">We can **model** relationships between **structure and disease**.</p>

## Fundamental Hypothesis

Changes in genomic expression manifest as physical changes in tumor morphology

<div class="fragment fade-in-then-out l-double">
<div>
![ ](img/badve2008_fig4b1.svg){ width=80% }
</div>
<div> 
![ ](img/badve2008_fig4b2.svg){ width=80% }
</div>
</div>

<p style="text-align: left;"><small>
S. S. Badve et al., JCO (2008),
Paik et al., N Engl J Med (2004)
</small></p>

<div class="fragment" style="position:relative; top:-750px;">
![](img/paik2004_fig2.svg){ width=80% }
</div>

## Data Fusion Improves Predictions

<div class="l-multiple" style="grid-template-columns: auto auto auto;">
<div style="grid-row: 1;">
![Quantitative Histology](img/lee2015_quanthisto.png){ height=30% }
</div>
<div style="grid-row: 1;">
![&nbsp;](img/lee2015_lowdim1.png){ height=30% }
</div>
<div style="grid-row: 1 / span 2;vertical-align: middle;">
![Combined Embeddings](img/lee2015_combined.png){ height=30% }
</div>
<div style="grid-row: 2;">
![Mass Spectrometry](img/lee2015_massspect.png){ height=30% }
</div>
<div style="grid-row: 2;">
![Low-Dimensional Embeddings](img/lee2015_lowdim2.png){ height=30% }
</div>
</div>

## Atoms to Anatomy Paradigm

<div class="l-multiple" style="grid-template-columns: 1.5fr 1fr 1fr 1fr; row-gap:0;">
<div style="grid-row: 1 / span 2;">
![](img/ata01.png){ width=100% }
</div>
<div style="grid-row: 1;">
![](img/ata02.png){ height=356 width=456 }
</div>
<div style="grid-row: 1;">
![](img/ata03.png){ height=356 width=456 }
</div>
<div style="grid-row: 1;">
![](img/ata04.png){ height=356 width=456 }
</div>
<div style="grid-row: 2;">
![](img/ata05.png){ height=356 width=456 }
</div>
<div style="grid-row: 2;">
![](img/ata06.png){ height=356 width=456 }
</div>
<div style="grid-row: 2;">
![](img/ata07.png){ height=356 width=456 }
</div>
</div>

# 

## Machine Learning

Learning from Feature Data

## Fine Needle Aspirate Analysis

<div class="l-double">
<div>
![Benign FNA Image](img/fna_91_5691_malignant.png){ width=100% }
</div>
<div>
![Malignant FNA](img/fna_92_5311_benign.png){ width=100% }
</div>

## Building Informative Features

<p class="fragment">**Domain knowledge** identifies useful features.</p>

<p class="fragment">Pathologists already distinguish **benign** from **malignant** tumors.</p>

<p class="fragment">Our job is to convert **qualitative** features to **quantitative** ones.</p>

## Building Informative Features

The pathologist lists **cell nuclei** features of importance:

<div class="l-double">
<div>
1. Radius
2. Texture
3. Perimeter
4. Area
5. Smoothness
</div>
<div>
6. Compactness
7. Concavity
8. Concave Points
9. Symmetry
10. Fractal Dimension
</div>
</div>

<p class="fragment">**Feature extraction** results in 30 feature values per image.</p>

## Selecting Features for the FNA

To begin, we collect **training samples** to build a model.

<div class="txt-left">
<ul>
<li class="fragment">Collect a lot of example images for each class</li>
<li class="fragment">Get our expert to label each image as "Malignant" or "Benign"</li>
<li class="fragment">Measure the features of interest (image analysis or by hand)</li>
<li class="fragment">Build a histogram of the measured feature</li>
</ul>
</div>

## Texture of the Nuclei

<iframe frameborder="0" seamless='seamless' scrolling=no src="plots/texture_mean.html"></iframe>

## Average Radius of the Nuclei

<iframe frameborder="0" seamless='seamless' scrolling=no src="plots/radius_mean.html"></iframe>

## Characteristics of Good Features

<div class="txt-left">
<p class="fragment">**Descriptive:** Similar within a class, and different between classes</p>
<p class="fragment">**Relevant:** Features should make sense</p>
<p class="fragment">**Invariant:** Not dependent on how you measure them</p>
</div>

## Calculating Probabilities from Features

<iframe frameborder="0" seamless='seamless' scrolling=no src="plots/pdf_cdf.html"></iframe>

## Combinations of Features

**Combining features** often yields greater class separation.

## Multivariate Distribution

<iframe frameborder="0" seamless='seamless' scrolling=no src="plots/scatter_histogram_plot.html"></iframe>

## Multivariate Distribution

<iframe frameborder="0" seamless='seamless' scrolling=no src="plots/scatter_plot.html"></iframe>

## Variance vs. Generalization

<p>Linear boundaries do not model **variance** and miss obvious trends.</p>
<p class="fragment">Complex boundaries fit training perfectly, but do not **generalize**.</p>
<p class="fragment">In general, you want the **simplest** model with the best **performance**.</p>

## Tradeoff: Variance vs. Generalization

<p>Each of these decision boundaries makes errors!</p>
<p class="fragment">There is always a tradeoff; we need to consider the **cost**.</p>
<p class="fragment">Cost is defined by our goals and acceptable performance.</p>

## Costs

Should we prioritize some kinds of errors over others?

<div class="fragment txt-box">
Not all mistakes carry the same cost. For example:

- A patient is told they have a tumor when they do not (**false positive**)
- A patient is told they are cancer-free when they are not (**false negative**)
</div>

# 

## Neural Networks

Building Blocks for Deep Learning

## Biological Inspiration for Neural Networks

![Biological Neuron](img/biological_neuron.png){ width=75% }

## Anatomy of a[n Artificial] Neuron

![](img/ann.png){ width=75% }

## Simple Perceptron Decision Space

![Simple Perceptron](img/simple_perceptron_decision_space.png){ width=75% }

## Hidden Layers: Complex Decision Space

![Artificial Neural Network](img/hidden_layer_complex_decision_space.png){ width=75% }

## Simple Problem: XOR Classification

![XOR Problem](img/xor_classification.png){ width=35% }

## Neural Network Solution to XOR

![XOR Problem](img/xor_solution.png){ width=50% }

## Details of Neural Network Weights

![Network Weights](img/network_weights.png){ width=35% }

## Training Neural Networks: Finding the Weights

<div class="l-multiple" style="grid-template-columns: 1.5fr 0.5fr;font-family: 'Caveat';color: #e56a54;">
<div style="grid-row: 1 / 5; grid-column: 1;">
![Backpropagation Schematic](img/backpropagation_schematic.png){ width=60% }
</div>
<div class="fragment fade-in-then-out" data-fragment-index="3" style="grid-row: 1; grid-column: 2;">
Step 3: Calculate error of the result
</div>
<div class="fragment fade-in-then-out" data-fragment-index="4" style="grid-row: 2; grid-column: 2;">
Step 4: Calculate gradients and modify weights and biases
</div>
<div class="fragment fade-in-then-out" data-fragment-index="2" style="grid-row: 3; grid-column: 2;">
Step 2: Calculate network output
</div>
<div class="fragment fade-in-then-out" data-fragment-index="1" style="grid-row: 4; grid-column: 2;">
Step 1: Pick a training example
</div>
</div>


## Why Is It Called A "Black Box"?

<iframe frameborder="0" seamless='seamless' scrolling=yes src="https://playground.tensorflow.org/"></iframe>

# 

## Deep Learning

How Does It Work?

## "Strong" AI

<div class="l-multiple" style="grid-template-columns: 1fr 1fr 1fr;">
<div>
![](img/strong_ai_hal.jpg){ width=100% }
</div>
<div>
![](img/strong_ai_johnny.jpg){ width=100% }
</div>
<div>
![](img/strong_ai_battlestar.jpg){ width=100% }
</div>
</div>

## "Weak" AI

<div class="l-multiple" style="grid-template-columns: 1fr 1fr 1fr;">
<div>
![](img/weak_ai_cars.jpg){ width=100% }
</div>
<div>
![](img/weak_ai_faces.jpg){ width=100% }
</div>
<div>
![](img/weak_ai_translation.jpg){ width=100% }
</div>
</div>

## Deep Classifiers

<div class="txt-left">

**Hand Crafted Features:** Selecting features relevant to the image classes

**Deep Learning:** Use the input samples themselves to identify classes

<p class="fragment">Innovations that make deep learning possible:</p>

<ul>
<li class="fragment">Large amounts of well-annotated data</li>
<li class="fragment">Commodity-level, highly parallel hardware</li>
<li class="fragment">Innovations in training algorithms</li>
</ul>
</div>

## Simple Example: MNIST Handwriting Dataset

![MNIST Handwriting Sample](img/mnist.png){ width=30% }

## Images in Neural Networks

<div class="l-double">
<div>
![Pixels of an Image](img/zerogrid.png)
</div>
<div>
![Vectorized Images](img/zerosgrid.png)
</div>
</div>

## Images in Neural Networks

<div class="l-double">
<div>
![Pixels of an Image](img/onegrid.png)
</div>
<div>
![Vectorized Images](img/onesgrid.png)
</div>
</div>

## Comparing Zeros to Ones

<div class="l-double">
<div>
![All Zeros](img/zeros.png){width=100%}
</div>
<div>
![All Ones](img/ones.png){width=100%}
</div>
</div>

## Images in Neural Networks

<div class="l-double" style="grid-template-columns: 0.75fr 1fr;">
<div>
![Image Input](img/zerogrid_highlighted.png){width=80%}
</div>
<div>
![Input to Neural Network](img/backpropagation_zero.png){width=80%}
</div>
</div>

## Images in Neural Networks

<div class="l-double" style="grid-template-columns: 0.75fr 1fr;">
<div>
![Image Input](img/onegrid_highlighted.png){width=80%}
</div>
<div>
![Input to Neural Network](img/backpropagation_one.png){width=80%}
</div>
</div>


## Do You Know What These Are?

<div class="l-double">
<div>
![](img/he_cell01_resized.png){width=80%}
</div>
<div>
![](img/he_cell02_resized.png){width=80%}
</div>
</div>

## Do You Know What These Are?

<div class="l-double">
<div>
![](img/he_nocell01_resized.png){width=80%}
</div>
<div>
![](img/he_nocell02_resized.png){width=80%}
</div>
</div>

## How Do You Know?

<div class="l-double">
<div class="txt-left" style="width: 100%;">
Let's do some quick **calculations**...

- Number of pixels: $64\times64=4,096$
- Color values: $4,096\times3=12,288$

<p class="fragment">With just over **12,000 values**, our brains can identify the type of object in this image.</p>

<p class="fragment">That seems like a lot, but that's just **12kb** worth of input data!</p>

<p class="fragment">But we aren't done yet...</p>
</div>
<div>
![](img/he_cell01_resized.png){width=80%}
</div>
</div>

## Modifications to NNs Needed

<div class="l-double">
<div class="txt-left">

- Input Size: 12,288
- Hidden Units (double): 24,000
- Input-to-Hidden Weights: 294 Million
- Output Classes: 3
- Hidden-to-Output Weights: 882 Million

<p class="fragment">**Total Weights: 1.17 Billion**</p>

<p class="fragment">Our brains do a **ton** of computing!</p>

<p class="fragment">We need a new approach...</p>

</div>
<div>
![](img/he_cell02_resized.png){width=80%}
</div>
</div>

## Exploiting Spatial Relationships

<div class="l-double">
<div class="txt-left" style="width: 100%;">

Images have some nice properties:

<p class="fragment">**Spatially Localized:** Allows us to restrict the number of weights from input to output</p>
<p class="fragment">**Scale-dependent:** Reducing image scale allows us to find connections between shapes and objects</p>
</div>
<div>
![](img/he_cell01_resized.png){width=80%}
</div>
</div>


## Convolutional Neural Network Architecture

![VGG 16 Network Architecture](img/vgg16.png){width=100%}

## Filter Responses

<div class="l-double">
<div>
![Dog](img/cifar10_dog_input.png)
</div>
<div>
![CNN Architecture](img/vgg16.png){width=100%}
</div>
</div>

## Filter Responses

<div class="l-double">
<div>
![Dog](img/cifar10_dog_input.png)
</div>
<div>
![Filter Responses](img/cifar10_dog_filters.png){width=60%}
</div>
</div>

## Patch-Based Classification: Segmentation

![H&E Tissue Sample](img/he_tissue_sample.jpg){ width=33% }

## Filter Responses

<div class="l-multiple" style="grid-template-columns: 1fr 1fr 1fr 1fr">
<div style="grid-row: 1 / span 2; grid-column: 1;">
![](img/he_tissue_sample.jpg)
</div>
<div style="grid-row: 1; grid-column: 2 / span 3;">
![](img/vgg16.png){width=80%}
</div>
<div style="grid-row: 2; grid-column:2;">
![](img/he_filter01.png){width=80%}
</div>
<div style="grid-row: 2; grid-column:3;">
![](img/he_filter02.png){width=80%}
</div>
<div style="grid-row: 2; grid-column:4;">
![](img/he_filter03.png){width=80%}
</div>
</div>


## Results of Classification

<div class="l-double">
<div>
![Ground Truth Label Map](img/ground_truth.png)
</div>
<div>
![Classification Output](img/segmentation.jpg)
</div>
</div>

# 

## Need for Annotations

The Importance of Data

## Large, Annotated Datasets

<p class="fragment">ML benefits from **large, well-annotated** datasets</p>

<p class="fragment">**Natural** images are abundant, easy-to-label data</p>

<p class="fragment">However, it's not so easy for **pathology**...</p>

## Natural vs. Specialized Image Datasets

<div class="l-multiple" style="grid-template-columns: 1fr 1fr 1fr">
<div>
![Toddler](img/kenji.jpg){ width=100% }
</div>
<div>
![Dog (Akita)](img/kyoshi.jpg){ width=100% }
</div>
<div>
![Tissue (H&E)](img/he_tissue_sample_imagesearch.jpg){ width=100% }
</div>
</div>

## Human Faces Are Well-Annotated

![https://cloud.google.com/vision](img/kenji_imagesearch.png)

## Dog Faces Are Still Common

![https://cloud.google.com/vision](img/kyoshi_imagesearch.png)

## Medical Images are Sparse

!https://cloud.google.com/vision(img/he_imagesearch01.png)

## There Is Some Data However

![https://cloud.google.com/vision](img/he_imagesearch02.png)

## Disparity in Dataset Sizes

![Not Shown: ImageNet (14 Million)](img/image_database_sizes.png)

## Challenges in Building Pathology Datasets

<p class="fragment">**Data Generation:** Limited scope, proprietary software, lack of standards</p>

<p class="fragment">**Data Hosting / Access:** Large, high throughput storage options needed</p>

<p class="fragment">**Annotations:** Difficult, time-consuming, application dependent</p>

## Difficulty in Annotating Samples

<div class="l-double">
<div>
![](img/annotation_large_scale.png){ width=100% }
</div>
<div>
![](img/annotation_small_scale.png){ width=100% }
</div></div>

## Addressing Annotation with Formal Training

<div class="l-double">
<div>
![](img/annotation_stations_01.jpg){ width=100% }
</div>
<div>
![](img/annotation_stations_02.jpg){ width=100% }
</div></div>

# 

## Thank You!

