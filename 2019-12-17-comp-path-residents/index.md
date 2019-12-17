---
theme: scottdoy
highlight-style: atelier-dune-light
transition: slide
slide-level: 2

author: Scott Doyle
contact: scottdoy@bufalo.edu
title: Introduction to Machine Learning
subtitle: (aka Artificial Intelligence)
date: 2019-12-17
---

# 

## Machine Learning

A Brief Introduction

## Machine Learning Definitions

**Machine Learning** (ML) uses **collected data** to do something useful.

- Find underlying patterns (**knowledge discovery**)
- Simplify a complex phenomenon (**model building**)
- Place data into categories (**classification**)
- Predict future data (**regression**)

## Machine Learning Definitions

The job of the ML expert is to:

- Understand and identify the **goal**
- Collect **data**
- Select an appropriate **model** or **algorithm**
- Evaluate the system in terms of **costs**

## Types of Machine Learning

<div class="l-double">
<div>
**Supervised Learning**

Train a model using labeled datasets to predict the class of new, unseen data
</div>
<div> 
**Unsupervised Learning**

Identify natural groupings and patterns in unlabeled datasets
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

So... what do we mean by "data"?

## Data Comes in Many Forms

![Complex, Multi-Modal Data](img/data_formats.png){ width=70% }

## Quantitative Structure:<br/> Expression of Disease State

Biological structure is **primary data**.

We can quantify **biological structure**.

We can **model** relationships between **structure and disease**.

## Fundamental Hypothesis

Changes in genomic expression manifest as physical changes in tumor morphology

<div class="l-double">
<div>
![ ](img/badve2008_fig4b1.svg){ width=80% }
</div>
<div> 
![ ](img/badve2008_fig4b2.svg){ width=80% }
</div>
</div>

<p style="text-align: left;"><small>
S. S. Badve et al., JCO (2008)
</small></p>

## Fundamental Hypothesis

Changes in genomic expression manifest as physical changes in tumor morphology

![](img/paik2004_fig2.svg){ width=80% }

<p style="text-align: left;"><small>
Paik et al., N Engl J Med (2004)
</small></p>

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

## Bulding Informative Features

**Domain knowledge** identifies useful features.

Pathologists already distinguish **beign** from **malignant** tumors.

Our job is to convert **qualitative** features to **quantitative** ones.

## Building Informative Features

The pathologist lists cell nuclei features of importance:

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

**Feature extraction** results in 30 feature values per image.

## Selecting Features for the FNA

To begin, we collect **training samples** to build a model.

- Collect a lot of example images for each class
- Get our expert to label each image as "Malignant" or "Benign"
- Measure the features of interest (image analysis or by hand)
- Build a histogram of the measured feature

## Texture of the Nuclei

<iframe frameborder="0" seamless='seamless' scrolling=no src="plots/texture_mean.html"></iframe>

## Average Radius of the Nuclei

<iframe frameborder="0" seamless='seamless' scrolling=no src="plots/radius_mean.html"></iframe>

## Characteristics of Good Features

- **Descriptive:** Similar within a class, and different between classes
- **Relevant:** Features should make sense
- **Invariant:** Not dependent on how you measure them

## Calculating Probabilities from Features

<iframe frameborder="0" seamless='seamless' scrolling=no src="plots/pdf_cdf.html"></iframe>

## Combinations of Features

**Combining features** often yields greater class separation.

## Multivariate Distribution

<iframe frameborder="0" seamless='seamless' scrolling=no src="plots/scatter_histogram_plot.html"></iframe>

## Multivariate Distribution

<iframe frameborder="0" seamless='seamless' scrolling=no src="plots/scatter_plot.html"></iframe>

## Tradeoff: Variance vs. Generalization

Linear boundaries do not model **variance** and miss obvious trends.

Complex boundaries fit training perfectly, but do not **generalize**.

In general, you want the **simplest** model with the best **performance**.

## Tradeoff: Variance vs. Generalization

Each of these decision boundaries makes errors!

There is always a tradeoff; we need to consider the **cost**.

Cost is defined by our goals and acceptable performance.

## Costs

Should we prioritize some kinds of errors over others?

<div class="txt-box">
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

## Simple Perceptron Decision Space

![Simple Perceptron](img/simple_perceptron_decision_space.png){ width=75% }

## Hidden Layer: Complex, Nonlinear Decision Space

![Artificial Neural Network](img/hidden_layer_complex_decision_space.png){ width=75% }

## Simple Problem: XOR Classification

![XOR Problem](img/xor_classification.png){ width=35% }

## Neural Network Solution to XOR

![XOR Problem](img/xor_solution.png){ width=50% }

## Details of Neural Network Weights

![Network Weights](img/network_weights.png){ width=35% }

## Training Neural Networks: Finding the Weights

![Backpropagation Schematic](img/backpropagation_schematic.png){ width=40% }

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

Innovations that make deep learning possible:

- Large amounts of well-annotated data
- Commodity-level, highly parallel hardware
- Innovations in training algorithms
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
<div class="txt-left">
Let's do some quick **calculations**...

- Number of pixels: $64\times64=4,096$
- Color values: $4,096\times3=12,288$

With just over **12,000 values**, our brains can identify the type of object in this image.

That seems like a lot, but for a computer, that's just **12kb**!

But we aren't done yet...
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
- **Total Weights: 1.17 Billion**
- We need a new approach...

</div>
<div>
![](img/he_cell01_resized.png){width=80%}
</div>
</div>

## Exploiting Spatial Relationships

<div class="l-double">
<div class="txt-left">

Images are:

- **Spatially Localized:** Allows us to restrict the number of weights from input to output
- **Scale-dependent:** Reducing image scale allows us to find connections between shapes and objects
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

## Thank You!

