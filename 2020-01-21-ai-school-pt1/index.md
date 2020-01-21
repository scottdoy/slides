---
theme: scottdoy
highlight-style: atelier-dune-light
transition: fade
slide-level: 2

author: Scott Doyle
contact: scottdoy@buffalo.edu
title: Building an AI School for Pathology
subtitle: Teaching Humans and AI Together
date: 2020-01-21
---

# 

## Computational Cell Biology, Anatomy, and Pathology

Modernizing Medical Curricula

## Jacobs School of Medicine and Biomedical Sciences

<iframe
src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d374106.2934798287!2d-79.11410949052187!3d42.900161599625655!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x89d312af82c770df%3A0xd7f9a7357ec78bee!2sJacobs%20School%20Of%20Medicine%20And%20Biomedical%20Sciences!5e0!3m2!1sen!2suk!4v1579598356116!5m2!1sen!2suk"
width="600" height="450" frameborder="0" style="border:0;"
allowfullscreen=""></iframe>

## Downtown Buffalo Campus

![](img/downtown.png){ width=100% }

## PhD in Pathology and Anatomical Sciences

> <span class="fragment">Our PhD program in Computational Cell Biology, Anatomy
> and Pathology aims to produce scientists with <span style="background-color:
> yellow;">knowledge of biological principles at all levels of
> scale</span></span> <span class="fragment">and who are enabled by proficiency
> in <span style="background-color: yellow;">computational imaging methodologies
> </span></span><span class="fragment">and <span
> style="background-color: yellow;">data analyses.</span></span>

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

## Quantitative Structure:<br/> Expression of Disease State

<p class="fragment">Biological structure is **primary data**. </p>

<p class="fragment">We can **quantify** biological structure.</p>

<p class="fragment">We can **model** relationships between **structure and disease**.</p>

## Fundamental Hypothesis

Changes in genomic expression manifest as physical changes in tumor morphology

<div class="fragment l-double">
<div>
![ ](img/badve2008_fig4b1.svg){ width=80% }
</div>
<div> 
![ ](img/badve2008_fig4b2.svg){ width=80% }
</div>
</div>

<p class="fragment" style="text-align: left;"><small>
S. S. Badve et al., JCO (2008),
Paik et al., N Engl J Med (2004)
</small></p>

## Fundamental Hypothesis

Changes in genomic expression manifest as physical changes in tumor morphology

<div>
![](img/paik2004_fig2.svg){ width=80% }
</div>

<p style="text-align: left;"><small>
S. S. Badve et al., JCO (2008),
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

<p style="text-align: right;"><small>
Lee, G., et al. IEEE TMI (2019)
</small></p>

## Training Students in AI

Engineer-Physician Training

## AI Is Just Another Test

<div class="txt-left">
<p class="fragment">**Understanding the Methodology**</p>
<ul><li class="fragment">“How does this test work? What does it require to run?”</li></ul>
<p class="fragment">**Interpreting the Results**</p>
<ul><li class="fragment">“What are the outputs I can expect?”</li></ul>
<p class="fragment">**Software Solutions for AI-Human Interaction**</p>
<ul><li class="fragment">“How do I do it?”</li></ul>
</div>

# 
## Understanding the Methodology

How Does It Work, and What Does It Require

## How Does Machine Learning Work?

<div class="txt-left">
<ul>
<li class="fragment">Supervised vs. Unsupervised Methods</li>
<li class="fragment">Feature Engineering</li>
<li class="fragment">Data Visualization</li>
</ul>
</div>

## Example Problem: Fine Needle Aspirates

<div class="l-double">
<div>
![Benign FNA Image](img/fna_92_5311_benign.png){ width=80% }
</div>
<div> 
![Malignant FNA Image](img/fna_91_5691_malignant.png){ width=80% }
</div>
</div>

<p class="fragment">**Problem Statement:** <br />Predict whether a patient's tumor is
benign or malignant, given an FNA image</p>

## Building Informative Features

<p class="fragment">**Domain knowledge** identifies useful features.</p>
<p class="fragment">Pathologists **already** distinguish benign from malignant tumors.</p>
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

## Texture of the Nuclei

<iframe frameborder="0" seamless='seamless' scrolling=no src="plots/texture_mean.html"></iframe>

## Average Radius of the Nuclei

<iframe frameborder="0" seamless='seamless' scrolling=no src="plots/radius_mean.html"></iframe>

## Multivariate Distribution

<iframe frameborder="0" seamless='seamless' scrolling=no src="plots/scatter_histogram_plot.html"></iframe>

## What Data is Required for Machine Learning?

<div class="txt-left">
<ul>
<li class="fragment">Data Collection / Feature Extraction</li>
<li class="fragment">Data Wrangling and Cleaning</li>
</ul>
</div>

## Characteristics of Good Features

<div class="txt-left">
<p class="fragment">**Descriptive:** Similar within a class, and different between classes</p>
<p class="fragment">**Relevant:** Features should make sense</p>
<p class="fragment">**Invariant:** Not dependent on how you measure them</p>
</div>

## Another Example: Oral Cavity Cancer

<div class="l-double">
<div>
![](img/044a.jpg){ width=80% }
</div>
<div>
![](img/014b.jpg){ width=80% }
</div>
</div>

## OCC Characteristics

| WPOI   | Definition                   | Points |
|--------|------------------------------|--------|
| Type 1 | Pushing border               |      0 |
| Type 2 | Finger-like Growth           |      0 |
| Type 3 | Large Islands, >15 cells     |      0 |
| Type 4 | Small islands, <=15 cells    |     +1 |
| Type 5 | Satellites, >=1mm from Tumor |     +3 |
|        |                              |        |

<p style="text-align: left;"><small>
Brandwein, et al., Am. J of Surg. Path. (2010),
Li, et al. Head and Neck Path. (2013),
Sinha, et al. Mod. Path. (2018)
</small></p>

## Extracting Architecture

<div class="l-double">
<div>
![](img/044a_original_boundaries.png){ width=80% }
</div>
<div>
![](img/014b_boundary.png){ width=80% }
</div>
</div>

## Extracting Architecture

<div class="l-double">
<div>
![](img/044a_delaunay_constrained.png){ width=80% }
</div>
<div>
![](img/014b_delaunay_constrained.png){ width=80% }
</div>
</div>

## Extracting Architecture

<div class="l-double">
<div>
![](img/044a_wave_graphs.png){ width=80% }
</div>
<div>
![](img/014b_wave_graphs.png){ width=80% }
</div>
</div>
## Quantitative Features as Classifiers

<div class="l-multiple" style="grid-template-columns: 1.5fr 0.5fr 1.5fr;">
<div style="grid-row: 1;">
![&nbsp;](img/044a_delaunay_constrained.png){ width=50% }
</div>
<div style="grid-row: 1;"><small>
$$ \begin{bmatrix}
39.7 \\
1189.5 \\
1149.8 \\
192.7 \\
211.6 \\
\vdots \\
\end{bmatrix} $$</small>
</div>
<div style="grid-row: 1 / span 2; vertical-align: bottom;">
![](img/occult_metastasis_tsne.png){width=80%}
</div>
<div style="grid-row: 2;">
![&nbsp;](img/014b_delaunay_constrained.png){ width=50% }
</div>
<div style="grid-row: 2;"><small>
$$ \begin{bmatrix}
35.8 \\
1314.1 \\
1278.2 \\
313.8 \\
339.8 \\
\vdots \\
\end{bmatrix} $$</small>
</div>
</div>


# 
## Interpreting the Results

What do the Outputs Mean

## Calculating Probabilities from Features

<iframe frameborder="0" seamless='seamless' scrolling=no src="plots/pdf_cdf.html"></iframe>

## Understanding Segmentation Probabilities

![Unet Architecture](img/Unet.svg){width=100%}

## Segmentation Output

<div class="l-double">
<div>
![](img/014b.jpg){ width=80% }
</div>
<div>
![](img/014b_color_label.png){ width=80% }
</div>
</div>

## Probabilistic Segmentation

<div class="l-multiple" style="grid-template-columns: 1fr 1fr 1fr;">
<div>
![](img/014b_class1.png){width=100%}
</div>
<div>
![](img/014b_class2.png){width=100%}
</div>
<div>
![](img/014b_class3.png){width=100%}
</div>
</div>

# 
## Computational Assignments

Coding Homework for Non-Engineers

## Nuclei Annotations for Cell Types in Fiji

![](img/Fiji.png){width=70%}

## Displaying Annotated Patches

![](img/matlab_patches.png){width=45%}

## Representing Data in Colorspace

![](img/matlab_rgb_space.png){width=70%}

## Running Classification in MATLAB

![](img/matlab_classification_learner.png){width=70%}

## Data Clustering

<div class="txt-left">
<ul>
<li>Do the data clusters separate out, or are they mixed?</li>
<li class="fragment">Will RGB values be able to distinguish between them?</li>
<li class="fragment">Can other image traits distinguish these groups more easily?</li>
</ul>
</div>

## Classification

<div class="txt-left">
<ul>
<li>What are the classification error rates?</li>
<li class="fragment">Try to reduce error by adjusting classifier parameters.</li>
</ul>
</div>

## Classification Learner

<div class="txt-left">
<ul>
<li>Which classifier performed the best?</li>
<li class="fragment">Did any classifier achieve 100% performance? If so, which one?</li>
<li class="fragment">Create a Confusion Matrix. Which classes are misclassified the most?</li>
</ul>
</div>


# 
## Software Solutions for ML / AI

Walking the Walk

## Software Stack

![](img/software_stack.png){width=70%}

## Digital Image Storage: Omero

![](img/omero.png){width=70%}

## Human-in-the-Loop: Annotation Stations

<div class="l-double">
<div>
![](img/annotation_stations_01.jpg){height=90%}
</div>
<div>
![](img/annotation_stations_02.jpg){height=90%}
</div>
</div>

## Online Annotations

![](img/histomicstk.png){ width=60% }

## Online Quality Assurance and Feedback

<div class="l-double">
<div>
![](img/django01.png){width=80%}
</div>
<div>
![](img/django02.png){width=80%}
</div>
</div>

## Online Quality Assurance and Feedback

![](img/django03.png){width=50%}

# 

## Concluding Remarks

## Goals of the Program

<div class="txt-left">
<ul>
<li>Pathologists will be users of these tools.</li>
<li class="fragment">They should understand:</li>
<ul>
<li class="fragment">How they work</li>
<li class="fragment">How to interpret them</li>
<li class="fragment">When they are an appropriate test to order for a
patient</li>
</ul>
<li class="fragment">**We are open to suggestions and advice on building this program!**</li>
</ul>
</div>




# 

## Thank You!

