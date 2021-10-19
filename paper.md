---
title: 'VeridicalFlow: a Python package for building build stable, trustworthy data-science pipelines'
tags:
  - python
  - stability
  - reproducibility
  - data science
  - caching
authors:
  - name: James Duncan^[Equal Contribution]
    orcid: 0000-0003-3297-681X
    affiliation: 4
  - name: Rush Kapoor^[Equal Contribution]
    affiliation: 2
  - name: Abhineet Agarwal^[Equal Contribution]
    affiliation: 3
  - name: Chandan Singh^[Equal Contribution]
    orcid: 0000-0003-0318-2340
    affiliation: 2
  - name: Bin Yu
    affiliation: "1, 2, 4"
affiliations:
 - name: Graduate Group in Biostatistics, University of California, Berkeley
   index: 1
 - name: EECS Department, University of California, Berkeley
   index: 2
 - name: Physics Department, University of California, Berkeley
   index: 3
 - name: Statistics Department, University of California, Berkeley
   index: 4
date: 21 October 2021
bibliography: references.bib
---

# Summary

`VeridicalFlow` is a Python package that simplifies the building of reproducible and trustworthy data-science pipelines.
It provides users a simple interface for stability analysis, i.e. checking the robustness of results from a data-science pipeline to various judgement calls made during modeling.
This ensures that arbitrary judgement calls made by data-practitioners (e.g. specifying a default imputation strategy) do not dramatically alter the final conclusions made in a modeling pipeline.
In addition to wrappers facilitating stability analysis, `VeridicalFlow` also automates many cumbersome coding aspects of Python pipelines, including experiment tracking and saving, parallelization, and caching, all through integrations with existing Python packages.

# Statement of need

Stability is a central concern in modern statistical practice, as practitioners make many judgement calls during the data-science life cycle which often go unchecked [@yu2020veridical].  It is a common-sense principle related to notions of scientific reproducibility [@fisher1937design; @ivie2018reproducibility] and sensitivity analysis [@saltelli2002sensitivity]. Moreover, stability serves as a prerequisite for understanding which parts of a model will generalize and can be interpreted [@murdoch2019definitions].

Importantly, current software packages offer very little support to facilitate stability analyses. `VeridicalFlow` helps fill this gap by making stability analysis simple, reproducible, and computationally efficient. This enables a practitioner to represent a pipeline with many different perturbations in a simple-to-code way. 

# Features

Using `VeridicalFlows`'s simple wrappers easily enables many best practices for data science, and makes writing pipelines easy.

| Stability                                                    | Computation                                                  | Reproducibility                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------- |
| Replace a single function (e.g. preprocessing) with a set of functions and easily assess the stability of downstream results | Automatic parallelization and caching throughout the pipeline | Automatic experiment tracking and saving |



The main features of `VeridicalFlow` center around stability analysis. The central concept is to replace given functions with a set of functions subject to different pipeline perturbations. Then, a set of useful analysis functions and computations enable easy assessment of stability to these perturbations.

The package also helps users to improve the efficacy of their computational pipeline. Computation is (optionally) handled through [Ray](https://www.ray.io/) [@moritz2018ray], which easily faciliates parallelization across different machines and along different perturbations of the pipeline. Caching is handled via [joblib](https://joblib.readthedocs.io/en/latest/), so that individual parts of the pipeline do not need to be rerun.

Experiment-tracking and saving are (optionally) handled via integration with [MLflow](https://mlflow.org/) [@zaharia2018accelerating], which enables automatic experiment tracking and saving.

# Acknowledgements

The code here heavily derives from the wonderful work of previous projects. It hinges on the data-science infrastructure of Python, including packages such as [pandas](https://pandas.pydata.org/) [@mckinney2011pandas], [NumPy](https://numpy.org/) [@van2011numpy], and [scikit-learn](https://scikit-learn.org/) [@pedregosa2011scikit] as well as newer projects such as [imodels](https://github.com/csinva/imodels) [@singh2021imodels] and [NetworkX](https://networkx.org/) [@hagbergnetworkx].

# References
