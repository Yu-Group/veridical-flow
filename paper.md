---
title: 'VeridicalFlow: a python package for building trustworthy data-science pipelines with PCS'
tags:
  - python
  - stability
  - reproducibility
  - data science
  - caching
authors:
  - name: James Duncan^[Equal contribution]
    orcid: 0000-0003-3297-681X
    affiliation: 1
  - name: Rush Kapoor^[Equal contribution]
    affiliation: 2
  - name: Abhineet Agarwal^[Equal contribution]
    affiliation: 3
  - name: Chandan Singh^[Equal contribution]
    orcid: 0000-0003-0318-2340
    affiliation: 2
  - name: Bin Yu
    affiliation: "1, 2"
affiliations:
 - name: Statistics Department, University of California, Berkeley
   index: 1
 - name: EECS Department, University of California, Berkeley
   index: 2
 - name: Physics Department, University of California, Berkeley
   index: 3
date: 21 October 2021
bibliography: references.bib
---

# Summary

`VeridicalFlow` is a Python package for simplifying building reproducible and trustworthy data-science pipelines using the PCS framework [@yu2020veridical].
It provides users a simple interface for stability analysis, i.e. checking the robustness of results from a data-science pipeline to various judgement calls made during modeling.
This ensures that arbitrary judgement calls made by data-practitioners (e.g. specifying a default imputation strategy) do not dramatically alter the final conclusions made in a modeling pipeline.
In addition to wrappers facilitating stability analysis, `VeridicalFlow` also automates many cumbersome coding aspects of python pipelines, including experiment tracking and saving, parallelization, and caching, all through integrations with existing python packages.
Overall, the package helps to code using the PCS (predictability-computability-stability) framework, by screening models for predictive performance, helping automate computation, and facilitating stability analysis.

# Statement of need

Predictability, computability, and stability are central concerns in modern statistical practice, as they are required to help vet that findings reflect reality, can be reasonably computed, and are robust as the many judgement calls during the data-science life cycle which often go unchecked [@yu2020veridical]. 

The package focuses on stability, but also provides wrappers to help support and improve predictability and computability. Stability is a common-sense principle related to notions of scientific reproducibility [@fisher1937design,@ivie2018reproducibility], sample variability, robust statistics, sensitivity analysis [@saltelli2002sensitivity], and stability in numerical analysis and control theory. Moreover, stability serves as a prerequisite for understanding which parts of a model will generalize and can be interpreted [@murdoch2019definitions].

Importantly, current software packages offer very little support to facilitate stability analyses. `VeridicalFlow` helps fill this gap by making stability analysis simple, reproducible, and computationally efficient. This enables a practitioner to represent a pipeline with many different perturbations in a simple-to-code way, while using prediction analysis for reality check and screening out bad models. 

# Features

Using `VeridicalFlows`'s simple wrappers easily enables many best practices for data science, and makes writing pipelines easy.

![](docs/logo_vflow_straight.jpg)

| Stability                                                    | Computability                                                | Reproducibility                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------- |
| Replace a single function (e.g. preprocessing) with a set of functions representing different judgement calls and easily assess the stability of downstream results | Automatic parallelization and caching throughout the pipeline | Automatic experiment tracking and saving |



The main features of `VeridicalFlow` center around stability analysis. The central concept is to replace given functions with a set of functions subject to different pipeline perturbations that are documented and argued for in a PCS documentation. Then, a set of useful analysis functions and computations enable easily assessing the stability to these perturbations.

The package also helps users to improve the efficacy of their computational pipeline. Computation is (optionally) handled through Ray [@moritz2018ray], which easily faciliates parallelization across different machines and along different perturbations of the pipeline. Caching is handled via [joblib](https://joblib.readthedocs.io/en/latest/), so that individual parts of the pipeline do not need to be rerun.

Experiment-tracking and saving are (optionally) handled via integration with MLFlow [@zaharia2018accelerating], which enables automatic experiment tracking and saving.

# Acknowledgements

The work here was supported in part by NSF Grants DMS-1613002, 1953191, 2015341, IIS 1741340, the Center for Science of Information (CSoI), an NSF Science and Technology Center, under grant agreement CCF-0939370, NSF grant 2023505 on Collaborative Research: Foundations of Data Science Institute (FODSI), the NSF and the Simons Foundation for the Collaboration on the Theoretical Foundations of Deep Learning through awards DMS-2031883 and 814639, and a Chan Zuckerberg Biohub Intercampus Research Award.

The code here heavily derives from the wonderful work of previous projects. It hinges on the data-science infrastructure of python, including packages such as pandas [@mckinney2011pandas], numpy [@van2011numpy], and scikit-learn [@pedregosa2011scikit] as well as newer projects such as imodels [@singh2021imodels] and networkx [@hagbergnetworkx].

# References