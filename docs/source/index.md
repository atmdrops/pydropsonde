% HALO-DROPS documentation master file, created by
% sphinx-quickstart on Tue Feb 14 01:00:50 2023.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.

```{warning}
Currently under active development
```
# HALO-DROPS

Stands for **HALO DROpsonde Protocol & Software (HALO-DROPS)**
## Objective

Create a holistic dropsonde protocol for: 
(a) collection, processing and distribution of HALO dropsonde data & 
(b) making operations uniform across future HALO experiments

## Need for a uniform HALO dropsondes protocol

Soundings do not have a better substitute for recording atmospheric profiles, particularly for wind measurements and for measurements in cloudy skies. For this reason, HALO dropsondes are part of the core components for several campaign configurations, such as the <a href="https://www.halo.dlr.de/science/missions/narval2/narval2.html" target="_blank">NARVAL experiments</a>, <a href="http://www.pa.op.dlr.de/nawdex/" target="_blank">NAWDEX</a>, <a href="https://eurec4a.eu/" target="_blank">EUREC$^4$A</a>, <a href="https://halo-ac3.de/" target="_blank">HALO-(AC)$^3$</a>, etc. Despite their usefulness and consistent deployment, there is a lack of uniformity in their operations and data handling across different campaigns. 

Currently, every HALO campaign has its own strategy for dealing with operations, collection, quality control, processing and distribution of dropsondes’ data. This ends up in a lot of redundancy in decision-making and is prone to errors. Moreover, comparing dropsondes among different campaigns becomes cumbersome because of the non-uniformity in data structure. These differences can be resolved by developing a uniform protocol across HALO campaigns and an accompanying software package that makes it easy to standardize data handling of HALO dropsondes, also applicable retroactively to past campaigns’ data. This package aims to build such a protocol, called **HALO DROpsonde Protocol & Software (HALO-DROPS)**.

For a quick start to using `halodrops`, go to {doc}`Getting Started <tutorial/index>`.

```{toctree}
:caption: 'Contents:'
:maxdepth: 1

tutorial/index
howto/index
explanation/index
handbook/index
apidocs/index
```