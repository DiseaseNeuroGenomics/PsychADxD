# 9_spatial Spatial trascriptomic analysis

The code in this directory is organized to easily facilitate the analysis done in the paper for 10X Xenium data.
It is organized such that individual files can be called from command line sequentially and will generate both data outputs to disk and the figures included in the manuscript.

## Requirements
The code should be run in a Python environment with the following packages installed (version requirements were not checked, but the version used for the analysis is listed):
- scanpy 1.9.8
- squidpy 1.2.2
- scvi-tools 1.1.2 (of particular note, scANVI inference was substantially modified and a version >=1.1.0 is strongly recommended)
- numpy 1.26.3
- pandas 2.1.4
- matplotlib 3.8.2
- seaborn 0.13.1
- scikit-image 0.22.0

## Download data
Xenium data for this analysis is available via [Zenodo](https://doi.org/10.5281/zenodo.14606775).
To run the analysis included here, download the data from this repository, extract all files, and run the code as described below.

## Run analysis
This directory contains 5 Python files, each with a complete command line interface. While `run_scanvi.py` is meant as a stand-alone file, used internally by the rest, the remaining files should be run in sequence to reproduce the results in the manuscript.
To do this, after downloading the data and setting up the environment, simply run the following commands (replace argument in "" by relevant paths):
```
# load the data, filter cells, annotate at class level, generate Fig. S4 b-d
python load_labelClass_filter.py -i "/path/to/extracted/data" -r "/path/to/radc/h5ad" -p "path/to/psychad/color/palette/csv"

# transfer subclass labels, generate Fig. 2 b-c & S4 i
python labelSubclass_plotSpatial.py -i "/path/to/extracted/data/"analysis/final.h5ad -r "/path/to/radc/h5ad" --palette "path/to/psychad/color/palette/csv"

# transfer subtypes, generate Figs. S4 g, h, j
python labelSubtype_plotSpatial.py -i "/path/to/extracted/data/"analysis/xenium_annotated_subclass.h5ad -r "/path/to/radc/h5ad"

# automatically identify cortical domains, calculate spatial densities, make Fig. S4 e, f, k
python labelID_densityAnalysis.py -i "/path/to/extracted/data/"analysis/xenium_annotated_subtype.h5ad -l "path/to/layer/description/csv"
```

## Other notes
Each of the above command line calls can be modified and customized substantially, and individual functions may be imported from the pipelines to be used in independent analyses.
Command line calls and functions contain comments, explanations and help texts to support such use.
