# Folder structure

## Important scripts

### extract_FPGA_required_data

The file that extract needed data for FPGA given a trained CPU index.

For example, given an index of SIFT100M PQ16 8192 clusters, extract:

* query vectors
* quantizers
  * product quantizer / vector quantizer
* software results
  * vec_ID / distance
* HBM contents
  * the contents for each bank, include paddings
  * control info: for each cluster, the start entry ID, the total entries in this cluster, and for the last entry where is the last non-padding element

Note: currently, OPQ is not supported in this script.

### IVFPQ_1B_search

The software reference for the hardware design: using self-implemented functions for the core search functions. The results are verified on a trained 1B dataset.

This script also saves some intermediate results for hardware implementation verification, e.g., distance LUT. These contents are not needed given each module of the hardware implementation is already verified.

Content includes:

* selecting clusters to scan in
* construct distance LUT
* estimate distance between query vector and database vectors by LUT addition
* perform search given a query

### IVFPQ_sample_search

The toy version of "IVFPQ_1B_search". Results not verified on real datasets.

## unused test scripts

load_npy_files

access_PQ_centroids

get_invlists

get_matrix_from_PCA