#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gene Ontology (GO) Enrichment Analysis using GOATOOLS

Created on Fri Jun 28 15:42:17 2024

@author: ferenc.kagan
"""

from goatools.base import download_go_basic_obo, download_ncbi_associations
from goatools.obo_parser import GODag
from goatools.associations import read_ncbi_gene2go
from goatools.go_enrichment import GOEnrichmentStudy
from goatools.godag_plot import plot_results
import sys

# Adjust the system path to include custom scripts if necessary
sys.path.append('/Users/ferenc.kagan/Documents/Projects/hw/temp')
from genes_ncbi_9606_proteincoding import GENEID2NT

##########################################
##### DOWNLOAD AND LOAD GO DATASETS ######
##########################################

# Download the GO basic ontology and NCBI gene2go associations if not already available
# Uncomment the following lines if you need to download the datasets:
# obo_fname = download_go_basic_obo()
# gene2go = download_ncbi_associations()

# Load human gene associations from gene2go file
geneid2gos = read_ncbi_gene2go("gene2go", taxids=[9606])

# Display the number of genes with GO associations
print(f"{len(geneid2gos):,} annotated genes")

# Load the Gene Ontology DAG (Directed Acyclic Graph)
obodag = GODag("go-basic.obo")

##########################################
##### RUN GO ENRICHMENT ANALYSIS #########
##########################################

# Prepare the GO enrichment study object
goeaobj = GOEnrichmentStudy(
    GENEID2NT.keys(),  # List of human protein-coding genes
    geneid2gos,        # GeneID/GO associations
    obodag,            # GO DAG (ontology)
    propagate_counts=False,
    alpha=0.05,        # Significance cutoff
    methods=['fdr_bh'] # Multiple testing correction method
)

# Run enrichment analysis on the selected genes (from previous script)
# Data is still loaded in the session
# Make sure to pass the list of gene IDs as `extracted_numbers`
goea_results_all = goeaobj.run_study(extracted_numbers)

# Filter for significant GO terms
goea_results_sig = [r for r in goea_results_all if r.p_fdr_bh < 0.05]

##########################################
##### SAVE RESULTS AND PLOTS #############
##########################################

# Save the significant GO terms to an Excel file
output_excel_path = "../output/significant_terms.xlsx"
goeaobj.wr_xlsx(output_excel_path, goea_results_sig)

# Generate and save a plot of the significant GO terms
output_plot_path = "../output/significant_terms.png"
plot_results(output_plot_path, goea_results_sig)

print(f"GO enrichment results saved to {output_excel_path} and {output_plot_path}")
