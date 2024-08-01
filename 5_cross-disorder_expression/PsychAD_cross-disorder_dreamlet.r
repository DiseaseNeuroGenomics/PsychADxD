# purge all functions and variables
rm(list=ls())
ls(pos = ".GlobalEnv")
Sys.setenv(TZ='America/New_York')

# dreamlet universe
library(dreamlet)
library(crumblr)
library(zenith)

# data IO
library(SingleCellExperiment) 
library(zellkonverter)
library(tidyr)
library(tidyverse)
library(arrow)

# plotting
library(ggplot2)
library(dplyr)
library(RColorBrewer)
library(cowplot)
library(ggtree)
library(aplot)
library(purrr)
library(circlize)
library(ComplexHeatmap)

# meta
library(muscat)
library(metafor)
library(broom)

# versions
sessionInfo()
print(paste0('dreamlet v',packageVersion("dreamlet")))
print(paste0('crumblr v',packageVersion("crumblr")))
print(paste0('variancePartition v',packageVersion("variancePartition")))
print(paste0('zenith v',packageVersion("zenith")))
print(paste0('zellkonverter v',packageVersion("zellkonverter")))
print(paste0('BiocManager v',BiocManager::version()))

# Protein Coding Genes
pcg = read.csv('/path/to/PsychAD_rowData_34890.csv')
pcg = pcg[pcg$gene_type=='protein_coding','gene_name']

# crossDis with meta-analysis, within FDR
meta = read_parquet('/path/to/res_meta.parquet')
crossDis_AD <- meta %>% filter(AnnoLevel=='subclass' & coef=='m10x' & method=='FE') %>% filter(ID %in% pcg) %>% select(AnnoLevel, ID, assay, estimate, std.error, statistic, p.value) %>% group_by(assay) %>% mutate(FDR = p.adjust(p.value, method = "fdr"), log10FDR = log10(FDR))
crossDis_SCZ <- meta %>% filter(AnnoLevel=='subclass' & coef=='m11x' & method=='FE') %>% filter(ID %in% pcg) %>% select(AnnoLevel, ID, assay, estimate, std.error, statistic, p.value) %>% group_by(assay) %>% mutate(FDR = p.adjust(p.value, method = "fdr"), log10FDR = log10(FDR))
crossDis_DLBD <- meta %>% filter(AnnoLevel=='subclass' & coef=='m12x' & method=='FE') %>% filter(ID %in% pcg) %>% select(AnnoLevel, ID, assay, estimate, std.error, statistic, p.value) %>% group_by(assay) %>% mutate(FDR = p.adjust(p.value, method = "fdr"), log10FDR = log10(FDR))
crossDis_Vas <- meta %>% filter(AnnoLevel=='subclass' & coef=='m13x' & method=='FE') %>% filter(ID %in% pcg) %>% select(AnnoLevel, ID, assay, estimate, std.error, statistic, p.value) %>% group_by(assay) %>% mutate(FDR = p.adjust(p.value, method = "fdr"), log10FDR = log10(FDR))
crossDis_Tau <- meta %>% filter(AnnoLevel=='subclass' & coef=='m14x' & method=='FE') %>% filter(ID %in% pcg) %>% select(AnnoLevel, ID, assay, estimate, std.error, statistic, p.value) %>% group_by(assay) %>% mutate(FDR = p.adjust(p.value, method = "fdr"), log10FDR = log10(FDR))
crossDis_PD <- meta %>% filter(AnnoLevel=='subclass' & coef=='m15x' & method=='FE') %>% filter(ID %in% pcg) %>% select(AnnoLevel, ID, assay, estimate, std.error, statistic, p.value) %>% group_by(assay) %>% mutate(FDR = p.adjust(p.value, method = "fdr"), log10FDR = log10(FDR))
tbl = read_parquet('/path/to/topTable_combined.parquet')
# crossDis with one brain bank only, within FDR
crossDis_BD <- tbl %>% filter(Dataset!='AGING' & SampleLevel=='SubID' & AnnoLevel=='subclass' & coef=='c53xBD_crossDis - c53xControls_crossDis') %>% filter(ID %in% pcg) %>% mutate(estimate=logFC, std.error=logFC/t, statistic=t, p.value=P.Value) %>% select(AnnoLevel, ID, assay, estimate, std.error, statistic, p.value) %>% group_by(assay) %>% mutate(FDR = p.adjust(p.value, method = "fdr"), log10FDR = log10(FDR))
crossDis_FTD <- tbl %>% filter(Dataset!='AGING' & SampleLevel=='SubID' & AnnoLevel=='subclass' & coef=='c58xFTD_crossDis - c58xControls_crossDis') %>% filter(ID %in% pcg) %>% mutate(estimate=logFC, std.error=logFC/t, statistic=t, p.value=P.Value) %>% select(AnnoLevel, ID, assay, estimate, std.error, statistic, p.value) %>% group_by(assay) %>% mutate(FDR = p.adjust(p.value, method = "fdr"), log10FDR = log10(FDR))

# remove shared
excl_genes = read.csv("/path/to/PsychAD_mashr_shared.csv")
excl_genes_filtered <- lapply(excl_genes[, -1], function(x) excl_genes$X[x >= 0.01]) # use 0.01 to remove shared genes per subclass
                              
AD <- crossDis_AD %>% filter(!(ID %in% unlist(excl_genes_filtered))) %>% mutate(dx='AD')
SCZ <- crossDis_SCZ %>% filter(!(ID %in% unlist(excl_genes_filtered))) %>% mutate(dx='SCZ')
DLBD <- crossDis_DLBD %>% filter(!(ID %in% unlist(excl_genes_filtered))) %>% mutate(dx='DLBD')
Vas <- crossDis_Vas %>% filter(!(ID %in% unlist(excl_genes_filtered))) %>% mutate(dx='Vas')
Tau <- crossDis_Tau %>% filter(!(ID %in% unlist(excl_genes_filtered))) %>% mutate(dx='Tau')
PD <- crossDis_PD %>% filter(!(ID %in% unlist(excl_genes_filtered))) %>% mutate(dx='PD')
FTD <- crossDis_FTD %>% filter(!(ID %in% unlist(excl_genes_filtered))) %>% mutate(dx='FTD')
BD <- crossDis_BD %>% filter(!(ID %in% unlist(excl_genes_filtered))) %>% mutate(dx='BD')
                              
df.merged = rbind(AD,DLBD,Vas,PD,Tau,FTD,SCZ,BD)

df_list <- list(AD = AD, DLBD = DLBD, PD = PD, Vas = Vas, Tau = Tau, FTD = FTD, SCZ = SCZ, BD = BD)

calculate_spearman <- function(df1, df2) {
    
    common_genes <- union(df1$ID, df2$ID)
    df1_common <- df1 %>% filter(ID %in% common_genes) 
    df2_common <- df2 %>% filter(ID %in% common_genes) 
    
    merged_df <- merge(df1_common, df2_common, by = c("ID", "assay"), suffixes = c("_1", "_2"))

    # Calculate Spearman correlation for each assay
    correlations <- merged_df %>% group_by(assay) %>%
    summarise(correlation = cor(estimate_1, estimate_2, method = "spearman", use = "pairwise.complete.obs"))

    return(correlations)
}

correlation_matrix <- combn(names(df_list), 2, simplify = FALSE, FUN = function(x) {
  df1 <- df_list[[x[1]]]
  df2 <- df_list[[x[2]]]
  corr <- calculate_spearman(df1, df2)
  corr$pair <- paste(x[1], x[2], sep = "-")
  return(corr)
}) %>% bind_rows()

correlation_matrix_wide <- correlation_matrix %>% 
  pivot_wider(names_from = pair, values_from = correlation)

correlation_matrix_wide[is.na(correlation_matrix_wide)] <- 0

select_top_genes <- function(df) {
    top_genes <- df[df$FDR<=0.05,]$ID
    return(top_genes)
}

# Function to calculate the number of shared genes between two data frames
calculate_shared_genes <- function(df1, df2) {
    shared_genes <- intersect(df1, df2)
    return(length(shared_genes))
}

# Create a new column for the number of shared genes in the correlation_matrix dataframe
correlation_matrix$shared_genes <- sapply(1:nrow(correlation_matrix), function(i) {
    pair <- strsplit(as.character(correlation_matrix$pair[i]), "-")[[1]]
    df1 <- df_list[[pair[1]]]
    df2 <- df_list[[pair[2]]]

    # Select the top genes for each dataset
    top_genes_df1 <- select_top_genes(df1)
    top_genes_df2 <- select_top_genes(df2)

    # Calculate the number of shared genes between the datasets
    return(calculate_shared_genes(top_genes_df1, top_genes_df2))
})

extract_top_genes_per_assay <- function(df_list) {
  top_genes_per_assay <- list()
  
    for(disorder_name in names(df_list)) {
        disorder_df <- df_list[[disorder_name]]
    
        for(a in unique(disorder_df$assay)) {
            assay_df <- disorder_df %>% filter(assay == a)
            top_genes <- select_top_genes(assay_df)
            # print(assay_df)

            # If the assay already has an entry in the list, intersect, else initialize
            if(a %in% names(top_genes_per_assay)) {
                top_genes_per_assay[[a]] <- c(top_genes_per_assay[[a]], top_genes)
            } else {
                top_genes_per_assay[[a]] <- top_genes
            }
        }
    }
    return(top_genes_per_assay)
}

top_genes_per_assay <- extract_top_genes_per_assay(df_list)

common_genes_counts_row <- sapply(names(top_genes_per_assay), function(x){length(top_genes_per_assay[[x]][table(top_genes_per_assay[[x]])>=2])})

disorder_pairs <- combn(names(df_list), 2, simplify = FALSE) # Generates all possible pairs

calculate_common_genes_counts <- function(pair, df_list) {

    ### common FDR significant genes
    top_genes_df1 <- select_top_genes(df_list[[pair[1]]])
    top_genes_df2 <- select_top_genes(df_list[[pair[2]]])

    common_genes <- length(intersect(top_genes_df1, top_genes_df2))
    return(common_genes)
}

common_genes_counts_col <- sapply(disorder_pairs, calculate_common_genes_counts, df_list)

names(common_genes_counts_col) <- sapply(disorder_pairs, function(pair) paste(pair, collapse = "-"))

library(ape)
library(dendextend)

plot_tree_simple = function(tree, xmax.scale=1.5){

    fig.tree = ggtree(tree, branch.length = "none", ladderize=FALSE) + 
               geom_tiplab(color = "black", size=4, hjust=0, offset=.2) +
               theme(legend.position="top left", plot.title = element_text(hjust = 0.5))

    # get default max value of x-axis
    xmax = layer_scales(fig.tree)$x$range$range[2]

    # increase x-axis width
    fig.tree = fig.tree + xlim(0, xmax*xmax.scale) 
    
    return(fig.tree)
}

# load
tree = read.tree(file="/path/to/tree_subclass_um.nwk")

### tree
fig.tree = plot_tree_simple(as.phylo(tree), xmax.scale=1.5) + theme(legend.position="bottom")

### assay_order
assay_order = (ggtree::get_taxa_name(fig.tree))
                                         
# Fig. 5b
pal_df <- read.csv('/path/to/PsychAD_color_palette.csv')

pal_class <- pal_df[pal_df$category=='class',]$color_hex
names(pal_class) <- pal_df[pal_df$category=='class',]$name

pal_subclass <- pal_df[pal_df$category=='subclass',]$color_hex
names(pal_subclass) <- pal_df[pal_df$category=='subclass',]$name

subclass2class <- pal_df[pal_df$category=='subclass',]$parent
names(subclass2class) <- pal_df[pal_df$category=='subclass',]$name
                                         
# make corr matrix wide
correlation_matrix_wide <- tidyr::pivot_wider(correlation_matrix[,1:3], id_cols=assay, names_from = pair, values_from = correlation)
mat <- as.matrix(correlation_matrix_wide[,-1, drop = FALSE])
rownames(mat) <- correlation_matrix_wide$assay
mat[is.na(mat)] <- 0

# reorder mat
mat <- mat[assay_order,]

# Order rows as mat
row_ordered <- common_genes_counts_row[match(rownames(mat),names(common_genes_counts_row))]
class = subclass2class[assay_order]
row_ha <- HeatmapAnnotation(class = class, col = list(class=pal_class),
                            'Shared\nFDR\nGenes' = anno_barplot(row_ordered, gp = gpar(fill = "#E4844D"), axis = TRUE),
                            which = "row", gap = unit(2, "mm"))

# Order cols as mat
col_ordered <- common_genes_counts_col[match(colnames(mat), names(common_genes_counts_col))]
blocks = c('NDD','NDD','NDD','NDD','NDD','NDD-NPD','NDD-NPD','NDD','NDD','NDD','NDD','NDD-NPD','NDD-NPD','NDD','NDD','NDD','NDD-NPD','NDD-NPD','NDD','NDD','NDD-NPD','NDD-NPD','NDD','NDD-NPD','NDD-NPD','NDD-NPD','NDD-NPD','NPD')
column_ha <- HeatmapAnnotation('Shared FDR Genes' = anno_barplot(col_ordered, gp = gpar(fill = "#E4C04D"), axis = TRUE),
                               Pair = blocks, col = list(Pair = c('NDD'='#469844','NDD-NPD'='#FEC65F','NPD'='#EE93BE')),
                               which = "column", gap = unit(2, "mm"))

# Heatmap
ht_list = Heatmap(mat, 
                  top_annotation = column_ha, 
                  right_annotation = row_ha,
                  row_km = 1, column_km = 3,
                  cluster_rows = FALSE, cluster_columns = TRUE, 
                  row_dend_width = unit(10, "mm"),
                  column_dend_height = unit(10, "mm"),
                  gap = unit(2, "mm"), rect_gp = gpar(col = "white"), column_gap = unit(2, "mm"),
                  column_title = NULL, row_title = NULL,
                  name = "Spearman", col = colorRamp2(c(-0.75, 0, 0.75), c("blue", "white", "red")))

pdf(file="PsychAD_Figure5B_crossDis_dreamlet_heatmap.pdf", width=9.5, height=7)
draw(ht_list)
dev.off()
