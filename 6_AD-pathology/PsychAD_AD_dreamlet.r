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

prefix='PsychAD_F6_META_dreamlet'

# merge data frames
meta = read_parquet("/path/to/res_meta.parquet")
meta.dx = meta %>% filter(AnnoLevel == 'subclass', coef == 'm01x', method == 'FE') %>%  
            ungroup() %>% group_by(assay) %>% mutate(FDR = p.adjust(p.value, "fdr")) %>% mutate('log10FDR' = -log10(FDR)) %>% mutate('coef' = 'dx_AD')
meta.ce = meta %>% filter(AnnoLevel == 'subclass', coef == 'm19x', method == 'FE') %>%
            ungroup() %>% group_by(assay) %>% mutate(FDR = p.adjust(p.value, "fdr")) %>% mutate('log10FDR' = -log10(FDR)) %>% mutate('coef' = 'CERAD')
meta.br = meta %>% filter(AnnoLevel == 'subclass', coef == 'm18x', method == 'FE') %>%
            ungroup() %>% group_by(assay) %>% mutate(FDR = p.adjust(p.value, "fdr")) %>% mutate('log10FDR' = -log10(FDR)) %>% mutate('coef' = 'Braak')
meta.de = meta %>% filter(AnnoLevel == 'subclass', coef == 'm21x', method == 'FE') %>%
            ungroup() %>% group_by(assay) %>% mutate(FDR = p.adjust(p.value, "fdr")) %>% mutate('log10FDR' = -log10(FDR)) %>% mutate('coef' = 'Dementia')
merged <- rbind(meta.dx, meta.ce, meta.br, meta.de)

# subset for protein_coding only
pcg = read.csv('/path/to/PsychAD_rowData_34890.csv')
pcg = pcg[pcg$gene_type=='protein_coding','gene_name']
merged_pcg = merged %>% filter(ID %in% pcg)

# Fig. 6d
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

# top genes that are pval < x, |effect size| > y

FDR_threshold = 0.01
logFC_threshold = 0.5

df_subset <- merged_pcg %>% filter(FDR < FDR_threshold, abs(estimate) >= logFC_threshold, n.studies>=2)
g <- table(df_subset$ID)
g <- names(g[g>=4])
length(g)

assay_order = (ggtree::get_taxa_name(fig.tree))
assay_order = setdiff(assay_order, c('IN_PVALB_CHC','EN_L5_ET','Adaptive','SMC','PC','Endo'))

# merge data frames
rbind(meta.dx,meta.ce,meta.br,meta.de) %>%
filter(assay %in% assay_order) %>%
filter(ID %in% g) %>%
group_by(ID) %>% 
mutate(logFC.order = sum(estimate)) %>%
mutate(logFC = estimate) %>%
arrange(logFC.order) -> df_top

# reorder genes
df_top$ID <- as.character(df_top$ID)
df_top$ID <- factor(df_top$ID, levels=unique(df_top$ID))

# reorder assay
df_top$assay <- factor(df_top$assay, levels=rev(assay_order))

df_top <- df_top %>% mutate(xadj = ifelse(coef %in% c("dx_AD", "CERAD"), -0.5/2,  0.5/2), 
                            yadj = ifelse(coef %in% c("dx_AD", "Dementia"), 0.5/2, -0.5/2))
df_top$xpos = as.numeric(factor(df_top$ID)) + df_top$xadj
df_top$ypos = as.numeric(factor(df_top$assay)) + df_top$yadj

# plot
options(repr.plot.width=26, repr.plot.height=12)

df_top %>% ggplot() + 
geom_point(aes(x=xpos, y=ypos, color=logFC, size=log10FDR), shape=15) +
geom_text(data = df_top %>% filter(ID == "DPYD", assay == "Micro"),
          aes(label = coef, x = xpos + 6*xadj +0, y = ypos + yadj +25), 
          fontface = "bold", color="black", size=4) +
scale_fill_gradient2(labels = function(breaks) paste0(breaks)) +

scale_color_gradient2(high="#D7191C", mid="white", low="#2C7BB6", midpoint = 0, limits=c(-1.0,1.0), oob = scales::squish) +
scale_size_area(limits = c(0,2), max_size = 4, oob = scales::squish) +

scale_x_continuous(breaks = 1:length(unique(df_top$ID)), name = NULL, labels = levels(df_top$ID)) +
scale_y_continuous(breaks = 1:length(unique(df_top$assay)), name = NULL, labels = levels(df_top$assay)) +

labs(size = bquote(-log[10]~(FDR)), color = bquote(log[2]~(FC))) +
cowplot::theme_cowplot() +
theme(axis.line = element_blank(), axis.ticks = element_blank(), axis.text = element_text(size = 20), 
      axis.text.x = element_text(angle=-90, vjust=0.5, hjust=0), axis.text.y = element_text(margin = margin(r = -50))) -> fig.hm

fig.hm

ggsave(paste0(prefix,"_AD.svg"), width = 26, height = 12)
