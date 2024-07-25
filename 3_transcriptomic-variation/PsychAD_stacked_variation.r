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
library(tidyverse)

# versions
sessionInfo()
print(paste0('dreamlet v',packageVersion("dreamlet")))
print(paste0('crumblr v',packageVersion("crumblr")))
print(paste0('variancePartition v',packageVersion("variancePartition")))
print(paste0('zenith v',packageVersion("zenith")))
print(paste0('zellkonverter v',packageVersion("zellkonverter")))
print(paste0('BiocManager v',BiocManager::version()))

prefix = 'PsychAD_F3_FULL_dx_Channel'

pb.stack <- readRDS(paste0(prefix,'.pb.stack.rds'))

# voom-style normalization
res.proc <- processAssays(pb.stack, ~ (1|stackedAssay) + (1|Channel) + (1|SubID) + (1|Source) + (1|Ethnicity) + dx_bit + scale(Age) + Sex + scale(PMI) + log(n_genes) + percent_mito + mito_genes + ribo_genes + mito_ribo, BPPARAM=SnowParam(2, progressbar=FALSE))

# save
saveRDS(res.proc, paste0(prefix,'.res.proc.rds'))

# show voom plot for each cell clusters
plotVoom(res.proc)

# variance partitioning analysis
vp <- fitVarPart(res.proc, ~ (1|stackedAssay) + (1|Channel) + (1|SubID) + (1|Source) + (1|Ethnicity) + dx_bit + scale(Age) + (1|Sex) + scale(PMI) + log(n_genes) + percent_mito + mito_genes + ribo_genes + mito_ribo, BPPARAM=SnowParam(2, progressbar=FALSE))

# save
saveRDS(vp, paste0(prefix,'.vp.rds'))

# rename colnames
colnames(vp)[colnames(vp) == "stackedAssay"] = 'CellType'
colnames(vp)[colnames(vp) == "SubID"] = 'BrainDonor'
colnames(vp)[colnames(vp) == "Source"] = 'BrainSource'
colnames(vp)[colnames(vp) == "Channel"] = 'TechRep'
colnames(vp)[colnames(vp) == "scale.Age."] = 'Age'
colnames(vp)[colnames(vp) == "Ethnicity"] = 'Ancestry'
colnames(vp)[colnames(vp) == "scale.PMI."] = 'PMI'
colnames(vp)[colnames(vp) == "dx_bit"] = 'Diagnosis'
colnames(vp)[colnames(vp) == "log.n_genes."] = 'n_genes'

# mean fraction of variance
apply(sortCols(vp)[,c(-1,-2)],2,mean)

# protein_coding only
rd <- read.csv('/path/to/PsychAD_rowData_34890.csv')
pcg <- rd$gene_name[rd$gene_type=='protein_coding']
vp[vp$gene %in% pcg,] -> vp_pcg

# Summarize variance fractions across cell types
vp.pcg.sub <- vp_pcg[,c('assay','gene','TechRep','Ancestry','Sex','BrainSource','CellType','BrainDonor','Diagnosis','Age','PMI','Residuals')]

# Fig. 3a
plotVarPart <- function(obj, col, label.angle = 45, convertToPercent = TRUE, ylim) {
  # convert to data.frame
  obj <- as.data.frame(obj[,c(-1)])

  # get gene name of each row
  obj$gene <- rownames(obj)

  # convert to data.frame for ggplot
  data <- reshape2::melt(obj, id = "gene")

  if (min(data$value) < 0) {
    warning("Some values are less than zero")
  }

  if (convertToPercent) {
    data$value <- data$value * 100

    if (missing(ylim)) {
      ylim <- c(0, 100)
    }
  } else {
    if (missing(ylim)) {
      ylim <- c(0, max(data$value))
    }
  }

  # violin plot
  fig <- ggplot(data = data, aes(x = variable, y = value)) +
    
    geom_violin(scale = "width", aes(fill = factor(variable))) +
    geom_boxplot(width = 0.07, fill = "grey", outlier.color = NULL, aes(color = factor(variable))) +
    geom_boxplot(width = 0.07, fill = "grey", outlier.shape = NA) +
    scale_fill_manual(values = col) +
    scale_color_manual(values = col) +

    xlab("") +
    ylab("Variance explained (%)") +
    
    theme_bw() +
    theme(legend.position = "none",
          legend.text = element_text(colour = "black"),
          plot.title = element_text(hjust = 0.5),
          axis.title = element_text(colour = "black", size = 13),
          axis.text = element_text(colour = "black", size = 13),
          axis.text.x = element_text(angle = label.angle, hjust = 1, vjust = 1),
          axis.line = element_line(colour = "black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank(),
          panel.background = element_blank(),
          plot.margin = unit(c(0, .3, 0, .8), "cm")) +
    guides(fill = guide_legend(title = NULL))

  return(fig)
}

vp.pcg.sub <- vp_pcg[,c('assay','gene','TechRep','Ancestry','Sex','BrainSource','CellType','BrainDonor','Diagnosis','Age','PMI','Residuals')]

require(RColorBrewer)
col = c(colorRampPalette(brewer.pal(11,"Spectral"))(ncol(vp.pcg.sub)-3), "grey85")

options(repr.plot.width=6, repr.plot.height=6)
plotVarPart(sortCols(vp.pcg.sub), col=col) + theme(aspect.ratio=1)

ggsave(paste0(prefix,"_pb.stack.varPart.PCG.subset.pdf"), width = 6, height = 6)

# Fig. 3b
vp.pcg.sub <- vp_pcg[,c('assay','gene','TechRep','Ancestry','Sex','BrainSource','CellType','BrainDonor','Diagnosis','Age','PMI','Residuals')]

vp.pcg.sub %>% 
as_tibble %>% 
reframe(CellType = gene[order(CellType, decreasing=TRUE)[1:3]],
        BrainDonor = gene[order(BrainDonor, decreasing=TRUE)[1:3]],
        BrainSource = gene[order(BrainSource, decreasing=TRUE)[1:3]],
        TechRep = gene[order(TechRep, decreasing=TRUE)[1:3]],
        Age = gene[order(Age, decreasing=TRUE)[1:3]],
        Sex = gene[order(Sex, decreasing=TRUE)[1:3]],
        Ancestry = gene[order(Ancestry, decreasing=TRUE)[1:3]],
        Diagnosis = gene[order(Diagnosis, decreasing=TRUE)[1:3]],
        PMI = gene[order(PMI, decreasing=TRUE)[1:3]],
        Residuals = gene[order(Residuals, decreasing=TRUE)[1:3]]) -> genes
genes <- unlist(genes)

df.vp.sub = sortCols(vp.pcg.sub)[(vp.pcg.sub$gene %in% genes),]
df.vp.sub$gene = factor(df.vp.sub$gene, rev(genes))
df.vp.sub = df.vp.sub[order(df.vp.sub$gene, decreasing=TRUE),]
 
require(RColorBrewer)
col = c(colorRampPalette(brewer.pal(11,"Spectral"))(ncol(vp.pcg.sub)-3), "grey85")

options(repr.plot.width=8, repr.plot.height=6)
fig.percent = plotPercentBars(df.vp.sub, col=col) + theme(aspect.ratio=1)
fig.percent

ggsave(paste0(prefix,"_pb.stack.varPart.PCG.topGenes.pdf"), width = 8, height = 6)

# Fig. 3c
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

pal_df <- read.csv('/path/to/PsychAD_color_palette.csv')

pal_subclass <- pal_df[pal_df$category=='subclass',]$color_hex
names(pal_subclass) <- pal_df[pal_df$category=='subclass',]$name

# extract data
df = extractData(res.proc, assay="stacked")
df <- df %>% mutate(assay = stackedAssay)

gene = genes[['CellType1']]
varname = "CellType"

options(repr.plot.width=12, repr.plot.height=6)

df_sub <- df %>% dplyr::select(c("SubID","Age","PMI","CERAD","BRAAK_AD","dx","dx_bit","prep","Sex","Ethnicity","Source","assay",all_of(gene)))
colnames(df_sub)[colnames(df_sub)==gene] = "expr"

df_ord = df_sub %>%
  group_by(assay) %>%
  summarize(mean = mean(expr)) %>%
  arrange(mean)

df_sub %>%
  mutate(CellType = factor(assay, df_ord$assay)) %>% 
  ggplot(aes(CellType, expr, fill=factor(assay, assay_order))) +
    geom_boxplot(lwd=0.1, outlier.size=0.1) +
    annotate("text", x=7, y=12, label= "CellType: 96.3%", size=8) +
    # scale_colour_manual(values = pal_subclass[levels(factor(df$assay))], name = "subclass") +
    scale_fill_manual(values = pal_subclass[levels(factor(df$assay))], name = "subclass") +
    theme_classic() +
    theme(aspect.ratio=0.8,
          plot.title = element_text(hjust = 0.5, size=16),
          legend.position="right", 
          axis.text.x = element_blank(), 
          axis.ticks.x = element_blank(),
          axis.text.y = element_text(size=14),
          axis.title = element_text(size=16)) +
    guides(fill=guide_legend(ncol=3)) +
    ggtitle(gene) +
    ylab(bquote(log[2]~CPM)) +
    xlab(varname) -> p.CellType

p.CellType

ggsave(paste0(prefix,"_pb.stack.varPart.PCG.HVG.ATP8A2.pdf"), width = 12, height = 6)

# Fig. 3d
gene = genes[['BrainDonor1']]
varname = "BrainDonor"

options(repr.plot.width=12, repr.plot.height=6)

df_sub <- df %>% dplyr::select(c("SubID","Age","PMI","CERAD","BRAAK_AD","dx","dx_bit","prep","Sex","Ethnicity","Source","assay",all_of(gene)))
colnames(df_sub)[colnames(df_sub)==gene] = "expr"

df_ord = df_sub %>%
  group_by(SubID) %>%
  summarize(order = median(expr)) %>%
  arrange(order)

df_sub %>%
  mutate(BrainDonor = factor(SubID, df_ord$SubID)) %>%
  group_by(BrainDonor) %>% mutate(median_expr=median(expr)) %>%
  dplyr::select(BrainDonor, assay, expr, median_expr) %>%
  ggplot(aes(BrainDonor, expr)) +
    geom_boxplot(fill = NA, outlier.size=0.1, lwd=0.1, alpha=0.1) +
    geom_point(aes(BrainDonor, median_expr), size=0.1) +
    annotate("text", x=400, y=8.5, label= "BrainDonor: 80.9%", size=8) +
    theme_classic() +
    theme(aspect.ratio=0.8,
          plot.title = element_text(hjust = 0.5, size=16),
          legend.position="right", 
          axis.text.x = element_blank(), 
          axis.ticks.x = element_blank(),
          axis.text.y = element_text(size=14),
          axis.title = element_text(size=16)) +
    ggtitle(gene) +
    ylab(bquote(log[2]~CPM)) +
    xlab(varname) -> p.BrainDonor

p.BrainDonor

ggsave(paste0(prefix,"_pb.stack.varPart.PCG.HVG.ARL17B.pdf"), width = 12, height = 6)

# genetic constraints
n=2000

df_vp %>% head(n) -> top
top$group <- 'High'

df_vp %>% tail(n) -> bot
bot$group <- 'Low'

# df_vp %>% tail(nrow(.)-n) %>% head(nrow(.)-n) %>% sample_n(n) %>% arrange(desc(BrainDonor)) -> bg
df_vp %>% sample_n(n, seed=777) %>% arrange(desc(BrainDonor)) -> bg
bg$group <- 'Random'

df_vp_subset <- rbind(top,bg,bot)
df_vp_subset$group <- factor(df_vp_subset$group, levels=c('Low','Random','High'))

# gnomAD v4 constraint metrics
tab <- read.csv('/path/to/gnomad.v4.0.constraint_metrics.tsv',sep='\t')
tab <- tab[,c('gene','transcript','mane_select','lof.oe','lof.pLI','lof.oe_ci.upper')] %>% drop_na()

tab %>% dplyr::select(c(gene,lof.oe,lof.pLI,lof.oe_ci.upper)) %>% drop_na %>% group_by(gene) %>% summarize(mean.lof.oe = mean(lof.oe, na.rm = TRUE),
                                                                                    mean.lof.pLI = mean(lof.pLI, na.rm = TRUE),
                                                                                    mean.loeuf = mean(lof.oe_ci.upper, na.rm = TRUE)) -> df_lof

# merge between vp and gnomAD
df_merged <- merge(df_vp_subset, df_lof, by='gene', sort=FALSE)

# LOEUF: upper bound of 90% confidence interval for o/e ratio for high confidence pLoF variants (lower values indicate more constrained)
library(ggpubr)
options(repr.plot.width=4, repr.plot.height=8)

# Visualize: Specify the comparisons you want
my_comparisons <- list(c("Random","High"),c("Low","Random"),c("Low","High"))

# Fig. 3f
ggboxplot(df_merged, x = "group", y = "mean.loeuf", fill = "group", palette = "Set2", add = "jitter", shape=".") +
    stat_compare_means(method = 'wilcox.test', comparisons = my_comparisons, label = "p.signif", step.increase = 0.08, size=6) + # Add pairwise comparisons p-value
    xlab('Variance of BrainDonor') + 
    ylab('gnomAD genetic constraint (mean LOEUF)') +
    theme(legend.position="none",
          axis.text.x = element_text(size=16, angle = 45, hjust = 1, vjust = 1),
          axis.text.y = element_text(size=16),
          axis.title = element_text(size=18)) -> p.loeuf.box

p.loeuf.box
ggsave(paste0(prefix,"_pb.stack.mean.loeuf.box.pdf"), width = 4, height = 6)

# enrichment
plot_gs_heatmap <- function(gs, assay_order, title, ntop=5, pmax=10){
    
    # combine data frames
    df_list = lapply(gs, function(x){as.data.frame(x)})
    assay_list = names(gs)

    for (i in seq_along(df_list)) {
        if(nrow(df_list[[i]])>0){
            df_list[[i]]$assay <- assay_list[[i]]
        }
    }
    df <- do.call(rbind, df_list)
    df <- df %>% mutate('log10FDR' = -log10(p.adjust))

    # filter gs
    gs = list()
    for(x in assay_order){
            df_sub <- df %>% filter(assay==x)
            p_sort = rev(sort(df_sub$p.adjust))
            cutoff = tail(p_sort, ntop)[1]

            # keep genesets with highest and lowest t-statistics
            idx = (df_sub$p.adjust <= cutoff) 
            gs = append(gs, df_sub$ID[idx])
        }
    gs = unique(unlist(gs))
    df <- df %>% filter(ID %in% gs)

    # create matrix from gene sets
    M = reshape2::dcast(df, assay ~ Description, value.var = "p.adjust")
    annot = M[,seq(1)]
    M = as.matrix(M[,-seq(1)])
    rownames(M) = annot
    M[is.na(M)]<-1
    hcl2 <- hclust(dist(t(M)))

    # sortByGeneset
    df$Description = factor(df$Description, hcl2$labels[hcl2$order])

    # sort assay
    df$assay = factor(df$assay, levels=assay_order)

    g <- ggplot(df, aes(assay, Description, fill=log10FDR, label=ifelse(p.adjust < 0.05, '*', ''))) + 
         geom_tile() + 
         theme_classic() + 
         scale_fill_gradient2("-log10(FDR)", low="white", high="red", limits=c(0, pmax), oob = scales::squish) + 
         theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1), plot.title = element_text(hjust = 0.5)) + 
         geom_text(vjust=1, hjust=0.5) + 
         ylab("") + xlab("") +
         ggtitle(title) +
         scale_x_discrete(breaks=assay_order, labels=assay_order)
    
    return(g)
}

library(clusterProfiler)
library("org.Hs.eg.db")
library(enrichplot)
library(msigdbr)

m_gs_HOUSEKEEPING <- msigdbr(species = "Homo sapiens", category = "C2") %>% dplyr::select(gs_name, gene_symbol) %>% filter(stringr::str_detect(gs_name, 'HOUSEKEEPING_GENES'))
m_gs_REACTOME <- msigdbr(species = "Homo sapiens", category = "C2") %>% dplyr::select(gs_name, gene_symbol) %>% filter(stringr::str_detect(gs_name, 'REACTOME'))
m_gs_merged <- rbind(m_gs_HOUSEKEEPING,m_gs_REACTOME)
m_gs_merged$gs_name <- gsub('REACTOME_','',m_gs_merged$gs_name)

df_subset <- df_merged

options(enrichment_force_universe = FALSE)

go_enrich_list = list()

em.h <- enricher(gene = df_subset[df_subset$group=='High','gene'],
               pvalueCutoff = 1,
               pAdjustMethod = "BH",
               universe = pcg,
               minGSSize = 1,
               maxGSSize = 2000,
               TERM2GENE=m_gs_merged)
go_enrich_list[['High']] = em.h

em.l <- enricher(gene = df_subset[df_subset$group=='Low','gene'],
               pvalueCutoff = 1,
               pAdjustMethod = "BH",
               universe = pcg,
               minGSSize = 1,
               maxGSSize = 2000,
               TERM2GENE=m_gs_merged)
go_enrich_list[['Low']] = em.l

em.r <- enricher(gene = df_subset[df_subset$group=='Random','gene'],
               pvalueCutoff = 1,
               pAdjustMethod = "BH",
               universe = pcg,
               minGSSize = 1,
               maxGSSize = 2000,
               TERM2GENE=m_gs_merged)
go_enrich_list[['Random']] = em.r

# Fig. 3g
options(repr.plot.width=12, repr.plot.height=8)
p.enrich <- plot_gs_heatmap(gs=go_enrich_list, assay_order=c('Low','Random','High'), title='Functional Enrichment', ntop=7, pmax=10) +
    theme(axis.text.x = element_text(size=16, angle = 45, hjust = 1, vjust = 1),
          axis.text.y = element_text(size=16),
          axis.title = element_text(size=18),
          plot.title = element_text(size=18)) +
    scale_y_discrete(limits=rev)

p.enrich
ggsave(paste0(prefix,"_pb.stack.mean.loeuf.pathway.pdf"), width = 12, height = 8)
