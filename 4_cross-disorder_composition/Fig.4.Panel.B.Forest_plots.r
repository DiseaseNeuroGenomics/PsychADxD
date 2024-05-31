suppressPackageStartupMessages({
library(dreamlet)
library(SingleCellExperiment)
library(crumblr)
library(qvalue)
library(cowplot)
library(ggcorrplot)
library(knitr)
library(aplot)
library(ggtree)
library(RColorBrewer)
library(broom)
library(dplyr)
library(metafor)
library(tidyverse)
library(ggplot2)
})

meta_analysis = function(tabList) {
    #' Perform meta-analysis on a list of data tables
    #'
    #' @param tabList A list of data tables
    #' @return A tibble containing the results of the meta-analysis
    #' 
    if(is.null(names(tabList))) {
        names(tabList) = as.character(seq(length(tabList)))
    }

    for(key in names(tabList)) {
        tabList[[key]]$Dataset = key
    }

    df = do.call(rbind, tabList) 

    df %>% 
        as_tibble() %>% 
        group_by(assay) %>% 
        do({
            dat = .
            dat = dat %>% mutate(sei = ifelse(t == 0, NA, abs(logFC / t)))
            tidy(rma(yi = logFC, sei = sei, data = dat, method = "FE"))
        }) %>% 
        select(-term, -type) %>% 
        mutate(FDR = p.adjust(p.value, "fdr")) %>% 
        mutate(log10FDR = -log10(FDR))
}

plotCoef2 = function(tab, coef, fig.tree, sums_by_group, low="grey90", mid = "red", high="darkred", ylab){
    #' Create a forest plot for a given disorder
    #'
    #' @param tab Data table with meta-analysis results
    #' @param coef Coefficient name
    #' @param fig.tree Phylogenetic tree figure
    #' @param sums_by_group List of cell counts by group
    #' @param low Color for low values in gradient
    #' @param mid Color for mid values in gradient
    #' @param high Color for high values in gradient
    #' @param ylab Label for the y-axis
    #' @return A ggplot object representing the forest plot
    #' 
    if (!is.null(sums_by_group[[coef]])) {
        disease_specific_counts <- data.frame(assay = names(sums_by_group[[coef]]), column_sums = sums_by_group[[coef]])
        tab <- merge(tab, disease_specific_counts, by = "assay", all.x = TRUE)
      
        if ("column_sums" %in% names(tab)) {
            min_size <- 1  
            max_size <- 10 
            max_sqrt_sum <- max(sqrt(tab$column_sums), na.rm = TRUE)
            min_sqrt_sum <- min(sqrt(tab$column_sums), na.rm = TRUE)
            
            tab$size <- (sqrt(tab$column_sums) - min_sqrt_sum) / (max_sqrt_sum - min_sqrt_sum) * (max_size - min_size) + min_size
            tab$size <- ifelse(is.na(tab$size), 1, pmin(pmax(tab$size, min_size), max_size))
        } else {
            tab$size <- rep(1, nrow(tab))
        }
    } else {
        tab$size <- rep(1, nrow(tab))
    }

    tab$logFC = tab$estimate
    tab$celltype = factor(tab$assay, rev(ggtree::get_taxa_name(fig.tree)))
    tab$se = tab$std.error
    fig.es = ggplot(tab, aes(celltype, logFC)) + 
        geom_hline(yintercept=0, linetype="dashed", color="grey", linewidth=1) +
        geom_errorbar(aes(ymin = logFC - 1.96*se, ymax = logFC + 1.96*se), width=0) +
        geom_point2(aes(color=pmin(4, -log10(FDR)), size=size)) +
        scale_color_gradient2(name = "-log10 FDR", low=low, mid=mid, high=high, midpoint=-log10(0.01), limits=c(0, 4), breaks=seq(0, 4, 1)) +
        scale_size_continuous(name="Cell Counts", breaks=seq(1, 10, 1), limits=c(1, 10)) +
        geom_text2(aes(label = '+', subset=FDR < 0.05), color = "white", size=6, vjust=.3, hjust=.5) +
        theme_classic() +
        coord_flip() +
        xlab('') + 
        ylab(ylab) +
        theme(axis.text.y=element_blank(), axis.text=element_text(size = 12), axis.ticks.y=element_blank(), text = element_text(size = 14)) +
        scale_y_continuous(breaks = c(-1, 0, 1), limits = c(-1, 1), expand = c(0, 0))
    return(fig.es)    
}

# Load data for different disorders (replace 'your_path' with the appropriate relative or absolute path to your data files)
load_data <- function() {
    #' Load data for different disorders
    #' @return A list of data frames for each disorder
    #' 
    disorders <- c('AD', 'DLBD', 'SCZ', 'Vasc', 'Tau', 'PD', 'FTD', 'BD')
    data_frames <- list()
    
    for (disorder in disorders) {
        file_path <- sprintf("your_path/%s.tsv.gz", disorder)
        data_frames[[disorder]] <- read.delim(file_path)
    }
    
    names(data_frames) <- disorders
    return(data_frames)
}

data_frames <- load_data()
disorder_names <- names(data_frames)

for (i in 1:length(data_frames)) {
  colnames(data_frames[[i]])[2] <- "assay"
  data_frames[[i]] <- data_frames[[i]][, 2:ncol(data_frames[[i]])]
}

list2env(setNames(data_frames, disorder_names), envir = .GlobalEnv)

# Load cell count data
load_cellcounts <- function() {
    #' Load cell count data for different disorders
    #' @return A list of data frames with cell counts for each disorder
    #' 
    disorders <- c('AD', 'DLBD', 'SCZ', 'FTD', 'Tauopathy', 'PD', 'Vascular', 'BD')
    cellcounts <- list()
    
    for (disorder in disorders) {
        file_pattern <- sprintf("your_path/cellCounts_%s_subtype.tsv", disorder)
        files <- list.files(pattern = file_pattern)
        for (file in files) {
            df <- read.table(file, sep = '\t', header = TRUE, row.names = 1)
            cellcounts[[disorder]] <- df
        }
    }
    
    names(cellcounts)[names(cellcounts) == 'Tauopathy'] <- 'Tau'
    names(cellcounts)[names(cellcounts) == 'Vascular'] <- 'Vasc'
    
    return(cellcounts)
}

cellcounts <- load_cellcounts()

traits <- c('AD', 'DLBD', 'SCZ', 'FTD', 'Tau', 'PD', 'Vasc', 'BD')

group_map <- c("EN_L2_3_IT", "EN_L3_5_IT_1", "EN_L3_5_IT_2", "EN_L3_5_IT_3", "EN_L5_6_NP", "EN_L5_ET", 
               "EN_L6B", "EN_L6_CT", "EN_L6_IT_1", "EN_L6_IT_2", "Endo", "IN_ADARB2", "IN_LAMP5_LHX6", 
               "IN_LAMP5_RELN", "IN_PVALB", "IN_SST", "IN_VIP", "Micro", "OPC", "Oligo", "PC", "PVM", 
               "SMC", "VLMC", "Adaptive", "Astro")

sums_by_group <- list()

for (trait in traits) {
  if (!is.null(cellcounts[[trait]])) {
    trait_cc <- cellcounts[[trait]]
    
    trait_cc$group <- sapply(row.names(trait_cc), function(x) {
      g <- group_map[sapply(group_map, function(y) grepl(y, x))]
      if (length(g) > 0) g[1] else "Other"
    })

    sums_by_group[[trait]] <- tapply(trait_cc$column_sums, trait_cc$group, sum)
  } else {
    sums_by_group[[trait]] <- NULL
  }
}

calculate_average_per_assay <- function(sum_group, selected_traits) {
    #' Calculate average cell counts per assay for selected traits
    #'
    #' @param sum_group List of summed cell counts by group
    #' @param selected_traits Traits to consider for averaging
    #' @return A vector of average cell counts per assay
    #' 
    selected_data <- sum_group[selected_traits]
    assay_names <- unique(unlist(lapply(selected_data, names)))

    averages <- sapply(assay_names, function(assay) {
        mean_vals <- sapply(selected_data, function(trait) ifelse(!is.null(trait[[assay]]), trait[[assay]], NA))
        mean(mean_vals, na.rm = TRUE)
    })

    return(averages)
}

traits_to_consider <- c("BD", "SCZ")
averages <- calculate_average_per_assay(sums_by_group, traits_to_consider)
sums_by_group$dx_meta_FE_NDD <- averages

traits_to_consider <- c('AD', 'DLBD', 'FTD', 'Tau', 'PD', 'Vasc')
averages <- calculate_average_per_assay(sums_by_group, traits_to_consider)
sums_by_group$dx_meta_FE_NPD <- averages

meta_of_meta = function(tabList) {
    #' Perform meta-analysis on a list of meta-analysis results
    #'
    #' @param tabList A list of data tables containing meta-analysis results
    #' @return A tibble containing the results of the meta-of-meta analysis
    #' 
    if(is.null(names(tabList))) {
        names(tabList) = as.character(seq(length(tabList)))
    }

    for(key in names(tabList)) {
        tabList[[key]]$Dataset = key
    }

    df = do.call(rbind, tabList)

    df %>% 
        as_tibble() %>% 
        group_by(assay) %>% 
        do({
            dat = .
            dat = dat %>% mutate(sei = ifelse(statistic == 0, NA, abs(estimate / statistic)))
            tidy(rma(yi = estimate, sei = sei, data = dat, method = "REML"))
        }) %>% 
        select(-term, -type) %>% 
        mutate(FDR = p.adjust(p.value, "fdr")) %>% 
        mutate(log10FDR = -log10(FDR))
}

# Meta-analysis for each disorder pair
AD = meta_analysis(list(data_frames$AD, data_frames$AD))
DLBD = meta_analysis(list(data_frames$DLBD, data_frames$DLBD))
BD = meta_analysis(list(data_frames$BD))
FTD = meta_analysis(list(data_frames$FTD))
Vasc = meta_analysis(list(data_frames$Vasc, data_frames$Vasc))
Tau = meta_analysis(list(data_frames$Tau, data_frames$Tau))
PD = meta_analysis(list(data_frames$PD, data_frames$PD))
SCZ = meta_analysis(list(data_frames$SCZ, data_frames$SCZ))

# Meta-of-meta
dx_meta_FE = meta_of_meta(list(AD, DLBD, BD, FTD, Vasc, Tau, PD, SCZ))
dx_meta_FE_NDD = meta_of_meta(list(AD, DLBD, FTD, Vasc, Tau, PD))
dx_meta_FE_NPD = meta_of_meta(list(BD, SCZ))

# Load phylogenetic tree (replace 'your_tree_path' with the appropriate path to your tree file)
tree = read.tree(file="your_tree_path/tree_subclass_um.nwk")
hc = tree
fig.tree = plotTree(ape::as.phylo(hc), xmax.scale=2.2) + theme(legend.position="bottom")

fig.es1 = plotCoef2(AD, coef="AD", fig.tree, sums_by_group, ylab='AD')
fig.es2 = plotCoef2(DLBD, coef="DLBD", fig.tree, sums_by_group, ylab='DLBD')
fig.es4 = plotCoef2(BD, coef="BD", fig.tree, sums_by_group, ylab='BD')
fig.es5 = plotCoef2(FTD, coef="FTD", fig.tree, sums_by_group, ylab='FTD')
fig.es6 = plotCoef2(Vasc, coef="Vasc", fig.tree, sums_by_group, ylab='Vasc')
fig.es7 = plotCoef2(Tau, coef="Tau", fig.tree, sums_by_group, ylab='Tau')
fig.es8 = plotCoef2(PD, coef="PD", fig.tree, sums_by_group, ylab='PD')
fig.es9 = plotCoef2(SCZ, coef="SCZ", fig.tree, sums_by_group, ylab='SCZ')
fig.es10 = plotCoef2(dx_meta_FE, coef="dx_meta_FE", fig.tree, sums_by_group, ylab='Shared')
fig.es11 = plotCoef2(dx_meta_FE_NDD, coef="dx_meta_FE_NDD", fig.tree, sums_by_group, ylab='NDDs')
fig.es12 = plotCoef2(dx_meta_FE_NPD, coef="dx_meta_FE_NPD", fig.tree, sums_by_group, ylab='NPDs')

spacer <- ggplot() + theme_void()

forest_plots = fig.es9 %>% insert_left(fig.tree, width=2) %>% 
    insert_right(fig.es4, width=1) %>% 
    insert_right(spacer, width=0.5) %>%
    insert_right(fig.es7, width=1) %>%
    insert_right(fig.es8, width=1) %>%
    insert_right(fig.es2, width=1) %>% 
    insert_right(fig.es6, width=1) %>%
    insert_right(fig.es1, width=1) %>% 
    insert_right(fig.es5, width=1) %>% 
    insert_right(spacer, width=0.5) %>%
    insert_right(fig.es12, width=1) %>%
    insert_right(fig.es11, width=1) %>%
    insert_right(spacer, width=0.5)

forest_plots

