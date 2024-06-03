suppressPackageStartupMessages({
  library(dreamlet)
  library(SingleCellExperiment)
  library(tidyverse)
  library(crumblr)
  library(qvalue)
  library(cowplot)
  library(ggcorrplot)
  library(knitr)
  library(aplot)
  library(ggtree)
  library(RColorBrewer)
  library(broom)
})

# Load data from specified files
# Update these paths with actual file locations
file_paths <- list(
  MSSM_AD = "path/to/MSSM_AD.tsv.gz",
  RUSH_AD = "path/to/RUSH_AD.tsv.gz",
  MSSM_DLBD = "path/to/MSSM_DLBD.tsv.gz",
  RUSH_DLBD = "path/to/RUSH_DLBD.tsv.gz",
  MSSM_SCZ = "path/to/MSSM_SCZ.tsv.gz",
  HBCC_SCZ = "path/to/HBCC_SCZ.tsv.gz",
  MSSM_Vasc = "path/to/MSSM_Vasc.tsv.gz",
  RUSH_Vasc = "path/to/RUSH_Vasc.tsv.gz",
  MSSM_Tau = "path/to/MSSM_Tau.tsv.gz",
  RUSH_Tau = "path/to/RUSH_Tau.tsv.gz",
  MSSM_PD = "path/to/MSSM_PD.tsv.gz",
  RUSH_PD = "path/to/RUSH_PD.tsv.gz",
  MSSM_FTD = "path/to/MSSM_FTD.tsv.gz",
  HBCC_BD = "path/to/HBCC_BD.tsv.gz"
)

# Function to read data
read_data <- function(paths) {
  lapply(paths, read.delim)
}

# Read all data files
data_frames <- read_data(file_paths)
disorder_names <- names(file_paths)

# Rename and clean columns in data frames
for (i in seq_along(data_frames)) {
  colnames(data_frames[[i]])[2] <- "assay"
  data_frames[[i]] <- data_frames[[i]][, -1]
}

list2env(setNames(data_frames, disorder_names), envir = .GlobalEnv)

#' Perform meta-analysis on a list of data frames
#'
#' @param tabList List of data frames to perform meta-analysis on
#' @return Data frame with meta-analysis results
meta_analysis <- function(tabList) {
  library(dplyr)
  library(metafor)
  
  if(is.null(names(tabList))) {
    names(tabList) <- as.character(seq_along(tabList))
  }

  for (key in names(tabList)) {
    tabList[[key]]$Dataset <- key
  }

  df <- bind_rows(tabList)

  df %>%
    group_by(assay) %>%
    do({
      dat <- .
      dat <- dat %>% mutate(sei = ifelse(t == 0, NA, abs(logFC / t)))
      tidy(rma(yi = logFC, sei = sei, data = dat, method = "FE"))
    }) %>%
    select(-term, -type) %>%
    mutate(FDR = p.adjust(p.value, "fdr")) %>%
    mutate(log10FDR = -log10(FDR))
}

#' Filter data frame for specified subtypes
#'
#' @param trait_df Data frame to filter
#' @param subtypes Vector of subtype names to filter for
#' @return Filtered data frame
filter_subtypes <- function(trait_df, subtypes) {
  trait_df %>% filter(assay %in% subtypes)
}

# Define subtypes of interest
subtypes <- c("VLMC_ABCA6", "VLMC_DCDC2", "VLMC_SLC4A4", "SMC_MYOCD", "SMC_NRP1", "PC_ADAMTS4", "PC_STAC")

# Perform meta-analysis for each disorder
AD <- meta_analysis(list(MSSM_AD, RUSH_AD))
DLBD <- meta_analysis(list(MSSM_DLBD, RUSH_DLBD))
BD <- meta_analysis(list(HBCC_BD))
FTD <- meta_analysis(list(MSSM_FTD))
Vasc <- meta_analysis(list(MSSM_Vasc, RUSH_Vasc))
Tau <- meta_analysis(list(MSSM_Tau, RUSH_Tau))
PD <- meta_analysis(list(MSSM_PD, RUSH_PD))
SCZ <- meta_analysis(list(MSSM_SCZ, HBCC_SCZ))

# Filter data frames for the specified subtypes
AD_filtered <- filter_subtypes(AD, subtypes)
DLBD_filtered <- filter_subtypes(DLBD, subtypes)
BD_filtered <- filter_subtypes(BD, subtypes)
FTD_filtered <- filter_subtypes(FTD, subtypes)
Vasc_filtered <- filter_subtypes(Vasc, subtypes)
Tau_filtered <- filter_subtypes(Tau, subtypes)
PD_filtered <- filter_subtypes(PD, subtypes)
SCZ_filtered <- filter_subtypes(SCZ, subtypes)

# Combine filtered data for heatmap
filtered_data <- list(
  AD = AD_filtered, DLBD = DLBD_filtered, BD = BD_filtered,
  FTD = FTD_filtered, Vasc = Vasc_filtered, Tau = Tau_filtered,
  PD = PD_filtered, SCZ = SCZ_filtered
)

combined_data <- bind_rows(
  AD_filtered %>% mutate(trait = "AD"),
  DLBD_filtered %>% mutate(trait = "DLBD"),
  BD_filtered %>% mutate(trait = "BD"),
  FTD_filtered %>% mutate(trait = "FTD"),
  Vasc_filtered %>% mutate(trait = "Vasc"),
  Tau_filtered %>% mutate(trait = "Tau"),
  PD_filtered %>% mutate(trait = "PD"),
  SCZ_filtered %>% mutate(trait = "SCZ")
)

# Prepare data for heatmap
combined_data <- combined_data %>% mutate(log10FDR = -log10(FDR))
row_order <- c("VLMC_ABCA6", "VLMC_DCDC2", "VLMC_SLC4A4", "SMC_MYOCD", "SMC_NRP1", "PC_ADAMTS4", "PC_STAC")

# Create data matrix for heatmap
data_matrix <- combined_data %>%
  select(assay, trait, estimate) %>%
  spread(trait, estimate) %>%
  column_to_rownames("assay") %>%
  .[row_order,]

data_matrix <- data_matrix %>% select(SCZ, BD, Tau, PD, DLBD, Vasc, AD, FTD)

# Create significance matrix
sig_matrix <- combined_data %>%
  select(assay, trait, log10FDR) %>%
  mutate(sig = ifelse(log10FDR > -log10(0.05), 1, 0)) %>%
  select(-log10FDR) %>%
  spread(trait, sig) %>%
  column_to_rownames("assay") %>%
  .[row_order,]

sig_matrix <- sig_matrix %>% select(SCZ, BD, Tau, PD, DLBD, Vasc, AD, FTD)

row_split <- rep(1:3, c(3, 2, 2))
row_gaps <- unit(c(0.5, 0.5), "cm")

# Draw heatmap
heatmap <- Heatmap(data_matrix,
                   name = "logFC",
                   col = colorRamp2(c(min(combined_data$estimate), 0, max(combined_data$estimate)), c("blue", "white", "red")),
                   row_split = row_split,
                   row_gap = row_gaps,
                   cluster_rows = FALSE,
                   cluster_columns = FALSE,
                   cell_fun = function(j, i, x, y, width, height, fill) {
                     if (sig_matrix[i, j] == 1) {
                       grid.text("+", x, y, gp = gpar(fontsize = 20, col = "black"))
                     }
                   })

draw(heatmap)

#' @keywords internal
NULL

#' @keywords internal
NULL

#' @param tabList List of data frames to perform meta-analysis on
#' @return Data frame with meta-analysis results
meta_analysis <- function(tabList) {
  library(dplyr)
  library(metafor)
  
  if(is.null(names(tabList))) {
    names(tabList) <- as.character(seq_along(tabList))
  }

  for (key in names(tabList)) {
    tabList[[key]]$Dataset <- key
  }

  df <- bind_rows(tabList)

  df %>%
    group_by(assay) %>%
    do({
      dat <- .
      dat <- dat %>% mutate(sei = ifelse(t == 0, NA, abs(logFC / t)))
      tidy(rma(yi = logFC, sei = sei, data = dat, method = "FE"))
    }) %>%
    select(-term, -type) %>%
    mutate(FDR = p.adjust(p.value, "fdr")) %>%
    mutate(log10FDR = -log10(FDR))
}

#' Filter data frame for specified subtypes
#'
#' @param trait_df Data frame to filter
#' @param subtypes Vector of subtype names to filter for
#' @return Filtered data frame
filter_subtypes <- function(trait_df, subtypes) {
  trait_df %>% filter(assay %in% subtypes)
}

