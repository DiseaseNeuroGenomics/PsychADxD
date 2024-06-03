suppressPackageStartupMessages({
  library(dreamlet)
  library(broom)
  library(metafor) 
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
})

# Load necessary data files
load("building_block.RData")
df <- read.csv("clinical_metadata_full.csv")

# Read processed data
pb.M <- readRDS("MSSM_PB_Channel_subclass.RDS")
pb.R <- readRDS("RUSH_PB_Channel_subclass.RDS")
pb.H <- readRDS("HBCC_PB_Channel_subclass.RDS")

# Harmonize column names
common_colnames <- intersect(intersect(colnames(colData(pb.M)), colnames(colData(pb.R))), colnames(colData(pb.H)))
colData(pb.M) <- colData(pb.M)[, common_colnames]
colData(pb.R) <- colData(pb.R)[, common_colnames]
colData(pb.H) <- colData(pb.H)[, common_colnames]

# Combine datasets
pb <- cbind(pb.M, pb.R, pb.H)

# Combine metadata
metadata(pb) <- metadata(pb.M)
metadata(pb)$aggr_means <- rbind(metadata(pb.M)$aggr_means, metadata(pb.R)$aggr_means, metadata(pb.H)$aggr_means)

# Add Channel variable
colData(pb)$Channel <- colnames(pb)

# Filter by SubID
pb <- pb[, (pb$SubID %in% df$SubID)]
pb$SubID <- factor(pb$SubID)

# Aggregate metadata
aggr_meta <- metadata(pb)$aggr_means %>%
  group_by(Channel) %>%
  summarise(n_genes = mean(n_genes, na.rm = TRUE), 
            percent_mito = mean(percent_mito, na.rm = TRUE),
            mito_genes = mean(mito_genes, na.rm = TRUE),
            ribo_genes = mean(ribo_genes, na.rm = TRUE),
            mito_ribo = mean(mito_ribo, na.rm = TRUE))

rownames(aggr_meta) <- aggr_meta[['Channel']]
colData(pb) <- cbind(colData(pb), aggr_meta)

# Create subsets
pb_CTRL <- pb[, !is.na(colData(pb)$c45x) & colData(pb)$c45x == 'Controls_crossDis']
pb_AD <- pb[, !is.na(colData(pb)$c45x) & (colData(pb)$c45x == 'AD_crossDis')]
pb_DLBD <- pb[, !is.na(colData(pb)$c49x) & (colData(pb)$c49x == 'DLBD_crossDis')]
pb_SCZ <- pb[, !is.na(colData(pb)$c47x) & (colData(pb)$c47x == 'SCZ_crossDis')]
pb_PD <- pb[, !is.na(colData(pb)$c56x) & (colData(pb)$c56x == 'PD_crossDis')]
pb_BD <- pb[, !is.na(colData(pb)$c53x) & (colData(pb)$c53x == 'BD_crossDis')]
pb_Tau <- pb[, !is.na(colData(pb)$c54x) & (colData(pb)$c54x == 'Tauopathy_crossDis')]
pb_Vasc <- pb[, !is.na(colData(pb)$c51x) & (colData(pb)$c51x == 'Vascular_crossDis')]
pb_FTD <- pb[, !is.na(colData(pb)$c58x) & (colData(pb)$c58x == 'FTD_crossDis')]

pb_NDD <- pb[, (!is.na(colData(pb)$c45x) & colData(pb)$c45x == 'AD_crossDis') | 
                 (!is.na(colData(pb)$c49x) & colData(pb)$c49x == 'DLBD_crossDis') |
                 (!is.na(colData(pb)$c56x) & colData(pb)$c56x == 'PD_crossDis') |
                 (!is.na(colData(pb)$c53x) & colData(pb)$c53x == 'BD_crossDis') |
                 (!is.na(colData(pb)$c54x) & colData(pb)$c54x == 'Tauopathy_crossDis') |
                 (!is.na(colData(pb)$c51x) & colData(pb)$c51x == 'Vascular_crossDis') |
                 (!is.na(colData(pb)$c58x) & colData(pb)$c58x == 'FTD_crossDis')]

pb_NPD <- pb[, (!is.na(colData(pb)$c47x) & colData(pb)$c47x == 'SCZ_crossDis') | 
                 (!is.na(colData(pb)$c53x) & colData(pb)$c53x == 'BD_crossDis')]

# Define functions for model fitting and data preparation

#' Prepare data for analysis
#'
#' @param pb A SingleCellExperiment object
#' @param subtype A character string representing the subtype to analyze
#' @return A data frame with the prepared data
prepare_data <- function(pb, subtype) {
  counts <- cellCounts(pb)
  cobj <- crumblr(counts)
  y <- t(cobj$E[subtype, , drop = FALSE])
  w <- t(cobj$weights[subtype, , drop = FALSE])
  w <- w / mean(w)
  data <- as.data.frame(colData(pb))
  data$y <- y
  data$w <- w
  return(data)
}

#' Fit a linear mixed-effects model and compare coefficients
#'
#' @param pb_disorder A SingleCellExperiment object for the disorder group
#' @param pb_control A SingleCellExperiment object for the control group
#' @param subtype A character string representing the subtype to analyze
#' @return A numeric value representing the difference in coefficients between disorder and control
res_fit <- function(pb_disorder, pb_control, subtype) {
  if (!inherits(pb_disorder, "SingleCellExperiment") || !inherits(pb_control, "SingleCellExperiment")) {
    stop("Both pb_disorder and pb_control must be SingleCellExperiment objects")
  }
  
  data_disorder <- prepare_data(pb_disorder, subtype)
  data_control <- prepare_data(pb_control, subtype)
  
  data_combined <- rbind(data_disorder, data_control)
  data_combined$Group <- c(rep("Disorder", nrow(data_disorder)), rep("Control", nrow(data_control)))
  
  # Fit the model
  model <- lmerTest::lmer(y ~ Group + Age + (1 | SubID) + (1 | pool) + (1 | prep) + (1 | Ethnicity) + PMI + (1 | Sex), data = data_combined, weights = data_combined$w)
  
  # Extract coefficient for 'GroupDisorder' which indicates difference from Control
  coef_disorder_control_diff <- summary(model)$coefficients['GroupDisorder', 'Estimate']
  return(coef_disorder_control_diff)
}

subtypes <- c("IN_LAMP5_RELN", "EN_L2_3_IT", "IN_SST", "EN_L3_5_IT_1", 
              "EN_L3_5_IT_3", "EN_L5_6_NP", "EN_L5_ET", "EN_L6B", "EN_L6_CT", "EN_L6_IT_1", "EN_L6_IT_2", 
              "IN_ADARB2", "IN_LAMP5_LHX6", "IN_PVALB", "IN_PVALB_CHC", "IN_SST", "IN_VIP", "Astro", "OPC", 
              "Oligo", "Micro", "PVM", "Adaptive", "VLMC", "SMC", "PC", "Endo")

disorders <- c("NDD", "NPD")

coefficients <- sapply(subtypes, function(subtype) {
  sapply(disorders, function(disorder) {
    pb_disorder <- get(paste0("pb_", disorder))
    pb_control <- get("pb_CTRL")
    res_fit(pb_disorder, pb_control, subtype)
  })
}, simplify = "array")

coef_df <- as.data.frame(t(coefficients))
names(coef_df) <- disorders
coef_df$Assay <- subtypes

long_coef_df <- gather(coef_df, key = "Disorder", value = "Adjusted_Coefficient", -Assay)

# Plotting the results
ggplot(long_coef_df, aes(x = Assay, y = Adjusted_Coefficient, fill = Disorder)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c("NDD" = "#D7191C", "NPD" = "#2B83BA")) +
  labs(title = "Comparison of Adjusted Coefficients Across Disorders", x = "Assay", y = "Adjusted Coefficient of Age")



