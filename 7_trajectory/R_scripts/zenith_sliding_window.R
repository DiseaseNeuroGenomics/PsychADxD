### Calculates the Zenith pathway enrichment for each individual donor

library(GSEABase)
library(zenith)

go.gs.bp = get_GeneOntology(onto="BP", to="SYMBOL")
data_dir = "../processed_data/zenith_input_sliding_window/"
output_dir = "../processed_data/zenith_output_sliding_window/"
cells = c("Immune")

for (i in 1:1){
    fn = paste0(data_dir, cells[i], ".csv")
    df = read.csv(fn)
    n_time_pts = (ncol(df) - 2) / 2
    for (j in 1:n_time_pts){
        print(paste0(cells[i], " donor ", j))
        res = df[,2+j]
        names(res) = df[,2]
        res.gsa = zenithPR_gsa(statistics=res, ids=names(res), geneSets=go.gs.bp,  progressbar=FALSE, use.ranks = FALSE, n_genes_min = 10)
        save_fn = paste0(output_dir, cells[i], "_braak_time", j, ".csv")
        write.csv(res.gsa, save_fn)
    }
}

