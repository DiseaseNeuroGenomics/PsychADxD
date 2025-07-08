### Calculates the Zenith pathway enrichment for early and late epochs, both for changes with respect to Braak, and
### for protective/damaging

library(GSEABase)
library(zenith)

go.gs.bp = get_GeneOntology(onto="BP", to="SYMBOL")
data_dir = "processed_data/zenith_input_early_late/"
output_dir = "processed_data/zenith_output_early_late/"
suffix = c( "braak_early", "braak_late", "resilience_early", "resilience_late")

cells = c("EN","IN","Astro","Immune","Oligo","OPC","Mural","Endo")

for (i in 1:8){
    fn = paste0(data_dir, cells[i], "_zenith_input.csv")
    print(fn)
    df = read.csv(fn)

    for (j in 1:4){
        res = df[,2+j]
        names(res) = df[,2]
        res.gsa = zenithPR_gsa(statistics=res[res>-99], ids=names(res[res>-99]), geneSets=go.gs.bp,  progressbar=FALSE, use.ranks = FALSE, n_genes_min = 10)
        save_fn = paste0(output_dir, cells[i], "_", suffix[j], ".csv")
        print(save_fn)
        write.csv(res.gsa, save_fn)
    }
}
