library(rrvgo)

fn = "processed_data/rrvgo/rrvgo_input.csv"
save_fn = "processed_data/rrvgo/rrvgo_output.csv"

df = read.csv(fn)
th = 0.80
method <- "Wang"

simMatrix <- calculateSimMatrix(df$go_id, orgdb="org.Hs.eg.db", ont="BP", method=method)
scores <- setNames(-log(pmax(1e-32, df$FDR)), df$go_id)
reducedTerms <- reduceSimMatrix(simMatrix, scores, threshold=th,orgdb="org.Hs.eg.db")
treemapPlot(reducedTerms, title="All cells")

write.csv(reducedTerms, save_fn)

df <- reducedTerms[reducedTerms$term == reducedTerms$parentTerm,]
print(dim(df))
