library(knitr)
library(scatterplot3d)
library(Rtsne)
library(coRanking)
library(dimRed)

deepwalkcora = read.table("coradeepwalk128.txt")
harpcora = read.table("coraharp128.txt")
graphsagecora = read.table("coragraphsage128.txt")
frmodelcora = read.table("corabatchlayout128.txt")

deepwalkciteseer = read.table("citeseerdeepwalk128.txt")
harpciteseer = read.table("citeseerharp128.txt")
graphsageciteseer = read.table("citeseergraphsage128.txt")
frmodelciteseer = read.table("citeseerbatchlayout128.txt")

emcoradw = embed(deepwalkcora, "tSNE")
emcoraharp = embed(harpcora, "tSNE")
emcorags = embed(graphsagecora, "tSNE")
emcorafr = embed(frmodelcora, "tSNE")

emciteseerdw = embed(deepwalkciteseer, "tSNE")
emciteseerharp = embed(harpciteseer, "tSNE")
emciteseergs = embed(graphsageciteseer, "tSNE")
emciteseerfr = embed(frmodelciteseer, "tSNE")

codwl = Q_local(emcoradw)
coharpl = Q_local(emcoraharp)
cogsl = Q_local(emcorags)
cofrl = Q_local(emcorafr)

cat("coralDW:",codwl,"\n")
cat("coralHARP:",coharpl, "\n")
cat("coralGraphSAGE:",cogsl, "\n")
cat("coralFR:",cofrl, "\n")

codwg = Q_global(emcoradw)
coharpg = Q_global(emcoraharp)
cogsg = Q_global(emcorags)
cofrg = Q_global(emcorafr)

cat("coragDW:",codwg,"\n")
cat("coragHARP:",coharpg,"\n")
cat("coragGraphSAGE:",cogsg, "\n")
cat("coragFR:",cofrg, "\n")



cidwl = Q_local(emciteseerdw)
ciharpl = Q_local(emciteseerharp)
cigsl = Q_local(emciteseergs)
cifrl = Q_local(emciteseerfr)

cat("citelDW:",cidwl,"\n")
cat("citelHARP:",ciharpl, "\n")
cat("citelGraphSAGE:",cigsl, "\n")
cat("citelFR:",cifrl, "\n")

cidwg = Q_global(emciteseerdw)
ciharpg = Q_global(emciteseerharp)
cigsg = Q_global(emciteseergs)
cifrg = Q_global(emciteseerfr)

cat("citegDW:",cidwg,"\n")
cat("citegHARP:",ciharpg,"\n")
cat("citegGraphSAGE:",cigsg, "\n")
cat("citegFR:",cifrg, "\n")
