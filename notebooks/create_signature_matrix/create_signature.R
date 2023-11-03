# Imports
library(anndata)
library(Seurat)
library(reticulate)
use_python("/home/owkin/.conda/envs/deepdeconv/bin/python")

library(plyr)
library(dplyr)
library(ggforce)
library(ggplot2)
library(reshape2) 
library(ComplexHeatmap)
library(corrplot)
library(caret)

library(DeconRNASeq)
library(nnls)
library(FARDEEP)
library(MIND)
source("~/deepdeconv/notebooks/create_signature_matrix/helpers/Signature_function.R")
source("~/deepdeconv/notebooks/create_signature_matrix/helpers/Deconvolution_function.R")

dir_out <- "~/project/Simon"


# Load data

dirsc <- "~/data/cross-tissue/omics/raw"
filepath <- file.path(dirsc,"local.h5ad")
ad <- read_h5ad(filepath)

raw_X <- t(ad$raw$X)
rownames(raw_X) <- ad$var_names
colnames(raw_X) <- ad$obs_names

train_test_cell_types = read.csv("~/project/train_test_index_matrix_granular_updated.csv", row.names = 2)
ad$obs$precise_groups_updated <- train_test_cell_types$precise_groups_updated
ad$obs$train_index <- train_test_cell_types$Train.index


# Convert ENSG to HGNC

annot_genes_latestv  <-  "~/deepdeconv/notebooks/create_signature_matrix/helpers/ensdb_hsapiens_v99.tsv" # This one covers everything in the CTI dataset
annot_ensdb_df <- data.table::fread(annot_genes_latestv)
cts_annot_df <- data.frame("Ensembl" = rownames(raw_X))  %>% 
dplyr::left_join(annot_ensdb_df, by = "Ensembl")
# Find duplicates
duplicates <- cts_annot_df[which(duplicated(cts_annot_df$HGNC) | duplicated(cts_annot_df$HGNC, fromLast=TRUE)),]
trainIndex <- which(ad$obs$train_index == "True")
scRNseq_train_duplicates <- raw_X[duplicates$Ensembl,unlist(trainIndex)]
duplicates$sd <- apply(scRNseq_train_duplicates, 1, sd)
# Find duplicates with lowest sd
duplicates_to_remove <- duplicates %>%
dplyr::group_by(HGNC) %>%
dplyr::top_n(-1, sd) %>%
dplyr::slice(1)
dim(cts_annot_df)
cts_annot_clean_df <- cts_annot_df %>%
  filter(!(Ensembl %in% duplicates_to_remove$Ensembl))
dim(cts_annot_clean_df)
# Create seurat object
raw_X_clean <- raw_X[!rownames(raw_X) %in% duplicates_to_remove$Ensembl,]
rownames(raw_X_clean) <- cts_annot_clean_df$HGNC

# Create Seurat Object
expr = CreateSeuratObject(counts=raw_X_clean, meta.data=as.data.frame(ad$obs))


dim(expr)
# Remove some cell types
expr_clean <- subset(x = expr, subset = precise_groups_updated != "To remove")
dim(expr_clean)
# Removing mitochondrial and ribosomal genes
genes.ribomit <- grep(pattern = "^RP[SL][[:digit:]]|^RP[[:digit:]]|^RPSA|^RPS|^RPL|^MT-|^MRPL",rownames(expr_clean))
expr_clean <- expr_clean[-c(genes.ribomit),]
dim(expr_clean) 
# Remove housekeeping genes and patient specific ones: ACTB, if only this one, not a big deal
genes2remove = grep(pattern = "^ACTB$|TMSB4X|IGKC|^IG[HL]",rownames(expr_clean))
expr_clean <- expr_clean[-c(genes2remove),]
dim(expr_clean)
# Convert back to ENSG to be in accordance with the CTI data
# The following doesn't work because renaming features in v3/v4 assays is not supported
# rownames(expr_clean) <- cts_annot_clean_df$Ensembl[match(rownames(expr_clean), cts_annot_clean_df$HGNC)]
# Therefore, one should recreate the Seurat object from scratch to rename the rownames


# Split dataset into 2

trainIndex <- which(expr$train_index == "True")
scRNseq_t <- expr[,unlist(trainIndex)]
# scRNseq_test <- expr[,-unlist(trainIndex)]


# Differential expression analysis

## WARNING. The signature matrix function will not work if there is space inside the cell type names.
## Therefore, if needed, one should remove the spaces for the creation of the idents, like in the three following lines.
# idents <- ifelse(scRNseq_t$precise_groups_updated == "CD4 T", "CD4T", scRNseq_t$precise_groups_updated)
# idents <- ifelse(idents == "CD8 T", "CD8T", idents)
# Idents(scRNseq_t) <- idents
Idents(scRNseq_t) <- scRNseq_t$precise_groups_updated
print(table(Idents(scRNseq_t)))
if(!file.exists(file.path(dir_out,paste0("DE_",unique(Idents(scRNseq_t))[length(unique(Idents(scRNseq_t)))],".txt")))){
    DGE_celltypes(scRNseq_t,Idents(scRNseq_t),file.path(dir_out))
}


# Signature matrix

if(!file.exists(file.path(dir_out,"CTI_granular_updated1.txt"))){

  scRNseq_t <- NormalizeData(object = scRNseq_t, normalization.method = "RC",scale.factor = 10000)

  signature <- buildSignatureMatrix_Seurat("CTI_granular_updated1",
      scRNseq_t,Idents(scRNseq_t),file.path(dir_out),
      pvaladj.cutoff=0.05,diff.cutoff=0.5,
      minG=50,maxG=200)
  write.table(signature,file.path(dir_out,"CTI_granular_updated.txt"),sep="\t",row.names=TRUE,col.names=NA)
  
}else{
  signature <- read.table(file.path(dir_out,"CTI_granular_updated.txt"),sep="\t",row.names=1,header=TRUE)
}


# Convert signature gene names from HGNC to ENSG

signature_ensg <- signature
rownames(signature_ensg) <- cts_annot_clean_df$Ensembl[match(rownames(signature_ensg), cts_annot_clean_df$HGNC)]
if(!file.exists(file.path(dir_out,"CTI_granular_updated_ensg.txt"))){
  write.table(signature_ensg,file.path(dir_out,"CTI_granular_updated_ensg.txt"),sep="\t",row.names=TRUE,col.names=NA)
}