library(data.table)
library(dplyr)



cts <- fread(path_counts) #path_counts is the path to the raw counts matrix with gene names as rownames, and samples as colnames
# Take out any gene containing NA values
gene_col <- colnames(cts)[1]
genes <- cts[[gene_col]]

# _______________________________
# Gene annotation: Ensembl to HGNC
# _______________________________

# Import dataframe of  Ensembl to HGNC annotation
annot_genes_latestv  <-  "../resources/ensdb_hsapiens_v99.tsv"
annot_genes_prevv  <-  "../resources/ensdb_hsapiens_v86.tsv"

annot_ensdb_df <- data.table::fread(annot_genes_latestv)
ensdb_hsapiens_v86_genes <- data.table::fread(annot_genes_prevv)

# Annotate ensembl genes to HGNC
# Calculate standard deviation (sd) of expression per gene
cts_annot_df <- data.frame("Ensembl" = rownames(cts), "sd" = apply(cts, 1, sd)) %>% 
dplyr::left_join(annot_ensdb_df, by = "Ensembl")

# Keep HGNC with highest standard deviation if duplicated
# Drop genes with sd=0
cts_annot_clean_df <- cts_annot_df %>%
dplyr::filter(sd != 0) %>%
dplyr::group_by(HGNC) %>%
dplyr::top_n(1, sd) %>%
dplyr::slice(1)

# Check if there are NAs in HGNC which can be covered using Ensembl db v86
ensg_missing_hgnc_index <- which(is.na(cts_annot_clean_df$HGNC))
ensg_missing_hgnc <- cts_annot_clean_df$Ensembl[ensg_missing_hgnc_index]

cts_annot_clean_df$HGNC[ensg_missing_hgnc_index] <-
ensdb_hsapiens_v86_genes$HGNC[match(ensg_missing_hgnc, ensdb_hsapiens_v86_genes$Ensembl)]

print(paste(cts_annot_clean_df$Ensembl[ensg_missing_hgnc_index], 
            ": HGNC missing in Ensembl v99 version. Replaced with HGNC values from Ensembl v86:",
    cts_annot_clean_df$HGNC[ensg_missing_hgnc_index]))

cts_annot_clean_df <- na.omit(cts_annot_clean_df)
cts <- cts[rownames(cts)%in%cts_annot_clean_df$Ensembl,]
rownames(cts) <- cts_annot_clean_df$HGNC[match(rownames(cts),cts_annot_clean_df$Ensembl)]

cts