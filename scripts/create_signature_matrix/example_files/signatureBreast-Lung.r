### from Almudena's script : project/processing_CHUV/omics-toolbox/GeoMX_WTA/workflow/utils_and_notebooks/Almudena/Deconvolution/Signature_lung.r
# the only difference is the granularity of cell types

#library(anndata)
library(Seurat)

library(dplyr)
library(plyr)
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
source("~/project/processing_CHUV/omics-toolbox/GeoMX_WTA/workflow/utils_and_notebooks/Almudena/Deconvolution/Deconvolution_function.R")
source("~/project/processing_CHUV/omics-toolbox/GeoMX_WTA/workflow/utils_and_notebooks/Almudena/Deconvolution/Signature_function.R")


dir_out <- "~/project/sandbox-SCa/deconvolution-signature/output/Breast-Lung"
dir.create(dir_out)

dirsc <- "~/project/processing_CHUV/omics-toolbox/GeoMX_WTA/workflow/utils_and_notebooks/Almudena/Deconvolution/Output/Signature/scrnaseq"
expr_obj <- Read10X(dirsc)
metadata <- read.table(file.path(dirsc,"metadata.csv"),sep=",",row.names=1,header=TRUE)
expr <- CreateSeuratObject(counts=expr_obj,meta.data=metadata)
#expr <- subset(x = expr, subset = tissue == "breast")

# remove  TNK_dividing (not enough specific)
# merge together DC and DC ativated but keep pDC apart
expr <- subset(x = expr, subset = Celltype != "TNK_dividing")

expr$Celltype2 <- ifelse(grepl("Tu",expr$Celltype),"Tumor",expr$Celltype)
# expr$Celltype2 <- ifelse(grepl("T_",expr$Celltype2),"TNK",expr$Celltype2)
# expr$Celltype2 <- ifelse(grepl("\\bNK\\b",expr$Celltype2),"TNK",expr$Celltype2)
# expr$Celltype2 <- ifelse(grepl("\\bTNK_dividing\\b",expr$Celltype2),"TNK",expr$Celltype2)
expr$Celltype2 <- ifelse(grepl("\\bDC_activated\\b",expr$Celltype2),"DC",expr$Celltype2)
# expr$Celltype2 <- ifelse(grepl("\\bDC_plasmacytoid\\b",expr$Celltype2),"DC",expr$Celltype2)
expr$Celltype2 <- ifelse(grepl("\\bPericyte\\b",expr$Celltype2),"Stroma",expr$Celltype2)
expr$Celltype2 <- ifelse(grepl("\\bFibro_muscle\\b",expr$Celltype2),"Stroma",expr$Celltype2)


# Removing mitochondrial and ribosomal genes
#expr2 <- NormalizeData(object = expr2, normalization.method = "RC",scale.factor = 1e6)
genes.ribomit <- grep(pattern = "^RP[SL][[:digit:]]|^RP[[:digit:]]|^RPSA|^RPS|^RPL|^MT-|^MRPL",rownames(expr))
dim(expr) 
expr <- expr[-c(genes.ribomit),]
dim(expr) 
# remove housekeeping genes and patient specific ones: ACTB, if only this one, not a big deal
genes2remove = grep(pattern = "^ACTB$|TMSB4X|IGKC|^IG[HL]",rownames(expr))
expr <- expr[-c(genes2remove),]
dim(expr)

# Split dataset into 2
trainIndex <- createDataPartition(as.character(expr$Celltype2), p = .5)
scRNseq_t <- expr[,unlist(trainIndex)]

scRNseq_test <- expr[,-unlist(trainIndex)]


# Differential expression analysis
#scRNseq_seurat <- CreateSeuratObject(scRNseq_t, project = "Signature")
Idents(scRNseq_t) <- scRNseq_t$Celltype2
print(table(Idents(scRNseq_t)))
if(!file.exists(file.path(dir_out,paste0("DE_",unique(Idents(scRNseq_t))[length(unique(Idents(scRNseq_t)))],".txt")))){
    DGE_celltypes(scRNseq_t,Idents(scRNseq_t),file.path(dir_out))
}


if(!file.exists(file.path(dir_out,"Chr_CHUV.txt"))){

  scRNseq_t <- NormalizeData(object = scRNseq_t, normalization.method = "RC",scale.factor = 10000)

  signature <- buildSignatureMatrix_Seurat("Chr_CHUV",
      scRNseq_t,Idents(scRNseq_t),file.path(dir_out),
      pvaladj.cutoff=0.05,diff.cutoff=0.5,
      minG=50,maxG=200)
  write.table(signature,file.path(dir_out,"Chr_CHUV.txt"),sep="\t",row.names=TRUE,col.names=NA)
  
}else{
  signature <- read.table(file.path(dir_out,"Chr_CHUV.txt"),sep="\t",row.names=1,header=TRUE)
}

################ Testing signature

#scRNseq_test <- NormalizeData(object = scRNseq_test, normalization.method = "RC",scale.factor = 10000)
scRNseq_test <- NormalizeData(object = scRNseq_test, normalization.method = "LogNormalize",scale.factor = 10000)

Idents(scRNseq_test) <- scRNseq_test$Celltype2
print(table(Idents(scRNseq_test)))

#### Sanity check 1
dir.create(file.path(dir_out,"Sanity_Check_1"), recursive = TRUE, showWarnings = FALSE)
T_test <- AverageExpression(object = scRNseq_test,slot="data")
T_test <- as.data.frame(T_test)
colnames(T_test) <- gsub("RNA.","",colnames(T_test))
#T_test <- 2^T_test

# IDsCells <- unique(Idents(scRNseq_t))
# T_test <- c()
# for(celltype in IDsCells){
#   T_test <- cbind(T_test,apply(scRNseq_test,1,function(y) mean(y[which(Idents(scRNseq_t)==celltype)]))) #to try both median and mean
#  }
# colnames(T_test) <- IDsCells
out <- Deconvolution(T_test,as.data.frame(signature))


out2 <- do.call(rbind.data.frame, out)
out2 <- melt(as.matrix(out2))
out2$Mix <- gsub(".*\\.","",out2[,1])
out2[,1] <- gsub("\\..*","",out2[,1])
colnames(out2) <- c("Method","CellType","Fraction","Mix")
out2 <- out2[which(out2$Mix == out2$CellType),]

p <- ggplot(out2, aes(x=CellType, y=Fraction, color=Method)) + 
  geom_point() + theme_bw() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
ggsave(file.path(dir_out,"Sanity_Check_1","SC1.png"))

p <- ggplot(out2, aes(x=CellType, y=Fraction, color=Method)) + 
  geom_point() + theme_bw() + scale_color_brewer(palette="Paired") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
ggsave(file.path(dir_out,"Sanity_Check_1","SC1_2.png"))


summary <- out2 %>%
    group_by(Method) %>%
    summarise_at(vars(Fraction), list(name = mean))
summary <- as.data.frame(summary)
summary <- summary[order(-summary$name),] 
write.table(summary,file.path(dir_out,"Sanity_Check_1","summary_frac.txt"),sep="\t",row.names=FALSE,col.names=TRUE)


# S = Rodeo(T_test, out[[6]])
# write.table(S, file.path(dir_out,"Sanity_Check_1",file="EstimatedS_6.txt"), sep="\t", quote=F,row.names=TRUE,col.names=NA)



#### Sanity check 2
dir.create(file.path(dir_out,"Sanity_Check_2"), recursive = TRUE, showWarnings = FALSE)
#colnames(scRNseq_test) <- as.character(Idents(scRNseq_test))
scRNseq_test <- RenameCells(scRNseq_test, Idents(scRNseq_test))

Nsampl <- 100
frac <- c()
T <- c()
for(i in 1:Nsampl){
  set.seed(i)
  CellSub <- sample(colnames(scRNseq_test),1000)
  scRNseq_testSub <- scRNseq_test[,CellSub]
  #scRNseq_testSub2 <- as.matrix(scRNseq_testSub)
  scRNseq_testSub2 <- GetAssayData(object = scRNseq_testSub, assay.type = "RNA", slot = "data")
  scRNseq_testSub2 <- t(scRNseq_testSub2)
  frac <- cbind(frac,table(gsub("_C$","",gsub("_G$","",gsub("_T$","",gsub("_A$","",substr(rownames(scRNseq_testSub2), 1, nchar(rownames(scRNseq_testSub2))-27))))))/1000)
  #frac <- cbind(frac,table(substr(rownames(scRNseq_testSub2), 1, nchar(rownames(scRNseq_testSub2))-27)))
  scRNseq_testSub2 <- apply(scRNseq_testSub2,2,mean)
  T <- cbind(T,scRNseq_testSub2)
}
fractions <- Deconvolution(T,as.data.frame(signature))
# S = Rodeo(T, fractions[[6]])
# write.table(S, file.path(dir_out,"Sanity_Check_2",file="EstimatedS_6.txt"), sep="\t", quote=F,row.names=TRUE,col.names=NA)

frac <- frac[order(match(rownames(frac),rownames(fractions[[1]]))),]
identical(rownames(frac),rownames(fractions[[1]]))
correl_all <- c()
for(i in 1:length(fractions)){
  for(samp in 1:Nsampl){
    correl <- cor.test(frac[,samp],fractions[[i]][,samp])$estimate
    correl_pval <- cor.test(frac[,samp],fractions[[i]][,samp])$p.value
    correl_all <- rbind(correl_all,c(correl,correl_pval))
  }
}
correl_all <- data.frame(Sample=rep(seq(1:Nsampl),10),Method=rep(names(fractions),Nsampl),Correlation=correl_all[,1],Pvalue=correl_all[,2])
write.table(correl_all,file.path(dir_out,"Sanity_Check_2","summary_correl.txt"),sep="\t",row.names=FALSE,col.names=TRUE)

p_meds <- ddply(correl_all, .(Method), summarise, med = median(Correlation))
p <- ggplot(correl_all, aes(x=Method, y=Correlation)) + 
  geom_boxplot() + theme_bw() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+ 
    geom_text(data = p_meds, aes(x = Method, y = med, label = round(med,4)), size = 3, vjust = -6.5)
ggsave(file.path(dir_out,"Sanity_Check_2","SC2.png"))


#### Sanity check 3
dir.create(file.path(dir_out,"Sanity_Check_3"), recursive = TRUE, showWarnings = FALSE)

NcellType <- unique(Idents(scRNseq_test))
for(CT in NcellType){
  assign(CT,scRNseq_test[,which(Idents(scRNseq_test) == CT)])
}

Nsampl <- 100
frac <- c()
T <- c()
for(i in 1:100){
  set.seed(i)
  Nprop <- runif(length(NcellType),0,1)
    
  Ncell <- c() 
  expr_N <- c()
  for(CT in NcellType){
    temp <- eval(parse(text = as.character(CT)))
    if(round(Nprop[which(NcellType == CT)]*100) == 0){next} 
    temp <- temp[,sample(colnames(temp),round(Nprop[which(NcellType == CT)]*50))]
    temp <-  GetAssayData(object = temp, assay.type = "RNA", slot = "data")
    temp <- t(temp)
    expr_N <- rbind(expr_N,temp)
    Ncell <- c(Ncell,nrow(temp)) 
  }
  total <- sum(Ncell)
  Ncell <- Ncell/total
  frac <- cbind(frac,Ncell)

  Tsub <- apply(expr_N,2,mean)
  T <- cbind(T,Tsub)
}

rownames(frac) <- NcellType
fractions <- Deconvolution(T,as.data.frame(signature))
# S = Rodeo(T, fractions[[10]])
# write.table(S, file.path(dir_out,"Sanity_Check_3",file="EstimatedS_10.txt"), sep="\t", quote=F,row.names=TRUE,col.names=NA)

frac <- frac[order(match(rownames(frac),rownames(fractions[[1]]))),]
identical(rownames(frac),rownames(fractions[[1]]))

correl_all <- c()
for(i in 1:length(fractions)){
  for(samp in 1:Nsampl){
    correl <- cor.test(frac[,samp],fractions[[i]][,samp])$estimate
    correl_pval <- cor.test(frac[,samp],fractions[[i]][,samp])$p.value
    correl_all <- rbind(correl_all,c(correl,correl_pval))
  }
}
correl_all <- data.frame(Method=rep(names(fractions),Nsampl),Correlation=correl_all[,1],Pvalue=correl_all[,2])
write.table(correl_all,file.path(dir_out,"Sanity_Check_3","summary_correl.txt"),sep="\t",row.names=FALSE,col.names=TRUE)

p <- ggplot(correl_all, aes(x=Method, y=Correlation)) + 
  geom_boxplot() + theme_bw() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
ggsave(file.path(dir_out,"Sanity_Check_3","SC3.png"))