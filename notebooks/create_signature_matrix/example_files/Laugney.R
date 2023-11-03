library(knitr)
library(dplyr)
library(ggforce)
library(ggplot2)
library(RColorBrewer)
library(reshape2) 
library(GeoDiff) 
library(SpatialDecon)
library(ComplexHeatmap)
library(Seurat)
library(caret)
library(anndata)
library(stringr)
library(ggpmisc)
library(ggpubr)
library(nnls)
library(DeconRNASeq)
library(MIND)
library(corrplot)
library(FARDEEP)
library(org.Hs.eg.db)
library(MASS)
library(Rodeo)
library(plyr)

output_dir <- "/mnt/disks/sdb/usr_data/DATA_ALMUDENA/Mosaic/Signatures"                  
dir.create(file.path(output_dir,"Laughney"))
output_dir <- "/mnt/disks/sdb/usr_data/DATA_ALMUDENA/Mosaic/Signatures/Laughney"  

######## Loading functions
source("~/Deconvolution/scr/Signature_functions.R")
source("~/Deconvolution/scr/Deconvolution_function.R")
source("~/Deconvolution/scr/Deconvolution_functions_DWLS.R")
source("/home/aespin-perez/Deconvolution/scr/scdc.R") #downaloaded from https://meichendong.github.io/SCDC/reference/deconv_simple.html    

######## Loading dataset
dir_sc <- "/data/datasets/Zenodo_input_data/data/12_input_adatas/" 
ad <- read_h5ad(file.path(dir_sc,"laughney_massague_2020_nsclc.h5ad"))
expr <- t(as.matrix(ad$X))
obs <- ad$obs

# transferring cell type annotations from integrated dataset into Laughney_Massague_2020 dataset
dir_sc <- "/data/datasets/CELLXGENE"
integr <- readRDS(file.path(dir_sc,"local.rds"))
integr2 <- subset(x = integr, subset = study == "Laughney_Massague_2020")
rm(integr)
rownames(integr2@meta.data) <- gsub("-.*","",rownames(integr2@meta.data))

integr2@meta.data <- integr2@meta.data[order(match(rownames(integr2@meta.data),colnames(expr))),]
expr2 <- expr[,colnames(expr) %in% rownames(integr2@meta.data)]
obs2 <- obs[colnames(expr) %in% rownames(integr2@meta.data),]
rm(expr)
print(identical(rownames(integr2@meta.data),colnames(expr2)))
obs2$cell_type <- integr2@meta.data$cell_type
roremove <- grep('neutrophil|type I pneumocyte|type II pneumocyte|multi-ciliated epithelial cell|club cell|capillary endothelial cell',obs2$cell_type)
obs2 <- obs2[-roremove,]
expr2 <- expr2[,-roremove]

# Removing mitochondrial and ribosomal genes
#expr2 <- NormalizeData(object = expr2, normalization.method = "RC",scale.factor = 1e6)
genes.ribomit <- grep(pattern = "^RP[SL][[:digit:]]|^RP[[:digit:]]|^RPSA|^RPS|^RPL|^MT-|^MRPL",rownames(expr2))
dim(expr2) 
expr2 <- expr2[-c(genes.ribomit),]
dim(expr2) 

# Grouping cell types
obs2$MainImmune <- ifelse(obs2$cell_type %in% c("non-classical monocyte","macrophage","classical monocyte","alveolar macrophage","myeloid cell"),
    "MonoMacro",
    ifelse(obs2$cell_type %in% c("CD8-positive, alpha-beta T cell","CD4-positive, alpha-beta T cell","natural killer cell","regulatory T cell"),
    "TNK",
    #ifelse(obs_train$cell_type %in% c("type I pneumocyte","type II pneumocyte","multi-ciliated epithelial cell","epithelial cell of lung","club cell","capillary endothelial cell"),
    ifelse(obs2$cell_type %in% c("epithelial cell of lung"),
    "Epithelial",
    ifelse(obs2$cell_type %in% c("malignant cell"),
    "Malignant",
    ifelse(obs2$cell_type %in% c("vein endothelial cell","pulmonary artery endothelial cell","endothelial cell of lymphatic vessel"),
    "Endothelial",
    ifelse(obs2$cell_type %in% c("stromal cell","smooth muscle cell","pericyte","perycite","mesothelial cell","fibroblast of lung","bronchus fibroblast of lung"),
    "Stroma",
    ifelse(obs2$cell_type %in% c("B cell","plasma cell"),
    "B",
    ifelse(obs2$cell_type %in% c("plasmacytoid dendritic cell","conventional dendritic cell","CD1c-positive myeloid dendritic cell","dendritic cell"),
    "DC",
    ifelse(obs2$cell_type %in% c("mast cell"),
    "Mast",
    as.character(obs2$cell_type))
    ))))))))
table(obs2$MainImmune)

# Split dataset into 2
trainIndex <- createDataPartition(obs2$cell_type, p = .5)
scRNseq_t <- expr2[,unlist(trainIndex)]
obs_train <- obs2[unlist(trainIndex),]

scRNseq_test <- expr2[,-unlist(trainIndex)]
obs_test <- obs2[-unlist(trainIndex),]


########## Building signature on the train dataset

# Differential expression analysis
scRNseq_seurat <- CreateSeuratObject(scRNseq_t, project = "DGE")
Idents(scRNseq_seurat) <- obs_train$MainImmune
if(!file.exists(file.path(output_dir,paste0("DE_",unique(obs_train$MainImmune)[length(unique(obs_train$MainImmune))],".txt")))){
    DGE_celltypes(scRNseq_seurat,Idents(scRNseq_seurat),file.path(output_dir))
}

if(!file.exists(file.path(output_dir,"Laughney.txt"))){

  scRNseq_t <- NormalizeData(object = scRNseq_t, normalization.method = "RC",scale.factor = 10000)

  signature <- buildSignatureMatrix("Laughney",
      scRNseq_t,obs_train$MainImmune,file.path(output_dir),
      pvaladj.cutoff=0.05,diff.cutoff=0.25,
      minG=50,maxG=200)
  write.table(signature,file.path(output_dir,"Laughney.txt"),sep="\t",row.names=TRUE,col.names=NA)
  
}else{
  signature <- read.table(file.path(output_dir,"Laughney.txt"),sep="\t",row.names=1,header=TRUE)
}

########### Testing signature

#signature <- read.table(file.path(output_dir,"Laughney.txt"),sep="\t",row.names=1,header=TRUE)

scRNseq_test <- NormalizeData(object = scRNseq_test, normalization.method = "RC",scale.factor = 10000)


#### Sanity check 1
dir.create(file.path(output_dir,"Sanity_Check_1"), recursive = TRUE, showWarnings = FALSE)

IDsCells <- unique(obs_test$MainImmune)
T_test <- c()
for(celltype in IDsCells){
  T_test <- cbind(T_test,apply(scRNseq_test,1,function(y) mean(y[which(obs_test$MainImmune==celltype)]))) #to try both median and mean
 }
colnames(T_test) <- IDsCells
out <- Deconvolution(T_test,as.data.frame(signature))


out2 <- do.call(rbind.data.frame, out)
out2 <- melt(as.matrix(out2))
out2$Mix <- gsub(".*\\.","",out2[,1])
out2[,1] <- gsub("\\..*","",out2[,1])
colnames(out2) <- c("Method","CellType","Fraction","Mix")
out2 <- out2[which(out2$Mix == out2$CellType),]

p <- ggplot(out2, aes(x=CellType, y=Fraction, color=Method)) + 
  geom_point() + theme_bw() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
ggsave(file.path(output_dir,"Sanity_Check_1","SC1.png"))

p <- ggplot(out2, aes(x=CellType, y=Fraction, color=Method)) + 
  geom_point() + theme_bw() + scale_color_brewer(palette="Paired") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
ggsave(file.path(output_dir,"Sanity_Check_1","SC1_2.png"))


summary <- out2 %>%
    group_by(Method) %>%
    summarise_at(vars(Fraction), list(name = mean))
summary <- as.data.frame(summary)
summary <- summary[order(-summary$name),] 
write.table(summary,file.path(output_dir,"Sanity_Check_1","summary_frac.txt"),sep="\t",row.names=FALSE,col.names=TRUE)


S = Rodeo(T_test, out[[6]])
write.table(S, file.path(output_dir,"Sanity_Check_1",file="EstimatedS_6.txt"), sep="\t", quote=F,row.names=TRUE,col.names=NA)



#### Sanity check 2
dir.create(file.path(output_dir,"Sanity_Check_2"), recursive = TRUE, showWarnings = FALSE)
colnames(scRNseq_test) <- obs_test$MainImmune

Nsampl <- 100
frac <- c()
T <- c()
for(i in 1:Nsampl){
  set.seed(i)
  CellSub <- sample(colnames(scRNseq_test),1000)
  scRNseq_testSub <- scRNseq_test[,CellSub]
  scRNseq_testSub2 <- as.matrix(scRNseq_testSub)
  scRNseq_testSub2 <- t(scRNseq_testSub2)
  frac <- cbind(frac,table(rownames(scRNseq_testSub2))/1000)
  scRNseq_testSub2 <- apply(scRNseq_testSub2,2,mean)
  T <- cbind(T,scRNseq_testSub2)
}
fractions <- Deconvolution(T,as.data.frame(signature))
S = Rodeo(T, fractions[[6]])
write.table(S, file.path(output_dir,"Sanity_Check_2",file="EstimatedS_6.txt"), sep="\t", quote=F,row.names=TRUE,col.names=NA)

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
write.table(correl_all,file.path(output_dir,"Sanity_Check_2","summary_correl.txt"),sep="\t",row.names=FALSE,col.names=TRUE)

p_meds <- ddply(correl_all, .(Method), summarise, med = median(Correlation))
p <- ggplot(correl_all, aes(x=Method, y=Correlation)) + 
  geom_boxplot() + theme_bw() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+ 
    geom_text(data = p_meds, aes(x = Method, y = med, label = round(med,4)), size = 3, vjust = -6.5)
ggsave(file.path(output_dir,"Sanity_Check_2","SC2.png"))


#### Sanity check 3
dir.create(file.path(output_dir,"Sanity_Check_3"), recursive = TRUE, showWarnings = FALSE)

NcellType <- unique(obs_test$MainImmune)
for(CT in NcellType){
  assign(CT,scRNseq_test[,which(obs_test$MainImmune == CT)])
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
    temp <- as.matrix(temp)
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
S = Rodeo(T, fractions[[10]])
write.table(S, file.path(output_dir,"Sanity_Check_3",file="EstimatedS_10.txt"), sep="\t", quote=F,row.names=TRUE,col.names=NA)

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
write.table(correl_all,file.path(output_dir,"Sanity_Check_3","summary_correl.txt"),sep="\t",row.names=FALSE,col.names=TRUE)

p <- ggplot(correl_all, aes(x=Method, y=Correlation)) + 
  geom_boxplot() + theme_bw() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
ggsave(file.path(output_dir,"Sanity_Check_3","SC3.png"))
