
### Function to conduct DGE
DGE_celltypes <- function(expr,clusters,output_dir){
    for(celltype in unique(clusters)){
        DEcelltype <- FindMarkers(object=expr, ident.1 = celltype, ident.2 = NULL, only.pos = TRUE, test.use = "bimod")
        write.table(DEcelltype,file.path(output_dir,paste0("DE_",celltype,".txt")),sep="\t",row.names=TRUE,col.names=NA)
    }
}

# ## Function to build signature. similar approach in "buildSignatureMatrixUsingSeurat" from https://github.com/dtsoucas/DWLS/blob/master/Deconvolution_functions.R

buildSignatureMatrix <- function(NameSig,expression,clusters,output_dir,diff.cutoff=NULL,pvaladj.cutoff=NULL,minG=NULL,maxG=NULL){
  if(is.null(diff.cutoff)){diff.cutoff <- 0.25}
  if(is.null(pvaladj.cutoff)){pvaladj.cutoff <- 0.05}
  if(is.null(minG)){minG <- 50}
  if(is.null(maxG)){maxG <- 200}

  cat(paste("Building signature using the following cutoffs:",diff.cutoff,"for FC,",pvaladj.cutoff,"for adjusted pvalue,",minG,"minimum number of genes and",maxG,"maximum number of genes\n"))

  NGenes<-c()
  IDsCells <- unique(clusters)
  for (celltype in IDsCells){
    DEcelltype <- read.table(file.path(output_dir,paste0("DE_",celltype,".txt")),sep="\t",row.names=1,header=TRUE)
    DEGenes<-rownames(DEcelltype)[intersect(which(DEcelltype$p_val_adj<pvaladj.cutoff),which(DEcelltype$avg_log2FC>diff.cutoff))]
    nonMir = grep("MIR|Mir", DEGenes, invert = T)
    assign(paste0("DE_filt.",celltype),DEcelltype[which(rownames(DEcelltype) %in% DEGenes[nonMir]),])
    NGenes<-c(NGenes,length(DEGenes[nonMir]))
  }
  names(NGenes) <- IDsCells
  write.table(NGenes,file.path(output_dir,paste0("Overview_Nmarkers_",diff.cutoff,"_",pvaladj.cutoff,"_",minG,"_",maxG,".txt")))
  #if(max(NGenes) < maxG){maxG <- max(NGenes)} 

  conditionNumbers<-c()
  ntop <- c()
  for(G in minG:maxG){

    IDGenes<-c()
    for(Ind in 1:length(IDsCells)){
      temp <- eval(parse(text = as.name(paste0("DE_filt.",IDsCells[Ind]))))
      temp <- temp[order(temp$avg_log2FC,decreasing=TRUE),]       # to try both sorting based on p_pval_adj or avg_log2FC
      IDGenes <- c(IDGenes,(rownames(temp)[1:min(G,NGenes[Ind])]))
    }

    IDGenes <- unique(IDGenes)
    ntop <- c(ntop,length(IDGenes))

    # building signature matrices for each G using top most significant markers
    ExprSubset <- expression[IDGenes,]
    if(max(ExprSubset) < 10){ExprSubset <- 2^ExprSubset}
    Sig <- c()
    for(celltype in IDsCells){
      Sig <- cbind(Sig,(apply(ExprSubset,1,function(y) mean(y[which(clusters==celltype)])))) #to try both median and mean
    }
    colnames(Sig) <- IDsCells
    conditionNumbers <- c(conditionNumbers,kappa(Sig))
  }
  CN <- data.frame(G=minG:maxG,Ngenes=ntop,ConditionNumber=conditionNumbers)
  a <- ggplot(CN, aes(x=G, y=ConditionNumber)) + theme_bw() + geom_point() + geom_vline(xintercept = which.min(conditionNumbers) + minG-1, linetype="dotted", color = "blue", size=1.5)
  ggsave(file.path(output_dir,paste0("Overview_ConditionNumbers_",diff.cutoff,"_",pvaladj.cutoff,"_",minG,"_",maxG,".png")))
  write.table(CN,file.path(output_dir,paste0("Overview_ConditionNumbers_",diff.cutoff,"_",pvaladj.cutoff,"_",minG,"_",maxG,".txt")))

  Gall <- which.min(conditionNumbers) + minG-1 # The optimal number of DE genes selected
  cat(paste("The optimal number of genes for the signature is",Gall,"\n"))

  # Build final signature matrix with the most optimal number of genes
  IDGenes<-c()
  for(Ind in 1:length(IDsCells)){
    temp <- eval(parse(text = as.name(paste0("DE_filt.",IDsCells[Ind]))))
    temp <- temp[order(temp$avg_log2FC,decreasing=TRUE),]
    IDGenes <- c(IDGenes,(rownames(temp)[1:min(Gall,NGenes[Ind])]))
   }

  IDGenes <- unique(IDGenes)
  ExprSubset <- expression[IDGenes,]
  Sig <- c()
   for(celltype in IDsCells){
     Sig <- cbind(Sig,(apply(ExprSubset,1,function(y) mean(y[which(clusters==celltype)])))) #to try both median and mean
   }
  colnames(Sig) <- IDsCells
  write.table(Sig,file.path(output_dir,paste0("Signature_",NameSig,"_",diff.cutoff,"_",pvaladj.cutoff,"_",minG,"_",maxG,".txt")),sep="\t",row.names=TRUE,col.names=NA)
  png(file.path(output_dir,paste0("Signature_",NameSig,"_",diff.cutoff,"_",pvaladj.cutoff,"_",minG,"_",maxG,".png")))
    a <- heatmap(Sig)
  dev.off()
  return(Sig)
}


buildSignatureMatrix_Seurat <- function(NameSig,expression,clusters,output_dir,diff.cutoff=NULL,pvaladj.cutoff=NULL,minG=NULL,maxG=NULL){
  if(is.null(diff.cutoff)){diff.cutoff <- 0.25}
  if(is.null(pvaladj.cutoff)){pvaladj.cutoff <- 0.05}
  if(is.null(minG)){minG <- 50}
  if(is.null(maxG)){maxG <- 200}

  cat(paste("Building signature using the following cutoffs:",diff.cutoff,"for FC,",pvaladj.cutoff,"for adjusted pvalue,",minG,"minimum number of genes and",maxG,"maximum number of genes\n"))

  NGenes<-c()
  IDsCells <- unique(clusters)
  for (celltype in IDsCells){
    DEcelltype <- read.table(file.path(output_dir,paste0("DE_",celltype,".txt")),sep="\t",row.names=1,header=TRUE)
    DEGenes<-rownames(DEcelltype)[intersect(which(DEcelltype$p_val_adj<pvaladj.cutoff),which(DEcelltype$avg_log2FC>diff.cutoff))]
    nonMir = grep("MIR|Mir", DEGenes, invert = T)
    assign(paste0("DE_filt.",celltype),DEcelltype[which(rownames(DEcelltype) %in% DEGenes[nonMir]),])
    NGenes<-c(NGenes,length(DEGenes[nonMir]))
  }
  names(NGenes) <- IDsCells
  write.table(NGenes,file.path(output_dir,paste0("Overview_Nmarkers_",diff.cutoff,"_",pvaladj.cutoff,"_",minG,"_",maxG,".txt")))
  #if(max(NGenes) < maxG){maxG <- max(NGenes)} 

  conditionNumbers<-c()
  ntop <- c()
  for(G in minG:maxG){

    IDGenes<-c()
    for(Ind in 1:length(IDsCells)){
      temp <- eval(parse(text = as.name(paste0("DE_filt.",IDsCells[Ind]))))
      temp <- temp[order(temp$avg_log2FC,decreasing=TRUE),]       # to try both sorting based on p_pval_adj or avg_log2FC
      IDGenes <- c(IDGenes,(rownames(temp)[1:min(G,NGenes[Ind])]))
    }

    IDGenes <- unique(IDGenes)
    ntop <- c(ntop,length(IDGenes))

    # building signature matrices for each G using top most significant markers
    ExprSubset <- GetAssayData(expression[IDGenes,], assay.type = "RNA", slot = "data")
    if(max(ExprSubset) < 10){ExprSubset <- 2^ExprSubset} # RNA slot is not available. normalized data is log2 or log10?
    Sig <- c()
    for(celltype in IDsCells){
      Sig <- cbind(Sig,(apply(ExprSubset,1,function(y) mean(y[which(clusters==celltype)])))) #to try both median and mean
    }
    colnames(Sig) <- IDsCells
    conditionNumbers <- c(conditionNumbers,kappa(Sig))
  }
  CN <- data.frame(G=minG:maxG,Ngenes=ntop,ConditionNumber=conditionNumbers)
  a <- ggplot(CN, aes(x=G, y=ConditionNumber)) + theme_bw() + geom_point() + geom_vline(xintercept = which.min(conditionNumbers) + minG-1, linetype="dotted", color = "blue", size=1.5)
  ggsave(file.path(output_dir,paste0("Overview_ConditionNumbers_",diff.cutoff,"_",pvaladj.cutoff,"_",minG,"_",maxG,".png")))
  write.table(CN,file.path(output_dir,paste0("Overview_ConditionNumbers_",diff.cutoff,"_",pvaladj.cutoff,"_",minG,"_",maxG,".txt")))

  Gall <- which.min(conditionNumbers) + minG-1 # The optimal number of DE genes selected
  cat(paste("The optimal number of genes for the signature is",Gall,"\n"))

  # Build final signature matrix with the optimal number of genes
  IDGenes<-c()
  for(Ind in 1:length(IDsCells)){
    temp <- eval(parse(text = as.name(paste0("DE_filt.",IDsCells[Ind]))))
    temp <- temp[order(temp$avg_log2FC,decreasing=TRUE),]
    IDGenes <- c(IDGenes,(rownames(temp)[1:min(Gall,NGenes[Ind])]))
   }

  IDGenes <- unique(IDGenes)
  ExprSubset <- GetAssayData(expression[IDGenes,], assay.type = "RNA", slot = "counts")
  Sig <- c()
   for(celltype in IDsCells){
     Sig <- cbind(Sig,(apply(ExprSubset,1,function(y) mean(y[which(clusters==celltype)])))) #to try both median and mean
   }
  colnames(Sig) <- IDsCells
  write.table(Sig,file.path(output_dir,paste0("Signature_",NameSig,"_",diff.cutoff,"_",pvaladj.cutoff,"_",minG,"_",maxG,".txt")),sep="\t",row.names=TRUE,col.names=NA)
  png(file.path(output_dir,paste0("Signature_",NameSig,"_",diff.cutoff,"_",pvaladj.cutoff,"_",minG,"_",maxG,".png")),units='mm',height=550,res = 300)
    a <- heatmap(Sig)
  dev.off()
  return(Sig)
}


