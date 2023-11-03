"""Different python functions useful for sanity checks in deconvolution."""

import anndata as ad
import pandas as pd
import mygene
from ..constants import GROUPS

def create_signature(adata: ad.AnnData,
                    signature_type: str = "almudena",
                     group: str = "primary_groups"):
  """Create the signature matrix from the single cell dataset."""
  if signature_type in ["laughney", "crosstissue_general"] \
        and group != "primary_groups":
    raise ValueError("Incompatabile number of cell types between the signature matrix and the grouping chosen for the dataset.")
  if signature_type == "laughney":
    signature = pd.read_csv(
        "/home/owkin/project/laughney_signature.csv", index_col=0
    ).drop(["Endothelial", "Malignant", "Stroma", "Epithelial"], axis=1)
  elif signature_type == "almudena":
    signature = read_almudena_signature(
        "/home/owkin/project/Almudena/Output/Crosstiss_Immune_norm/CTI.txt"
    )  # it is the normalised one (using adata.X and not adata.raw.X, to match this code)
  # elif signature_type == "crosstissue_general":
  elif signature_type == "crosstissue_granular_updated":
    signature = read_almudena_signature(
      "/home/owkin/project/Simon/CTI_granular_updated_ensg.txt"
      )
  if signature_type == "laughney":
      # map the HGNC notation to ENSG if the signature matrix uses HGNC notation
      mg = mygene.MyGeneInfo()
      genes = mg.querymany(
          signature.index,
          scopes="symbol",
          fields=["ensembl"],
          species="human",
          verbose=False,
          as_dataframe=True,
      )
      ensg_names = map_hgnc_to_ensg(genes, adata)
      signature = signature_type.copy()
      signature.index = ensg_names
  elif signature_type in set("crosstissue_general", "crosstissue_granular_updated"):
      signature = signature.copy()
  # intersection between all genes and marker genes
  intersection = list(set(adata.var_names).intersection(signature.index))
  signature = signature.loc[intersection]
  return signature


def add_cell_types_grouped(adata: ad.Anndata,
                           group: str = "primary_groups"):
  """Add the cell types grouped columns in Anndata according to the grouping choice."""
  groups = GROUPS[group]
  group_correspondence = {}
  for k, v in groups.items():
      for cell_type in v:
          group_correspondence[cell_type] = k

  adata.obs["cell_types_grouped"] = [
      group_correspondence[cell_type] for cell_type in adata.obs.Manually_curated_celltype
  ]
  index_to_keep = adata.obs.loc[adata.obs["cell_types_grouped"] != "To remove"].index
  adata = adata[index_to_keep]


def read_almudena_signature(path):
    """Read Almudena's signature matrix. Requires this function because it's a txt file
    delimited with various delimiters.
    """
    signature_almudena = []
    with open(path, "r") as file:
        for line in file:
            temp = []
            for elem in line.split("\t"):
                try:
                    temp.append(float(elem))
                except:
                    elem = elem.replace('"', "")
                    elem = elem.replace("\n", "")
                    temp.append(elem)
            signature_almudena.append(temp)

    signature_almudena = pd.DataFrame(signature_almudena).set_index(0)
    signature_almudena.columns = signature_almudena.iloc[0]
    signature_almudena = signature_almudena.drop("")
    signature_almudena.index.name = "Genes"
    signature_almudena.columns.name = None
    return signature_almudena


def map_hgnc_to_one_ensg(gene_names, adata):
    """
    If a HGNC symbol map to multiple ENSG symbols, choose the one that is in the
    single cell dataset.
    If the HGNC symbol maps to multiple ENSG symbols even inside the scRNAseq dataset,
    then the last one is chosen (no rationale).
    """
    chosen_gene = None
    for gene_name in gene_names:
        if gene_name in adata.var_names:
            chosen_gene = gene_name
    return chosen_gene


def map_hgnc_to_ensg(genes, adata):
    """
    Map the HGNC symbols from the signature matrix to the corresponding ENSG symbols
    of the scRNAseq dataset.
    """
    ensg_names = []
    for gene in genes.index:
        if len(genes.loc[gene].shape) > 1:
            # then one hgnc has multiple ensg lines in the dataframe
            gene_names = genes.loc[gene, "ensembl.gene"]
            gene_name = map_hgnc_to_one_ensg(gene_names, adata)
            if gene_name not in ensg_names:  # for duplicates
                ensg_names.append(gene_name)
        elif genes.loc[gene, "notfound"] is True:
            # then the hgnc gene cannot be mapped to ensg
            ensg_names.append("notfound")
        elif genes.loc[gene, "ensembl.gene"] != genes.loc[gene, "ensembl.gene"]:
            # then one hgnc gene has multiple ensg mappings in one line of the dataframe
            ensembl = genes.loc[gene, "ensembl"]
            gene_names = [ensembl[i]["gene"] for i in range(len(ensembl))]
            gene_name = map_hgnc_to_one_ensg(gene_names, adata)
            ensg_names.append(gene_name)
        else:
            # then one hgnc corresponds to one ensg
            ensg_names.append(genes.loc[gene, "ensembl.gene"])
    return ensg_names
