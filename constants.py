"""Constants and global variables to run the different deconv files."""

## Constants for run_mixupvi.py
TUNE_MIXUPVI = True
TRAINING_DATASET = "CTI"  # ["CTI", "TOY", "CTI_PROCESSED", "CTI_RAW"]
TRAINING_CELL_TYPE_GROUP = (
    "2nd_level_granularity"  # ["1st_level_granularity", "2nd_level_granularity", "3rd_level_granularity", "4th_level_granularity", "FACS_1st_level_granularity"]
)
N_GENES = 4000 # number of input genes after preprocessing (NOT USED INSIDE BENCHMARK)

## Constants for both run_mixupvi.py and run_benchmark.py
# For all VI models
LATENT_SIZE = 30
MAX_EPOCHS = 100
SAVE_MODEL = False
SEED = 3
# MixUpVI training hyperparameters
BATCH_SIZE = 1024
TRAIN_SIZE = 0.7 # as opposed to validation
CHECK_VAL_EVERY_N_EPOCH = None
if TRAIN_SIZE < 1:
    CHECK_VAL_EVERY_N_EPOCH = 1
# MixUpVI model hyperparameters
N_PSEUDOBULKS = 100
N_CELLS_PER_PSEUDOBULK = 256 # None (then will be batch size) or int (will cap at batch size)
N_HIDDEN = 512
CONT_COV = None  # None or list of continuous covariates to include
CAT_COV = None # None or ["donor_id", "assay"]
ENCODE_COVARIATES = False # whether to encode cont/cat covars (they are always decoded)
LOSS_COMPUTATION = "latent_space"  # ["latent_space", "reconstructed_space"]
PSEUDO_BULK = "pre_encoded"  # ["pre_encoded", "post_inference"]
SIGNATURE_TYPE = "post_inference"  # ["pre_encoded", "post_inference"]
MIXUP_PENALTY = "l2"  # ["l2", "kl"]
DISPERSION = "gene"  # ["gene", "gene_label"]
GENE_LIKELIHOOD = "zinb"  # ["zinb", "nb", "poisson"]
USE_BATCH_NORM = "none"  # ["encoder", "decoder", "none", "both"]

# Other : different possibilities of cell groupings with the CTI dataset
GROUPS = {
    "1st_level_granularity": {
        "B": [
            "ABCs",
            "GC_B (I)",
            "GC_B (II)",
            "Memory B cells",
            "Naive B cells",
            "Plasma cells",
            "Plasmablasts",
            "Pre-B",
            "Pro-B",
        ],
        "MonoMacro": [
            "Alveolar macrophages",
            "Classical monocytes",
            "Erythrophagocytic macrophages",
            "Intermediate macrophages",
            "Nonclassical monocytes",
        ],
        "TNK": [
            "Cycling T&NK",
            "MAIT",
            "NK_CD16+",
            "NK_CD56bright_CD16-",
            "T_CD4/CD8",
            "Teffector/EM_CD4",
            "Tem/emra_CD8",
            "Tfh",
            "Tgd_CRTAM+",
            "Tnaive/CM_CD4",
            "Tnaive/CM_CD4_activated",
            "Tnaive/CM_CD8",
            "Tregs",
            "Trm/em_CD8",
            "Trm_Tgd",
            "Trm_Th1/Th17",
            "Trm_gut_CD8",
            "ILC3",
        ],
        "DC": ["DC1", "DC2", "migDC", "pDC"],
        "Mast": ["Mast cells"],
        "To remove": [
            "Erythroid",
            "Megakaryocytes",
            "Progenitor",
            "Cycling",
            "T/B doublets",
            "MNP/B doublets",
            "MNP/T doublets",
            "Intestinal macrophages",
        ],
    },
    "2nd_level_granularity": {
        "B": ["ABCs", "GC_B (I)", "GC_B (II)", "Memory B cells", "Naive B cells",
            "Pre-B", "Pro-B"],
        "Plasma": ["Plasma cells", "Plasmablasts"],
        "Mono": ["Classical monocytes", "Nonclassical monocytes"],
        "CD8T": ["Tem/emra_CD8", "Tnaive/CM_CD8", "Trm/em_CD8", "Trm_gut_CD8"],
        "CD4T":["Teffector/EM_CD4", "Tfh", "Tnaive/CM_CD4", "Tnaive/CM_CD4_activated", "Trm_Th1/Th17"],
        "Tregs":["Tregs"],
        "NK": ["NK_CD16+", "NK_CD56bright_CD16-"],
        "DC": ["DC1", "DC2", "migDC", "pDC"],
        "Mast": ["Mast cells"],
        "To remove": ["Cycling", "T/B doublets", "Cycling T&NK", "MNP/B doublets",
                      "MNP/T doublets", "Alveolar macrophages",
                      "Erythrophagocytic macrophages", "Intermediate macrophages",
                      "Intestinal macrophages", "ILC3", "MAIT","T_CD4/CD8","Tgd_CRTAM+",
                      "Trm_Tgd", "Erythroid", "Megakaryocytes", "Progenitor"],
    },
    "3rd_level_granularity": {
        "ImmatureB": ["Pre-B", "Pro-B"],
       	"NaiveB": [ "Naive B cells"],
        "MemB": [ "Memory B cells"],
        "Plasma": ["Plasma cells", "Plasmablasts"],
        "Mono": ["Classical monocytes", "Nonclassical monocytes"],
        "Macro":["Alveolar macrophages","Erythrophagocytic macrophages", 
                 "Intermediate macrophages", "Intestinal macrophages"],
        "Naive_CD8T": [ "Tnaive/CM_CD8"],
        "Mem_CD8T": ["Tem/emra_CD8", "Trm/em_CD8", "Trm_gut_CD8"],
        "Naive_CD4T":[ "Tfh", "Tnaive/CM_CD4", "Tnaive/CM_CD4_activated"],
        "Mem_CD4T":["Teffector/EM_CD4", "Trm_Th1/Th17"],
        "Tregs":["Tregs"],
        "gdT":["Tgd_CRTAM+", "Trm_Tgd"],
        "NK": ["NK_CD16+", "NK_CD56bright_CD16-"],
        "DC": ["DC1", "DC2", "migDC"],
        "pDC": ["pDC"],
        "Mast": ["Mast cells"],
        "To remove": ["ABCs", "GC_B (I)", "GC_B (II)","Cycling", "T/B doublets", 
                      "Cycling T&NK", "MNP/B doublets", "MNP/T doublets", "ILC3", 
                      "MAIT","T_CD4/CD8", "Erythroid", "Megakaryocytes", "Progenitor"],

    },
    "4th_level_granularity": {
        "immatureB": ["Pre-B", "Pro-B"],
       	"naiveB": [ "Naive B cells"],
        "memB": [ "Memory B cells"],
        "Plasma_cells": ["Plasma cells"],
        "Plasmablasts": ["Plasmablasts"],
        "Classical_monocytes": ["Classical monocytes"],
        "Non_Classical_monocytes": [ "Nonclassical monocytes"],
        "Macro":["Alveolar macrophages","Erythrophagocytic macrophages", "Intermediate macrophages",
                 "Intestinal macrophages"],
        "naive_CD8T": [ "Tnaive/CM_CD8"],
        "mem_CD8T": ["Tem/emra_CD8", "Trm/em_CD8", "Trm_gut_CD8"],
        "naive_CD4T":[ "Tfh", "Tnaive/CM_CD4", "Tnaive/CM_CD4_activated"],
        "mem_CD4T":["Teffector/EM_CD4", "Trm_Th1/Th17"],
        "Tregs":["Tregs"],
        "gdT":["Tgd_CRTAM+", "Trm_Tgd"],
        "NK_CD16_plus": ["NK_CD16+"],
        "NK_CD16_minus": [ "NK_CD56bright_CD16-"],
        "DC1": ["DC1"],
        "DC2": [ "DC2"],
        "migDC": [ "migDC"],
        "pDC": ["pDC"],
        "Mast": ["Mast cells"],
        "To remove": ["ABCs", "GC_B (I)", "GC_B (II)","Cycling", "T/B doublets", 
                      "Cycling T&NK", "MNP/B doublets", "MNP/T doublets", "ILC3", 
                      "MAIT","T_CD4/CD8", "Erythroid", "Megakaryocytes", "Progenitor"],
    },
    "FACS_1st_level_granularity": {
        "B": ["Pre-B", "Pro-B", "Naive B cells","Memory B cells","Plasma cells"],
        "NK": ["NK_CD16+", "NK_CD56bright_CD16-"],  
        "T": [ "Tnaive/CM_CD8","Tem/emra_CD8", "Trm/em_CD8", "Trm_gut_CD8","Tfh", 
              "Tnaive/CM_CD4", "Tnaive/CM_CD4_activated", "Teffector/EM_CD4", 
              "Trm_Th1/Th17","Tregs","T_CD4/CD8","Tgd_CRTAM+", "Trm_Tgd","MAIT"], 
        "Mono": ["Classical monocytes", "Nonclassical monocytes"], 
        "DC": ["DC1", "DC2", "migDC", "pDC"],
        "To remove":["Plasmablasts","ABCs", "GC_B (I)", "GC_B (II)","Cycling", 
                     "T/B doublets", "Cycling T&NK", "MNP/B doublets", "MNP/T doublets",
                     "ILC3", "Erythroid", "Megakaryocytes", "Progenitor",
                     "Alveolar macrophages","Erythrophagocytic macrophages", 
                     "Intermediate macrophages", "Intestinal macrophages","Mast cells"]
    }
}
