"""Constants and global variables to run the different deconv files."""

## constants for run_mixupvi.py
TUNE_MIXUPVI = False
TRAINING_DATASET = "CTI_PROCESSED"  # ["CTI", "TOY", "CTI_PROCESSED", "CTI_RAW"]
TRAINING_CELL_TYPE_GROUP = (
    "2nd_level_granularity"  # ["1st_level_granularity", "2nd_level_granularity", "3rd_level_granularity", "4th_level_granularity", "FACS_1st_level_granularity"]
)

## constants for run_pseudobulk_benchmark.py
SIGNATURE_CHOICE = "CTI_2nd_level_granularity" # ["laughney", "CTI_1st_level_granularity", "CTI_2nd_level_granularity", "CTI_3rd_level_granularity", "CTI_4th_level_granularity", "FACS_1st_level_granularity"]
if SIGNATURE_CHOICE in ["laughney", "CTI_1st_level_granularity"]:
    BENCHMARK_CELL_TYPE_GROUP = "1st_level_granularity"
elif SIGNATURE_CHOICE == "CTI_2nd_level_granularity":
    BENCHMARK_CELL_TYPE_GROUP = "2nd_level_granularity"
elif SIGNATURE_CHOICE == "CTI_3rd_level_granularity":
    BENCHMARK_CELL_TYPE_GROUP = "3rd_level_granularity"
elif SIGNATURE_CHOICE == "CTI_4th_level_granularity":
    BENCHMARK_CELL_TYPE_GROUP = "4th_level_granularity"
elif SIGNATURE_CHOICE == "CTI_4th_level_granularity":
    BENCHMARK_CELL_TYPE_GROUP = "FACS_1st_level_granularity"
else:
    BENCHMARK_CELL_TYPE_GROUP = None # no signature was created
BENCHMARK_DATASET = "CTI"  # ["CTI", "TOY", "CTI_PROCESSED", "CTI_RAW"]
N_SAMPLES = 500 # number of pseudbulk samples to create and assess for deconvolution
GENERATIVE_MODELS = ["MixupVI"] #, "DestVI"] # "scVI", "CondscVI", "DestVI"
# GENERATIVE_MODELS = [] # if only want baselines
BASELINES = ["nnls"] # "nnls", "TAPE", "Scaden"
# BASELINES = ["nnls"] # if only want nnls

## general mixupvi constants when training it or preprocessing data
SAVE_MODEL = False
N_GENES = 3000 # number of input genes after preprocessing
# MixUpVI training hyperparameters
MAX_EPOCHS = 100
BATCH_SIZE = 4092
TRAIN_SIZE = 0.7 # as opposed to validation
CHECK_VAL_EVERY_N_EPOCH = None
if TRAIN_SIZE < 1:
    CHECK_VAL_EVERY_N_EPOCH = 1
# MixUpVI model hyperparameters
CONT_COV = None  # list of continuous covariates to include
CAT_COV = ["donor_id", "assay"] # list of categorical covariates to include
ENCODE_COVARIATES = True # whether to encode cont/cat covars (they are always decoded)
SIGNATURE_TYPE = "post_inference"  # ["pre_encoded", "post_inference"]
USE_BATCH_NORM = "none"  # ["encoder", "decoder", "none", "both"]
LOSS_COMPUTATION = "latent_space"  # ["latent_space", "reconstructed_space"]
PSEUDO_BULK = "pre_encoded"  # ["pre_encoded", "post_inference"]
MIXUP_PENALTY = "l2"  # ["l2", "kl"]
DISPERSION = "gene"  # ["gene", "gene_cell"]
GENE_LIKELIHOOD = "zinb"  # ["zinb", "nb", "poisson"]
LATENT_SIZE = 30

# different possibilities of cell groupings with the CTI dataset
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
        "Plasma cells": ["Plasma cells"],
        "Plasmablasts": ["Plasmablasts"],
        "Classical monocytes": ["Classical monocytes"],
        "Non-Classical monocytes": [ "Nonclassical monocytes"],
        "Macro":["Alveolar macrophages","Erythrophagocytic macrophages", "Intermediate macrophages",
                 "Intestinal macrophages"],
        "naive_CD8T": [ "Tnaive/CM_CD8"],
        "mem_CD8T": ["Tem/emra_CD8", "Trm/em_CD8", "Trm_gut_CD8"],
        "naive_CD4T":[ "Tfh", "Tnaive/CM_CD4", "Tnaive/CM_CD4_activated"],
        "mem_CD4T":["Teffector/EM_CD4", "Trm_Th1/Th17"],
        "Tregs":["Tregs"],
        "gdT":["Tgd_CRTAM+", "Trm_Tgd"],
        "NK_CD16+": ["NK_CD16+"],
        "NK_CD16-": [ "NK_CD56bright_CD16-"],
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
