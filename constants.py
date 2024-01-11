"""Constants and global variables to run the different deconv files."""


## constants for run_mixupvi.py and benchmark_utils/training_utils.py
# MixUpVI training constants
TUNE_MIXUPVI = True
SAVE_MODEL = False
PATH = "/home/owkin/project/scvi_models/models/test_run"
TRAINING_DATASET = "CTI"  # ["CTI", "TOY", "CTI_PROCESSED", "CTI_RAW"]
TRAINING_LOG = True # whether to log transform the data
MAX_EPOCHS = 100
BATCH_SIZE = 2048
TRAIN_SIZE = 0.7 # as opposed to validation
if TRAIN_SIZE < 1:
    CHECK_VAL_EVERY_N_EPOCH = 1
# MixUpVI specific constants and constraints
TRAINING_CELL_TYPE_GROUP = (
    "updated_granular_groups"  # ["primary_groups", "precise_groups", "updated_granular_groups"]
)
CONT_COV = None  # list of continuous covariates to include
ENCODE_COVARIATES = False  # should be always False for now, we don't encode cat covar
ENCODE_CONT_COVARIATES = False  # True or False, whether to include cont covar
SIGNATURE_TYPE = "pre_encoded"  # ["pre_encoded", "post_inference"]
USE_BATCH_NORM = "none"  # ["encoder", "decoder", "none", "both"]
LOSS_COMPUTATION = "latent_space"  # ["latent_space", "reconstructed_space"]
PSEUDO_BULK = "pre_encoded"  # ["pre_encoded", "post_inference"]
MIXUP_PENALTY = "l2"  # ["l2", "kl"]
DISPERSION = "gene"  # ["gene", "gene_cell"]
GENE_LIKELIHOOD = "zinb"  # ["zinb", "nb", "poisson"]

## constants for run_pseudobulk_benchmark.py
SIGNATURE_CHOICE = "crosstissue_granular_updated" # ["laughney", "crosstissue_general", "crosstissue_granular_updated"]
if SIGNATURE_CHOICE in ["laughney", "crosstissue_general"]:
    BENCHMARK_CELL_TYPE_GROUP = "primary_groups"
elif SIGNATURE_CHOICE == "crosstissue_granular_updated":
    BENCHMARK_CELL_TYPE_GROUP = "updated_granular_groups"
else:
    BENCHMARK_CELL_TYPE_GROUP = None # no signature was created for the "precise_groups" grouping right now
BENCHMARK_DATASET = "CTI"  # ["CTI", "TOY", "CTI_PROCESSED", "CTI_RAW"]
BENCHMARK_LOG = False # whether to log transform the data - nb it wasn't done for the signature
N_CELLS = 2000 # number of cells for creation of pseudobulk and latent pseudobulk
N_SAMPLES = 400 # number of pseudbulk samples to create and assess for deconvolution
ONLY_FIT_BASELINE_NNLS = False # if True, don't fit the deep learning models


# different possibilities of cell groupings with the CTI dataset
GROUPS = {
    "primary_groups": {
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
    "precise_groups": {
        "B": [
            "ABCs",
            "GC_B (I)",
            "GC_B (II)",
            "Memory B cells",
            "Naive B cells",
            "Pre-B",
            "Pro-B"
        ],
        "Plasma": ["Plasma cells", "Plasmablasts"],
        "Mono": ["Classical monocytes", "Nonclassical monocytes"],
        "CD8T": ["Tem/emra_CD8", "Tnaive/CM_CD8", "Trm/em_CD8", "Trm_gut_CD8"],
        "CD4T":["Teffector/EM_CD4", "Tfh", "Tnaive/CM_CD4", "Tnaive/CM_CD4_activated", "Tregs",
                 "Trm_Th1/Th17"],
        "T": ["MAIT","T_CD4/CD8","Tgd_CRTAM+","Trm_Tgd"],
        "NK": ["NK_CD16+", "NK_CD56bright_CD16-"],
        "DC": ["DC1", "DC2", "migDC", "pDC"],
        "Mast": ["Mast cells"],
        "RedBlood": ["Erythroid"],
        "BoneMarrow": ["Megakaryocytes"],
        "NonDifferentiated": ["Progenitor"],
        "To remove":  ["Cycling", "T/B doublets", "Cycling T&NK",
            "MNP/B doublets", "MNP/T doublets","Alveolar macrophages",
            "Erythrophagocytic macrophages",
            "Intermediate macrophages",
            "Intestinal macrophages", "ILC3"],
    },
    "updated_granular_groups": {
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
    }
}
