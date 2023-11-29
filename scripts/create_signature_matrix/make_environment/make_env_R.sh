## CREATION OF ENVIRONMENT: 

# Initialize conda in terminal and choose shell (bash or zsh. To checkout available shells simply enter conda init)
conda init bash
exec bash

### Generate R / python stable environment
env_name="R_environnement"



### Create environment and install mamba

# OPTION 1: in a specicic folder (ONLY do if enough space on workspace)
# This option avoids loss of environment when server crashes
conda create -c conda-forge --yes --prefix /home/owkin/.conda/envs/$env_name mamba 

# Activate environment: 
conda activate /home/owkin/.conda/envs/$env_name

# To avoid long path of environment each time it is activated:
conda config --set env_prompt '({name})'



# OPTION 2: If not wishing to create environment in specific path:
#conda create -c conda-forge --yes -n $env_name mamba
#conda activate $env_name



# Add conda forge and bioconda channels
conda config --add channels conda-forge
conda config --add channels bioconda

# Set conda channel priorities:

# Keep flexible:
conda config --set channel_priority flexible

# Otherwise: If using snakemake omics pre-processing pipeline:
#conda config --set channel_priority strict

### INSTALL R PACKAGES

## Basic packages
mamba install --yes r-base
mamba install --yes radian
conda install --yes -c r r-jsonlite
mamba install --yes r-languageserver

## Very important to install IRkernel manually in R (see bottom of this script)

### Specific pacakges: (choose the ones adapted to your project)
mamba install --yes bioconductor-fgsea
mamba install --yes r-msigdbr
mamba install --yes bioconductor-deseq2
mamba install --yes r-circlize 
mamba install --yes r r-tidyverse     
mamba install --yes bioconductor-consensusclusterplus
mamba install --yes bioconductor-complexheatmap
mamba install --yes bioconductor-genomeinfodb
mamba install --yes r-ggpubr
mamba install --yes r r-gridextra
mamba install --yes r-gert
mamba install --yes r-dplyr
mamba install --yes bioconductor-enhancedvolcano
mamba install --yes r-survminer
mamba install --yes bioconductor-tximport
mamba install --yes r-ggplot2
mamba install --yes r-ggrepel
mamba install --yes r-devtools
mamba install --yes r-ggpmisc
mamba install --yes r-units
mamba install --yes bioconductor-genomicranges
mamba install --yes bioconductor-ensdb.hsapiens.v86
mamba install --yes bioconductor-biomart
mamba install --yes bioconductor-limma
mamba install --yes bioconductor-edger
mamba install --yes r-rocr
mamba install --yes bioconductor-gsva
mamba install --yes r-mcpcounter
mamba install --yes r-survival
mamba install --yes conda-forge r-survminer
mamba install --yes bioconductor-category
mamba install --yes bioconductor-clusterprofiler
mamba install --yes bioconductor-reactomepa
mamba install --yes bioconductor-org.hs.eg.db
mamba install --yes r-fpc
mamba install --yes r r-nmf
mamba install --yes r-bedr
mamba install --yes bioconductor-bsgenome.hsapiens.ucsc.hg19
mamba install --yes bioconductor-bsgenome.hsapiens.ucsc.hg38
mamba install --yes r-httpgd
mamba install --yes bedtools
mamba install --yes r-venn
mamba install --yes r-ggridges


### UPDATE VScode settings to better handle printing of R graphics etc
# Important: NEED TO CHECK WHERE IS YOUR JSON SETTINGS FILE
# Note: Can also manually modify settings.json file inside vscode
VS_settings="/home/owkin/.vscode-R/settings.json"

echo "{" > $VS_settings
echo "    \"workbench.colorTheme\": \"Quiet Light\"," >> $VS_settings
echo "    \"debug.internalConsoleOptions\": \"neverOpen\"," >> $VS_settings
echo "    \"notebook.displayOrder\": [\"image/png\", \"text/markdown\", \"text/plain\"]," >> $VS_settings
echo "    \"editor.multiCursorModifier\": \"ctrlCmd\"," >> $VS_settings
echo "    \"workbench.editorAssociations\": [" >> $VS_settings
echo "        {" >> $VS_settings
echo "            \"viewType\": \"jupyter-notebook\"," >> $VS_settings
echo "            \"filenamePattern\": \"*.ipynb\"" >> $VS_settings
echo "        }" >> $VS_settings
echo "    ]," >> $VS_settings
echo "    \"r.rpath.linux\": \"/home/owkin/.conda/envs/${env_name}/bin/R\"," >> $VS_settings
echo "    \"r.rterm.windows\": \"/home/owkin/.conda/envs/${env_name}/bin/R\"," >> $VS_settings
echo "    \"r.rterm.mac\": \"/home/owkin/.conda/envs/${env_name}/bin/R\"," >> $VS_settings
echo "    \"r.sessionWatcher\": true," >> $VS_settings
echo "    \"r.bracketedPaste\": true," >> $VS_settings
echo "    \"r.rterm.linux\": \"/home/owkin/.conda/envs/${env_name}/bin/radian\"," >> $VS_settings
echo "    \"r.lsp.path\": \"/home/owkin/.conda/envs/${env_name}/bin/R\"," >> $VS_settings
echo "    \"r.lsp.debug\": true," >> $VS_settings
echo "    \"r.lsp.diagnostics\": true," >> $VS_settings
echo "    \"r.rterm.option\": [" >> $VS_settings
echo "    \"--no-save\"," >> $VS_settings
echo "    \"--no-restore\"," >> $VS_settings
echo "    \"--r-binary=/home/owkin/.conda/envs/${env_name}/bin/R\"," >> $VS_settings
echo "    \"r.plot.useHttpgd\": true" >> $VS_settings
echo "    ]," >> $VS_settings
echo "}" >> $VS_settings


## INSTALL PYTHON PACKAGES
### Basic packages
mamba install --yes ipywidgets
mamba install --yes -c anaconda jupyter
mamba install --yes -c anaconda ipykernel

### Specific
mamba install --yes pandas numpy plotnine 
mamba install --yes patchwork #useful library for layouts of plots like ggplot
mamba install --yes boto3 # retrieval of aws buckets




### MANUAL INSTALLATION OF R PACKAGES SO THAT R WORKS IN NOTEBOOKS 
### Note that it only work on this version of R, select this kernel in R notebooks
#/workspace/envs/${env_name}/bin/Rscript -e "install.packages('IRkernel', repos='http://cran.us.r-project.org')"
#/workspace/envs/${env_name}/bin/Rscript -e "IRkernel::installspec()"
#/workspace/envs/${env_name}/bin/Rscript -e "install.packages('languageserver', repos='http://cran.us.r-project.org')"

##DEBUGGING: If having problems installing some of the packages above:

# If not able to install IRKernel from R: 
#conda install --yes -c conda-forge r-irkernel

# Problem to install IRkernel::installspec()" : need to install first jupyter client and genutils
#conda install --yes -c esri jupyter_client
#pip install --upgrade setuptools pip
#pip install ipython_genutils

# Problems to select R kernel do the following: 
#python3 -m pip install ipykernel
#python3 -m ipykernel install --user