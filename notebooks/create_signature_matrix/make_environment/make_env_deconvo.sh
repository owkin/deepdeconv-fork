conda activate R_environnement

mamba install --yes -c conda-forge -c bioconda r-dwls bioconductor-preprocesscore bioconductor-pcamethods adapt xbioc r-scdc bioconductor-deconrnaseq r-tidyverse r-dplyr bioconductor-annotationdbi r-data.table 

R -e "install.packages('FARDEEP', repos='http://cran.us.r-project.org')"
R -e "install.packages('ADAPTS', repos='http://cran.us.r-project.org')"
R -e "install.packages('R.utils', repos='http://cran.us.r-project.org')"
R -e "install.packages('devtools', repos='http://cran.us.r-project.org')"
R -e "devtools::install_github('randel/MIND')"