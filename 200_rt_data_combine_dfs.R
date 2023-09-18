# # ---------------------------------------------------------------------------
# # Title:    Combine RT data files
# # Contains: Code to combine individual RT data files into a big dataframe
# # Project:  DPX-EEG
# # Author:   Jose C. Garcia Alanis (alanis.jcg@gmail.com)
# # Date:     Mon Sep 18 12:06:31 2023
# # ---------------------------------------------------------------------------

# set working directory to current directory
if(interactive()){
  path <- dirname(rstudioapi::getSourceEditorContext()$path)
} else {
  path <- normalizePath('./')
}
setwd(path)

# get utility functions
source('utils.R')

# load nesserary packages
load.package(c('rjson', 'dplyr'))

# get paths to data
paths <- fromJSON(file="paths.json")

# location of RT files
rt_files <- Sys.glob(
  paste(paths$bids, 'derivatives', 'rt/sub-*/*.tsv', sep = '/')
)

# import the data
rt_list <- lapply(rt_files, read.table, sep = '\t', header = T)

# row bind RT data.frames
rt_df <- bind_rows(rt_list, .id = NULL)

# recode block variable
rt_df <- rt_df %>%
  mutate(block =  ifelse(block == 0, 1, 2)) %>%
  mutate(block = factor(block, labels = c('Block 1', 'Block 2')))

# save RT data.frame
fpath <- paste(paths$bids, 'derivatives', 'behavioural_analysis', sep = '/')
dir.create(fpath)
save(rt_df, file = paste(fpath, 'RT.RData', sep = '/'))
