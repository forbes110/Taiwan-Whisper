"""
Merge multiple data source(tsv files), first add src_id column for each tsv to ensure the splits will maintain source balanced.
The valid dataset sould be 
1. 0.1 of the duration of original train dataset
2. The distribution of the valid dataset over sources should be the same as the distribution of train dataset.
"""