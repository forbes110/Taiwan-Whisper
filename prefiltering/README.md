# Prefiltering
1. Run `common_hallicination_removal.sh` to check/remove data with common hallucinated vocabs from the result of `initial_inference.sh`
2. Run `validator_inference.sh` to generate predication of validator, "whisper-base"
3. Run `elim_hallucination.sh` to remove all hallucination by the result of validator and generate a cleaned dataset
4. Remove data source(e.g., different course) with high hallucination rate > threshold%
5. After data source selected, merge them with another column "source_id" to do train/valid splits with source balanced for KD, and then the test dataset would be done at performance-evaluation step, should be some open dataset like "common_voice".