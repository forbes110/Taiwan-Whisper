# Pseudo-Labelling Steps

1. Run `python3 filter_data.py` to make sure all sample rates are 16000 Hz.

2. Given a directory with lots of audios(e.g., *.flac, *.wav, *.m4a), make a tsv file "dataset_path" as first-step dataset metadata by `make_paths.py`.
for example: a `raw_data.csv` file with content:
    ```
    audio_path
    /mnt/dataset_1T/tmp_dir/example/lcMAHaXJflI.m4a
    /mnt/dataset_1T/tmp_dir/example/p8J-WHSz47E.m4a
    /mnt/dataset_1T/tmp_dir/example/q4HmpXp0I-g.m4a
    /mnt/dataset_1T/tmp_dir/example/StRDn_NWGvQ.m4a
    /mnt/dataset_1T/tmp_dir/example/ZmpMTBnCI4w.m4a
    /mnt/dataset_1T/tmp_dir/example/BN1YP9VIB08.m4a
    /mnt/dataset_1T/tmp_dir/example/n2LwQd_rZ5g.m4a
    ```
    /mnt/dataset_1T/tmp_dir/example/QZGURfv1DDQ.m4a

3. Run `bash initial_inference.sh` to get psuedo label with time stamps, 
    TODO: do detection of "Taiwanese Hokkien"(to remove)

4. Run `bash post_processing.sh` to change simplified to traditional chinese for pseudo-label.

5. Run `bash prepare_dataset.sh` to get segments with 30 secs for all data and make file to flac.(return data_pair dir)

6. Run `gen_metadata.sh` to generate a metadata.tsv of all audio pathes, note that the "valid-percent" need to be set to 0.(return audio_paths.scv)

