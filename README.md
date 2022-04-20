# pod_pcgrl

1. If on mac, open a terminal and cd into project root and run `sh setup_data_dirs.sh`. This creates the folders for levels generated during inference and csv training data files created when gen_pod_target_distribution.py is run.
2. Next from the same terminal instance run `python3 gen_pod_target_distribution.py`to generate the training data csvs. 
3. Then from the same terminal instance run `python3 cnn_train.py`
4. Finally, run inference on the trained model from step 3 via running `python3 inference.py`
