DATASET_CONFIG={
    "unsw_nb15": {
        "file_path": "datasets/RAW/UNSW-NB15/UNSW_NB15_training-set.csv",
        "label_column": "label",
        "columns": None,
         "categorical_columns": [
            "proto",
            "service",
            "state"
        ],
         "drop_columns": [
            "id",
            "attack_cat"
        ],
         "target_type": "binary",

        # UNSW already has:
        # 0 = normal
        # 1 = attack
        "target_already_numeric": True,
    },
    "cic2017": {
        "file_path": "datasets/RAW/CIC-IDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "label_column": " Label",
        "columns": None,
        "categorical_columns": [],
        "drop_columns": [],
        "target_type": "binary",
        "target_already_numeric": False,
        "normal_labels": ["BENIGN"],
    }     
}