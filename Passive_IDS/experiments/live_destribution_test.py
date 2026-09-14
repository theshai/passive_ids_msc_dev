import pandas as pd

# ============================================================
# LIVE SAMPLE TO ANALYZE
# ============================================================

live_sample = {
    "dur": 26.983758,
    "proto": "tcp",
    "service": "-",
    "state": "FIN",
    "spkts": 68,
    "dpkts": 229,
    "sbytes": 4923,
    "dbytes": 340246,
    "rate": 11.00662109406703,
    "dttl": 121,
    "smean": 72.3970588235294,
    "dmean": 1485.7903930131004,
    "sinpkt": 402.74234328358204,
    "dinpkt": 117.5579956140351,
    "sjit": 2102.779964636061,
    "djit": 1154.81604804538,
    "sloss": 0,
    "dloss": 8,
    "sload": 1459.5446638677977,
    "dload": 100874.31113190386,
    "stcpb": 916922441,
    "dtcpb": 2344104176,
    "synack": 0.160463,
    "ackdat": 0.001851,
    "tcprtt": 0.162314,
    "is_sm_ips_ports": 0,
    "response_body_len": 0
}


# ============================================================
# LOAD UNSW DATASET
# ============================================================

def load_dataset():

    # --------------------------------------------------------
    # CHANGE THIS SECTION TO MATCH YOUR EXISTING LOADER
    # --------------------------------------------------------
    #
    # Example if you want to load directly:
    #
    # df = pd.read_csv(
    #     "datasets/RAW/UNSW-NB15/UNSW_NB15_training-set.csv"
    # )
    #
    # return df
    #
    # --------------------------------------------------------

    dataset_path = (
        "datasets/RAW/UNSW-NB15/"
        "UNSW_NB15_training-set.csv"
    )

    df = pd.read_csv(dataset_path)

    return df


# ============================================================
# NUMERIC FEATURE DISTRIBUTION
# ============================================================

def compare_numeric_features(
    df,
    live_sample
):

    numeric_features = [
        "dur",
        "spkts",
        "dpkts",
        "sbytes",
        "dbytes",
        "rate",
        "dttl",
        "sload",
        "dload",
        "sloss",
        "dloss",
        "sinpkt",
        "dinpkt",
        "sjit",
        "djit",
        "stcpb",
        "dtcpb",
        "tcprtt",
        "synack",
        "ackdat",
        "smean",
        "dmean",
        "response_body_len",
        "is_sm_ips_ports"
    ]

    normal_df = df[
        df["label"] == 0
    ]

    attack_df = df[
        df["label"] == 1
    ]

    print()
    print("=" * 140)
    print("NUMERIC FEATURE DISTRIBUTION COMPARISON")
    print("=" * 140)

    print(
        f"{'Feature':25}"
        f"{'Live':>15}"
        f"{'Normal Med':>15}"
        f"{'Normal P05':>15}"
        f"{'Normal P95':>15}"
        f"{'Attack Med':>15}"
        f"{'Attack P05':>15}"
        f"{'Attack P95':>15}"
    )

    print("-" * 140)

    for feature in numeric_features:

        if feature not in df.columns:
            print(
                f"{feature:25}"
                f"{'NOT IN DATASET':>15}"
            )
            continue

        if feature not in live_sample:
            print(
                f"{feature:25}"
                f"{'NOT IN LIVE':>15}"
            )
            continue

        live_value = live_sample[
            feature
        ]

        normal = pd.to_numeric(
            normal_df[feature],
            errors="coerce"
        ).dropna()

        attack = pd.to_numeric(
            attack_df[feature],
            errors="coerce"
        ).dropna()

        if len(normal) == 0 or len(attack) == 0:

            print(
                f"{feature:25}"
                f"{live_value:15.3f}"
                f"{'NO DATA':>15}"
            )

            continue

        print(
            f"{feature:25}"
            f"{live_value:15.3f}"
            f"{normal.median():15.3f}"
            f"{normal.quantile(0.05):15.3f}"
            f"{normal.quantile(0.95):15.3f}"
            f"{attack.median():15.3f}"
            f"{attack.quantile(0.05):15.3f}"
            f"{attack.quantile(0.95):15.3f}"
        )


# ============================================================
# CATEGORICAL FEATURE DISTRIBUTION
# ============================================================

def compare_categorical_features(
    df,
    live_sample
):

    categorical_features = [
        "proto",
        "service",
        "state"
    ]

    normal_df = df[
        df["label"] == 0
    ]

    attack_df = df[
        df["label"] == 1
    ]

    print()
    print("=" * 100)
    print("CATEGORICAL FEATURE DISTRIBUTION")
    print("=" * 100)

    for feature in categorical_features:

        if feature not in df.columns:
            print(
                f"{feature}: "
                f"NOT FOUND IN DATASET"
            )
            continue

        value = live_sample[
            feature
        ]

        normal_pct = (
            normal_df[feature]
            .astype(str)
            .value_counts(
                normalize=True
            )
            .get(
                str(value),
                0
            )
            * 100
        )

        attack_pct = (
            attack_df[feature]
            .astype(str)
            .value_counts(
                normalize=True
            )
            .get(
                str(value),
                0
            )
            * 100
        )

        print()
        print(
            f"{feature} = {value}"
        )

        print(
            f"  Normal records: "
            f"{normal_pct:.2f}%"
        )

        print(
            f"  Attack records: "
            f"{attack_pct:.2f}%"
        )


# ============================================================
# SIMPLE RANGE CHECK
# ============================================================

def check_live_position(
    df,
    live_sample
):

    numeric_features = [
        "dur",
        "spkts",
        "dpkts",
        "sbytes",
        "dbytes",
        "rate",
        "dttl",
        "sload",
        "dload",
        "sloss",
        "dloss",
        "sinpkt",
        "dinpkt",
        "sjit",
        "djit",
        "stcpb",
        "dtcpb",
        "tcprtt",
        "synack",
        "ackdat",
        "smean",
        "dmean",
        "response_body_len"
    ]

    normal_df = df[
        df["label"] == 0
    ]

    attack_df = df[
        df["label"] == 1
    ]

    print()
    print("=" * 100)
    print("LIVE VALUE POSITION CHECK")
    print("=" * 100)

    for feature in numeric_features:

        if (
            feature not in df.columns
            or feature not in live_sample
        ):
            continue

        live_value = live_sample[
            feature
        ]

        normal = pd.to_numeric(
            normal_df[feature],
            errors="coerce"
        ).dropna()

        attack = pd.to_numeric(
            attack_df[feature],
            errors="coerce"
        ).dropna()

        if (
            len(normal) == 0
            or len(attack) == 0
        ):
            continue

        normal_p05 = normal.quantile(
            0.05
        )

        normal_p95 = normal.quantile(
            0.95
        )

        attack_p05 = attack.quantile(
            0.05
        )

        attack_p95 = attack.quantile(
            0.95
        )

        normal_inside = (
            normal_p05
            <= live_value
            <= normal_p95
        )

        attack_inside = (
            attack_p05
            <= live_value
            <= attack_p95
        )

        if (
            normal_inside
            and not attack_inside
        ):

            result = (
                "MORE NORMAL-LIKE"
            )

        elif (
            attack_inside
            and not normal_inside
        ):

            result = (
                "MORE ATTACK-LIKE"
            )

        elif (
            normal_inside
            and attack_inside
        ):

            result = (
                "OVERLAPS BOTH"
            )

        else:

            result = (
                "OUTSIDE BOTH 5-95%"
            )

        print(
            f"{feature:25}"
            f"live={live_value:15.3f}  "
            f"{result}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading UNSW-NB15 dataset..."
    )

    df = load_dataset()

    print(
        "Dataset shape:",
        df.shape
    )

    print(
        "Dataset columns:",
        df.columns.tolist()
    )

    print()

    print(
        "Label distribution:"
    )

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    compare_numeric_features(
        df,
        live_sample
    )

    compare_categorical_features(
        df,
        live_sample
    )

    check_live_position(
        df,
        live_sample
    )


if __name__ == "__main__":
    main()