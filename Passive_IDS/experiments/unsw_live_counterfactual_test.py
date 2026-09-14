import copy
import joblib
import pandas as pd


# ============================================================
# KNOWN-NORMAL LIVE FLOW
# This is the flow from your own website that was predicted
# as ATTACK with ~0.9947 probability.
# ============================================================

live_sample = {
    "dur": 115.740486,
    "proto": "tcp",
    "service": "-",
    "state": "FIN",
    "spkts": 18,
    "dpkts": 15,
    "sbytes": 1664,
    "dbytes": 3766,
    "rate": 0.2851206275390964,
    "smean": 92.44444444444444,
    "dmean": 251.06666666666666,
    "sinpkt": 6808.262647058823,
    "dinpkt": 8265.416285714286,
    "sjit": 4627.302081313769,
    "djit": 3696.4114326721183,
    "sloss": 10,
    "dloss": 0,
    "sload": 115.01593314546821,
    "dload": 260.30649292417866,
    "stcpb": 2702026681,
    "dtcpb": 2248056091,
    "synack": 0.023459,
    "ackdat": 0.007665,
    "tcprtt": 0.031124,
    "is_sm_ips_ports": 0,
    "response_body_len": 0
}


# ============================================================
# LOAD SAVED MODEL ARTIFACTS
# IMPORTANT:
# These 3 files must all come from the SAME training run,
# the one where sttl was removed.
# ============================================================

encoder = joblib.load(
    "models/live_encoder_for_unsenb15_wo_swin_dwin_sttl_dttl.joblib"
)

model = joblib.load(
    "models/random_forest_model_for_unswnb15_wo_swin_dwin_sttl_dttl.joblib"
)

selected_columns = joblib.load(
    "models/live_selected_columns_for_unsenb15_wo_swin_dwin_sttl_dttl.joblib"
)


# ============================================================
# NORMAL UNSW MEDIANS
# Taken from the distribution output you just generated.
# ============================================================

normal_medians = {
    "dur": 0.039,
    "spkts": 12,
    "dpkts": 10,
    "sbytes": 1470,
    "dbytes": 1112,
    "rate": 1382.257,
    #"dttl": 29,
    "sload": 431327.656,
    "dload": 336120.641,
    "sloss": 3,
    "dloss": 2,
    "sinpkt": 1.431,
    "dinpkt": 0.911,
    "sjit": 54.423,
    "djit": 19.226,
    "stcpb": 1180207127.5,
    "dtcpb": 1152046036.5,
    "tcprtt": 0.001,
    "synack": 0.001,
    "ackdat": 0.0,
    "smean": 73,
    "dmean": 89,
    "response_body_len": 0,
    "is_sm_ips_ports": 0
}


# ============================================================
# PREDICT ONE SAMPLE
# ============================================================

def predict_sample(sample):

    df = pd.DataFrame([sample])

    encoded = encoder.transform(df)

    # If encoder output is not already pandas
    if not isinstance(encoded, pd.DataFrame):

        encoded = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out()
        )

    selected = encoded[selected_columns]

    prediction = model.predict(selected)[0]

    probability = model.predict_proba(
        selected
    )[0][1]

    return prediction, probability


# ============================================================
# MAIN TEST
# ============================================================

def main():

    # --------------------------------------------------------
    # Original live prediction
    # --------------------------------------------------------

    prediction, base_probability = predict_sample(
        live_sample
    )

    print()
    print("=" * 90)
    print("ORIGINAL LIVE SAMPLE")
    print("=" * 90)

    print("Prediction:", prediction)
    print(
        "Attack probability:",
        f"{base_probability:.6f}"
    )

    print()
    print("=" * 130)
    print("ONE-FEATURE COUNTERFACTUAL TEST")
    print("=" * 130)

    print(
        f"{'Feature':25}"
        f"{'Live Value':>18}"
        f"{'Normal Median':>18}"
        f"{'Original Prob':>18}"
        f"{'New Prob':>18}"
        f"{'Change':>18}"
    )

    print("-" * 130)

    results = []

    # --------------------------------------------------------
    # Change one feature at a time
    # --------------------------------------------------------

    for feature, normal_value in normal_medians.items():

        modified_sample = copy.deepcopy(
            live_sample
        )

        original_value = modified_sample[
            feature
        ]

        # Replace only this one feature
        modified_sample[
            feature
        ] = normal_value

        _, new_probability = predict_sample(
            modified_sample
        )

        change = (
            new_probability
            - base_probability
        )

        results.append({
            "feature": feature,
            "live_value": original_value,
            "normal_median": normal_value,
            "original_probability": base_probability,
            "new_probability": new_probability,
            "change": change
        })

    # --------------------------------------------------------
    # Sort by biggest probability reduction
    # --------------------------------------------------------

    results = sorted(
        results,
        key=lambda x: x["change"]
    )

    for result in results:

        print(
            f"{result['feature']:25}"
            f"{result['live_value']:18.3f}"
            f"{result['normal_median']:18.3f}"
            f"{result['original_probability']:18.4f}"
            f"{result['new_probability']:18.4f}"
            f"{result['change']:18.4f}"
        )

    # --------------------------------------------------------
    # Top features that reduce attack probability most
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("TOP FEATURES REDUCING ATTACK PROBABILITY")
    print("=" * 90)

    for result in results[:10]:

        print(
            f"{result['feature']:20} "
            f"{result['live_value']:.3f} -> "
            f"{result['normal_median']:.3f} | "
            f"probability "
            f"{base_probability:.4f} -> "
            f"{result['new_probability']:.4f} | "
            f"change={result['change']:.4f}"
        )


if __name__ == "__main__":
    main()