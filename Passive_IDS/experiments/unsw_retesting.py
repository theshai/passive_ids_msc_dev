import preprocessing.loader as ld
import preprocessing.inspector as insp
from preprocessing.dataset_config_info import DATASET_CONFIG
import preprocessing.cleaner as cl
import preprocessing.preperrer as prep
import preprocessing.analyzer as an

import pandas as pd
import numpy as np


"""
UNSW-NB15 SENSOR FEATURE VALIDATION

Purpose:

1. Validate UNSW rate formula.
2. Validate sload formula.
3. Validate dload formula.
4. Check UDP state semantics.
5. Investigate flows where:
      spkts = 1
      dpkts = 1
      spkts = 1 AND dpkts = 1
6. Compare those records with our live sensor UDP/DNS flow.

NO model training.
NO model saving.
NO artifacts are overwritten.
"""


def main() -> None:

    # ============================================================
    # LOAD UNSW-NB15
    # ============================================================

    dataset_name = "unsw_nb15"

    config = DATASET_CONFIG[dataset_name]

    df, report = an.analyze_dataset(
        file_path=config["file_path"],
        dataset_name=dataset_name,
        label_column=config["label_column"],
        columns=config["columns"]
    )

    an.print_report(report)

    # Prepare clean X and binary y
    X, y = prep.prepare_xy(
        df,
        dataset_name
    )

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)


    # ============================================================
    # CREATE TEMPORARY DATAFRAME WITH LABEL
    # ============================================================

    check_df = X.copy()

    check_df["label"] = y.values


    # ============================================================
    # REAL SENSOR FLOW WE ARE INVESTIGATING
    # ============================================================

    sensor_udp = {

        "proto": "udp",
        "service": "dns",
        "state": "INT",

        "dur": 0.019976,

        "spkts": 1,
        "dpkts": 1,

        "sbytes": 112,
        "dbytes": 277,

        "rate": 100.120144,

        "sload": 44853.824590,
        "dload": 110933.119744,

        "sloss": 0,
        "dloss": 0,

        "sinpkt": 0,
        "dinpkt": 0,

        "sjit": 0,
        "djit": 0,

        "stcpb": 0,
        "dtcpb": 0,

        "tcprtt": 0,
        "synack": 0,
        "ackdat": 0,

        "smean": 112,
        "dmean": 277,

        "response_body_len": 0,
        "is_sm_ips_ports": 0
    }


    # ============================================================
    # 1. RATE FORMULA VALIDATION
    # ============================================================

    rate_test = check_df[

        (check_df["dur"] > 0) &

        (
            check_df["spkts"]
            +
            check_df["dpkts"]
            > 0
        )

    ].copy()


    # Current sensor formula
    rate_test["rate_formula_current"] = (

        (
            rate_test["spkts"]
            +
            rate_test["dpkts"]
        )

        /

        rate_test["dur"]
    )


    # UNSW candidate formula
    rate_test["rate_formula_unsw"] = (

        (
            rate_test["spkts"]
            +
            rate_test["dpkts"]
            - 1
        )

        /

        rate_test["dur"]
    )


    rate_test["rate_error_current"] = abs(

        rate_test["rate"]

        -

        rate_test["rate_formula_current"]
    )


    rate_test["rate_error_unsw"] = abs(

        rate_test["rate"]

        -

        rate_test["rate_formula_unsw"]
    )


    print("\n========================================")
    print("RATE FORMULA TEST")
    print("========================================")

    print(
        "Rows tested:",
        len(rate_test)
    )

    print(
        "Mean error using total_packets / dur:",
        rate_test["rate_error_current"].mean()
    )

    print(
        "Mean error using (total_packets - 1) / dur:",
        rate_test["rate_error_unsw"].mean()
    )


    # ============================================================
    # 2. SLOAD FORMULA VALIDATION
    # ============================================================

    sload_test = check_df[

        (check_df["dur"] > 0) &

        (check_df["spkts"] > 1)

    ].copy()


    sload_test["sload_candidate"] = (

        (
            sload_test["sbytes"]
            * 8
            /
            sload_test["dur"]
        )

        *

        (
            (
                sload_test["spkts"]
                - 1
            )

            /

            sload_test["spkts"]
        )
    )


    sload_test["sload_error"] = abs(

        sload_test["sload"]

        -

        sload_test["sload_candidate"]
    )


    sload_test["sload_percent_error"] = (

        sload_test["sload_error"]

        /

        sload_test["sload"].replace(
            0,
            np.nan
        )

    ) * 100


    print("\n========================================")
    print("SLOAD CANDIDATE FORMULA")
    print("========================================")

    print(
        "Rows tested:",
        len(sload_test)
    )

    print(
        "Mean absolute error:",
        sload_test["sload_error"].mean()
    )

    print(
        "Median absolute error:",
        sload_test["sload_error"].median()
    )

    print(
        "Median percent error:",
        sload_test["sload_percent_error"].median()
    )

    print(
        "95th percentile percent error:",
        sload_test["sload_percent_error"].quantile(0.95)
    )


    # ============================================================
    # 3. DLOAD FORMULA VALIDATION
    # ============================================================

    dload_test = check_df[

        (check_df["dur"] > 0) &

        (check_df["dpkts"] > 1)

    ].copy()


    dload_test["dload_candidate"] = (

        (
            dload_test["dbytes"]
            * 8
            /
            dload_test["dur"]
        )

        *

        (
            (
                dload_test["dpkts"]
                - 1
            )

            /

            dload_test["dpkts"]
        )
    )


    dload_test["dload_error"] = abs(

        dload_test["dload"]

        -

        dload_test["dload_candidate"]
    )


    dload_test["dload_percent_error"] = (

        dload_test["dload_error"]

        /

        dload_test["dload"].replace(
            0,
            np.nan
        )

    ) * 100


    print("\n========================================")
    print("DLOAD CANDIDATE FORMULA")
    print("========================================")

    print(
        "Rows tested:",
        len(dload_test)
    )

    print(
        "Mean absolute error:",
        dload_test["dload_error"].mean()
    )

    print(
        "Median absolute error:",
        dload_test["dload_error"].median()
    )

    print(
        "Median percent error:",
        dload_test["dload_percent_error"].median()
    )

    print(
        "95th percentile percent error:",
        dload_test["dload_percent_error"].quantile(0.95)
    )


    # ============================================================
    # 4. NORMAL UDP STATE DISTRIBUTION
    # ============================================================

    udp_normal = check_df[

        (check_df["label"] == 0) &

        (check_df["proto"] == "udp")

    ]


    print("\n========================================")
    print("NORMAL UDP STATE DISTRIBUTION")
    print("========================================")

    print(
        udp_normal["state"]
        .value_counts()
    )


    # ============================================================
    # 5. NORMAL BIDIRECTIONAL UDP STATE DISTRIBUTION
    # ============================================================

    normal_bidirectional_udp = udp_normal[

        udp_normal["dpkts"] > 0

    ]


    print("\n========================================")
    print("NORMAL BIDIRECTIONAL UDP STATE")
    print("========================================")

    print(
        normal_bidirectional_udp["state"]
        .value_counts()
    )


    # ============================================================
    # 6. CHECK ALL ROWS WHERE SPKTS = 1
    # ============================================================

    spkts_one = check_df[

        check_df["spkts"] == 1

    ].copy()


    print("\n========================================")
    print("ROWS WHERE SPKTS = 1")
    print("========================================")

    print(
        "Count:",
        len(spkts_one)
    )


    print("\nLabels:")

    print(
        spkts_one["label"]
        .value_counts()
    )


    print("\nProtocols:")

    print(
        spkts_one["proto"]
        .value_counts()
        .head(20)
    )


    print("\nStates:")

    print(
        spkts_one["state"]
        .value_counts()
    )


    print("\nFirst 30 rows:")


    print(

        spkts_one[
            [
                "proto",
                "service",
                "state",

                "dur",

                "spkts",
                "dpkts",

                "sbytes",
                "dbytes",

                "rate",

                "sload",
                "dload",

                "sinpkt",
                "dinpkt",

                "smean",
                "dmean",

                "label"
            ]
        ]

        .head(30)

        .to_string(
            index=False
        )
    )


    # ============================================================
    # 7. CHECK ALL ROWS WHERE DPKTS = 1
    # ============================================================

    dpkts_one = check_df[

        check_df["dpkts"] == 1

    ].copy()


    print("\n========================================")
    print("ROWS WHERE DPKTS = 1")
    print("========================================")

    print(
        "Count:",
        len(dpkts_one)
    )


    print("\nLabels:")

    print(
        dpkts_one["label"]
        .value_counts()
    )


    print("\nProtocols:")

    print(
        dpkts_one["proto"]
        .value_counts()
        .head(20)
    )


    print("\nStates:")

    print(
        dpkts_one["state"]
        .value_counts()
    )


    print("\nFirst 30 rows:")


    print(

        dpkts_one[
            [
                "proto",
                "service",
                "state",

                "dur",

                "spkts",
                "dpkts",

                "sbytes",
                "dbytes",

                "rate",

                "sload",
                "dload",

                "sinpkt",
                "dinpkt",

                "smean",
                "dmean",

                "label"
            ]
        ]

        .head(30)

        .to_string(
            index=False
        )
    )


    # ============================================================
    # 8. MOST IMPORTANT:
    # SPKTS = 1 AND DPKTS = 1
    # ============================================================

    one_one = check_df[

        (check_df["spkts"] == 1)

        &

        (check_df["dpkts"] == 1)

    ].copy()


    print("\n========================================")
    print("ROWS WHERE SPKTS = 1 AND DPKTS = 1")
    print("========================================")

    print(
        "Count:",
        len(one_one)
    )


    print("\nLabels:")

    print(
        one_one["label"]
        .value_counts()
    )


    print("\nLabel percentages:")

    if len(one_one) > 0:

        print(
            one_one["label"]
            .value_counts(
                normalize=True
            )
            * 100
        )


    print("\nProtocols:")

    print(
        one_one["proto"]
        .value_counts()
        .head(30)
    )


    print("\nServices:")

    print(
        one_one["service"]
        .value_counts()
        .head(30)
    )


    print("\nStates:")

    print(
        one_one["state"]
        .value_counts()
    )


    print("\nFirst 30 rows:")


    print(

        one_one[
            [
                "proto",
                "service",
                "state",

                "dur",

                "spkts",
                "dpkts",

                "sbytes",
                "dbytes",

                "rate",

                "sload",
                "dload",

                "sinpkt",
                "dinpkt",

                "sjit",
                "djit",

                "smean",
                "dmean",

                "label"
            ]
        ]

        .head(30)

        .to_string(
            index=False
        )
    )


    # ============================================================
    # 9. NORMAL 1/1 FLOWS ONLY
    # ============================================================

    normal_one_one = one_one[

        one_one["label"] == 0

    ].copy()


    print("\n========================================")
    print("NORMAL SPKTS=1 / DPKTS=1")
    print("========================================")

    print(
        "Count:",
        len(normal_one_one)
    )


    print("\nProtocols:")

    print(
        normal_one_one["proto"]
        .value_counts()
        .head(30)
    )


    print("\nServices:")

    print(
        normal_one_one["service"]
        .value_counts()
        .head(30)
    )


    print("\nStates:")

    print(
        normal_one_one["state"]
        .value_counts()
    )


    print("\nStatistics:")


    one_one_features = [

        "dur",

        "sbytes",
        "dbytes",

        "rate",

        "sload",
        "dload",

        "sinpkt",
        "dinpkt",

        "sjit",
        "djit",

        "smean",
        "dmean"
    ]


    if len(normal_one_one) > 0:

        print(

            normal_one_one[
                one_one_features
            ]

            .describe()

            .T[
                [
                    "min",
                    "25%",
                    "50%",
                    "75%",
                    "max"
                ]
            ]
        )


    # ============================================================
    # 10. NORMAL UDP 1/1 FLOWS
    # ============================================================

    normal_udp_one_one = one_one[

        (one_one["label"] == 0)

        &

        (one_one["proto"] == "udp")

    ].copy()


    print("\n========================================")
    print("NORMAL UDP SPKTS=1 / DPKTS=1")
    print("========================================")

    print(
        "Count:",
        len(normal_udp_one_one)
    )


    print("\nServices:")

    print(
        normal_udp_one_one["service"]
        .value_counts()
        .head(30)
    )


    print("\nStates:")

    print(
        normal_udp_one_one["state"]
        .value_counts()
    )


    if len(normal_udp_one_one) > 0:

        print("\nStatistics:")

        print(

            normal_udp_one_one[
                one_one_features
            ]

            .describe()

            .T[
                [
                    "min",
                    "25%",
                    "50%",
                    "75%",
                    "max"
                ]
            ]
        )


        print("\nFirst 30 NORMAL UDP 1/1 records:")

        print(

            normal_udp_one_one[
                [
                    "service",
                    "state",

                    "dur",

                    "spkts",
                    "dpkts",

                    "sbytes",
                    "dbytes",

                    "rate",

                    "sload",
                    "dload",

                    "sinpkt",
                    "dinpkt",

                    "smean",
                    "dmean",

                    "label"
                ]
            ]

            .head(30)

            .to_string(
                index=False
            )
        )


    # ============================================================
    # 11. NORMAL UDP/DNS 1/1
    # THIS IS THE CLOSEST POSSIBLE MATCH TO OUR SENSOR FLOW
    # ============================================================

    normal_udp_dns_one_one = one_one[

        (one_one["label"] == 0)

        &

        (one_one["proto"] == "udp")

        &

        (one_one["service"] == "dns")

    ].copy()


    print("\n========================================")
    print("NORMAL UDP/DNS SPKTS=1 / DPKTS=1")
    print("========================================")

    print(
        "Count:",
        len(normal_udp_dns_one_one)
    )


    if len(normal_udp_dns_one_one) > 0:

        print("\nStates:")

        print(
            normal_udp_dns_one_one[
                "state"
            ].value_counts()
        )


        print("\nRows:")

        print(

            normal_udp_dns_one_one[
                [
                    "state",

                    "dur",

                    "spkts",
                    "dpkts",

                    "sbytes",
                    "dbytes",

                    "rate",

                    "sload",
                    "dload",

                    "sinpkt",
                    "dinpkt",

                    "sjit",
                    "djit",

                    "smean",
                    "dmean",

                    "label"
                ]
            ]

            .head(50)

            .to_string(
                index=False
            )
        )

    else:

        print(
            "NO normal UDP/DNS flows with spkts=1 and dpkts=1."
        )


    # ============================================================
    # 12. CHECK SLOAD WHEN SPKTS = 1
    # ============================================================

    print("\n========================================")
    print("SLOAD VALUES WHERE SPKTS = 1")
    print("========================================")


    if len(spkts_one) > 0:

        print(
            spkts_one["sload"]
            .describe()
        )


        print(
            "\nUnique SLOAD examples:"
        )

        print(

            spkts_one[
                [
                    "proto",
                    "dur",
                    "spkts",
                    "sbytes",
                    "sload",
                    "label"
                ]
            ]

            .drop_duplicates()

            .head(30)

            .to_string(
                index=False
            )
        )


    # ============================================================
    # 13. CHECK DLOAD WHEN DPKTS = 1
    # ============================================================

    print("\n========================================")
    print("DLOAD VALUES WHERE DPKTS = 1")
    print("========================================")


    if len(dpkts_one) > 0:

        print(
            dpkts_one["dload"]
            .describe()
        )


        print(
            "\nUnique DLOAD examples:"
        )

        print(

            dpkts_one[
                [
                    "proto",
                    "dur",
                    "dpkts",
                    "dbytes",
                    "dload",
                    "label"
                ]
            ]

            .drop_duplicates()

            .head(30)

            .to_string(
                index=False
            )
        )


    # ============================================================
    # 14. CALCULATE CORRECTED SENSOR VALUES WE ALREADY KNOW
    # ============================================================

    total_packets = (

        sensor_udp["spkts"]

        +

        sensor_udp["dpkts"]
    )


    corrected_rate = (

        (total_packets - 1)

        /

        sensor_udp["dur"]
    )


    corrected_state = (

        "CON"

        if (

            sensor_udp["spkts"] > 0

            and

            sensor_udp["dpkts"] > 0
        )

        else "INT"
    )


    print("\n========================================")
    print("CURRENT SENSOR FLOW")
    print("========================================")

    print(
        "proto:",
        sensor_udp["proto"]
    )

    print(
        "service:",
        sensor_udp["service"]
    )

    print(
        "state:",
        sensor_udp["state"]
    )

    print(
        "dur:",
        sensor_udp["dur"]
    )

    print(
        "spkts:",
        sensor_udp["spkts"]
    )

    print(
        "dpkts:",
        sensor_udp["dpkts"]
    )

    print(
        "sbytes:",
        sensor_udp["sbytes"]
    )

    print(
        "dbytes:",
        sensor_udp["dbytes"]
    )

    print(
        "rate:",
        sensor_udp["rate"]
    )

    print(
        "sload:",
        sensor_udp["sload"]
    )

    print(
        "dload:",
        sensor_udp["dload"]
    )


    print("\n========================================")
    print("CONFIRMED SENSOR CORRECTIONS SO FAR")
    print("========================================")

    print(
        "Original state:",
        sensor_udp["state"]
    )

    print(
        "Corrected state:",
        corrected_state
    )

    print()

    print(
        "Original rate:",
        sensor_udp["rate"]
    )

    print(
        "Corrected rate:",
        corrected_rate
    )


    # ============================================================
    # END
    # ============================================================

    print("\n========================================")
    print("END OF UNSW SENSOR VALIDATION")
    print("========================================")


if __name__ == "__main__":

    main()