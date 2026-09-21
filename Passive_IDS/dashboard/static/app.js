const API = "http://localhost:8000";


async function loadStatus() {

    try {

        const response =
            await fetch(`${API}/status`);

        const data =
            await response.json();


        const status =
            document.getElementById(
                "statusBadge"
            );


        if (data.capture_running) {

            status.innerHTML =
                "● Monitoring";

            status.className =
                "status running";

        } else {

            status.innerHTML =
                "● Stopped";

            status.className =
                "status stopped";

        }


        document.getElementById(
            "accuracy"
        ).innerText =
            formatPercent(
                data.metrics?.accuracy
            );


        document.getElementById(
            "recall"
        ).innerText =
            formatPercent(
                data.metrics?.recall
            );


        document.getElementById(
            "f1"
        ).innerText =
            formatPercent(
                data.metrics?.f1
            );


        document.getElementById(
            "fpr"
        ).innerText =
            formatPercent(
                data.metrics?.fpr
            );


        if (data.model) {

            document.getElementById(
                "modelSelect"
            ).value =
                data.model;

        }

        if (data.dataset) {

            document.getElementById(
                "datasetSelect"
            ).value =
                data.dataset;

        }


        if (data.protocol) {

            document.getElementById(
                "protocolSelect"
            ).value =
                data.protocol;

        }


    }
    catch (error) {

        console.error(
            "Unable to retrieve status",
            error
        );

    }

}


async function loadPredictions() {

    try {

        const response = await fetch(
            "http://localhost:8000/predictions"
        );

        if (!response.ok) {
            throw new Error(
                `HTTP ${response.status}`
            );
        }

        const predictions =
            await response.json();

        console.log(
            "Predictions:",
            predictions
        );

        const table =
            document.getElementById(
                "trafficTable"
            );

        table.innerHTML = "";


        predictions.forEach(row => {

            const tr =
                document.createElement("tr");

            const predictionClass =
                row.prediction === "NORMAL"
                    ? "normal"
                    : "attack";

            tr.innerHTML = `

                <td>${row.time ?? ""}</td>

                <td>${row.protocol ?? ""}</td>

                <td>${row.service ?? ""}</td>

                <td>${row.state ?? ""}</td>

                <td>${row.packets ?? ""}</td>

                <td>${row.bytes ?? ""}</td>

                <td class="${predictionClass}">
                    ${row.prediction ?? ""}
                </td>

                <td>
                    ${
                        row.probability != null
                            ? (
                                row.probability * 100
                              ).toFixed(1) + "%"
                            : ""
                    }
                </td>

            `;

            table.appendChild(tr);
        });


        document.getElementById(
            "flowCount"
        ).innerText =
            `${predictions.length} flows`;

    }
    catch(error) {

        console.error(
            "Unable to retrieve predictions:",
            error
        );
    }
}


async function startCapture() {

    const dataset =
        document.getElementById(
            "datasetSelect"
        ).value;

    const model =
        document.getElementById(
            "modelSelect"
        ).value;

    const protocol =
        document.getElementById(
            "protocolSelect"
        ).value;


    const config = {
        dataset: dataset,
        model: model,
        protocol: protocol
    };


    console.log(
        "Starting capture with configuration:",
        config
    );


    try {

        const response =
            await fetch(
                `${API}/capture/start`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(config)
                }
            );


        if (!response.ok) {

            const error =
                await response.text();

            console.error(
                "Unable to start capture:",
                error
            );

            return;
        }


        const result =
            await response.json();

        console.log(
            "Capture configuration accepted:",
            result
        );


        loadStatus();

    }
    catch (error) {

        console.error(
            "Unable to start capture:",
            error
        );

    }

}


async function stopCapture() {

    try {

        const response =
            await fetch(
                `${API}/capture/stop`,
                {
                    method: "POST"
                }
            );


        if (!response.ok) {

            const error =
                await response.text();

            console.error(
                "Unable to stop capture:",
                error
            );

            return;
        }


        const result =
            await response.json();

        console.log(
            "Capture stopped:",
            result
        );


        loadStatus();

    }
    catch (error) {

        console.error(
            "Unable to stop capture:",
            error
        );

    }

}





function formatPercent(value) {

    if (value === null ||
        value === undefined)
        return "--";


    return (
        value * 100
    ).toFixed(1) + "%";

}




loadStatus();
loadPredictions();


setInterval(
    loadStatus,
    2000
);


setInterval(
    loadPredictions,
    1000
);