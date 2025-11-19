const express = require("express");
const ort = require("onnxruntime-node");

const app = express();
app.use(express.json());

let session;

// Load ONNX model
async function loadModel() {
    try {
        session = await ort.InferenceSession.create("./model/kmeans.onnx");
        console.log("KMeans ONNX model loaded");
    } catch (err) {
        console.error("Error loading model:", err);
    }
}

loadModel();

// Endpoint predict
app.post("/predict", async (req, res) => {
    try {
        const { TransactionAmount, AccountBalance } = req.body;

        // Input harus float64 (double)
        const input = new ort.Tensor(
            "float64",
            new Float64Array([
                TransactionAmount,
                AccountBalance
            ]),
            [1, 2] // shape: batch=1, features=2
        );

        // Nama input model dari ONNX
        const feeds = {};
        feeds[session.inputNames[0]] = input;

        const results = await session.run(feeds);

        // Ambil output cluster
        const outputName = session.outputNames[0];
        const cluster = results[outputName].data[0];

        res.json({ cluster: Number(cluster) });

    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});


app.listen(3000, () => console.log("Server running on port 3000"));

module.exports = app;
