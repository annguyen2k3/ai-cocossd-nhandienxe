const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const sharp = require("sharp");

const app = express();
const upload = multer({ dest: "uploads/" });

let model;
const CLASS_NAMES = [
    "toyota-camry",
    "toyota-fortuner",
    "toyota-inova",
    "vin-vf3",
    "vin-vf7",
    "vin-luxA2.0",
];
const IMAGE_SIZE = 224;

async function loadModel() {
    model = await tf.loadLayersModel("file://car_model/model.json");
    console.log("✅ Mô hình đã tải xong!");
}

loadModel();

app.use(express.static("public"));

app.post("/predict", upload.single("image"), async (req, res) => {
    const filePath = req.file.path;
    const mimeType = req.file.mimetype;

    // ✅ Kiểm tra định dạng ảnh
    const allowedTypes = ["image/jpeg", "image/png", "image/gif", "image/bmp"];
    if (!allowedTypes.includes(mimeType)) {
        fs.unlinkSync(filePath);
        return res.status(400).json({
            prediction: "❌ Định dạng ảnh không hợp lệ!",
            confidence: null,
        });
    }

    try {
        const buffer = fs.readFileSync(filePath);
        const image = await sharp(buffer).resize(224, 224).toBuffer();
        let tensor = tf.node
            .decodeImage(image, 3)
            .toFloat()
            .div(tf.scalar(255))
            .expandDims();

        const prediction = model.predict(tensor);
        const probabilities = prediction.dataSync(); // Lấy danh sách xác suất
        const classIndex = prediction.argMax(1).dataSync()[0];
        const confidence = probabilities[classIndex]; // Xác suất cao nhất

        console.log(
            `Dự đoán: ${
                CLASS_NAMES[classIndex]
            } - Xác suất: ${confidence.toFixed(2)}`
        );

        // Nếu nhỏ hơn 65% thì trả về không xác định
        if (confidence < 0.9) {
            return res.json({
                prediction: "Không xác định",
                confidence: confidence.toFixed(2),
            });
        }

        res.json({
            prediction: CLASS_NAMES[classIndex],
            confidence: confidence.toFixed(2),
        });
    } catch (error) {
        res.status(500).json({
            error: "❌ Lỗi xử lý ảnh: " + error.message,
            confidence: null,
        });
    } finally {
        fs.unlinkSync(filePath);
    }
});

app.listen(3000, () =>
    console.log("🚀 Server đang chạy tại http://localhost:3000")
);
