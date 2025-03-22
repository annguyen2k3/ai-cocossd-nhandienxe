const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

const CLASS_NAMES = [
    "toyota-camry",
    "toyota-fortuner",
    "toyota-inova",
    "vin-vf3",
    "vin-vf7",
    "vin-luxA2.0",
];
const IMAGE_SIZE = 224;

async function loadImagesFromFolder(folderPath, labelIndex) {
    const files = fs.readdirSync(folderPath);
    let images = [];
    let labels = [];

    for (const file of files) {
        const filePath = path.join(folderPath, file);
        const buffer = fs.readFileSync(filePath);

        const image = await sharp(buffer)
            .resize(IMAGE_SIZE, IMAGE_SIZE)
            .toBuffer();
        let tensor = tf.node
            .decodeImage(image, 3)
            .toFloat()
            .div(tf.scalar(255));

        images.push(tensor);
        labels.push(labelIndex);
    }

    return { images, labels };
}

async function loadDataset() {
    let images = [];
    let labels = [];

    for (let i = 0; i < CLASS_NAMES.length; i++) {
        const folderPath = `dataset/${CLASS_NAMES[i]}`;
        const { images: imgs, labels: lbls } = await loadImagesFromFolder(
            folderPath,
            i
        );
        images.push(...imgs);
        labels.push(...lbls);
    }

    return {
        images: tf.stack(images),
        labels: tf.tensor1d(labels, "float32"),
    };
}

function createModel() {
    const model = tf.sequential();
    model.add(
        tf.layers.conv2d({
            filters: 16,
            kernelSize: 3,
            activation: "relu",
            inputShape: [IMAGE_SIZE, IMAGE_SIZE, 3],
        })
    );
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(
        tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: "relu" })
    );
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: "relu" }));
    model.add(
        tf.layers.dense({ units: CLASS_NAMES.length, activation: "softmax" })
    );

    model.compile({
        optimizer: "adam",
        loss: "sparseCategoricalCrossentropy",
        metrics: ["accuracy"],
    });
    return model;
}

async function trainModel() {
    const { images, labels } = await loadDataset();
    const model = createModel();

    await model.fit(images, labels, {
        epochs: 10,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(
                    `📈 Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(
                        4
                    )}, accuracy = ${(logs.acc * 100).toFixed(2)}%`
                );

                if (epoch === 9) {
                    const modelInfo = {
                        totalImages: images.shape[0],
                        epochs: 10,
                        classNames: CLASS_NAMES,
                        imagesPerClass: {},
                        accuracy: (logs.acc * 100).toFixed(2),
                    };

                    // Đếm số ảnh mỗi lớp
                    CLASS_NAMES.forEach((className, index) => {
                        modelInfo.imagesPerClass[className] = labels
                            .arraySync()
                            .filter((l) => l === index).length;
                    });

                    fs.writeFileSync(
                        "public/model-info.json",
                        JSON.stringify(modelInfo, null, 2)
                    );
                    console.log(
                        "✅ Đã lưu thông tin mô hình vào public/model-info.json"
                    );
                }
            },
        },
    });

    await model.save("file://car_model");
    console.log("✅ Mô hình đã được lưu vào thư mục 'car_model'!");
}

trainModel();
