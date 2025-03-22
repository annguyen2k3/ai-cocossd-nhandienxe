const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

const DATASET_DIR = "./dataset";
const AUGMENTATIONS_PER_IMAGE = 3; // Số ảnh tạo thêm cho mỗi ảnh gốc

const augmentImage = async (inputPath, outputDir, baseName, index) => {
    const image = sharp(inputPath);

    const metadata = await image.metadata();
    const width = metadata.width;
    const height = metadata.height;

    const operations = [
        image.flop(), // Lật ngang
        image.rotate(10), // Xoay 10 độ
        image.modulate({ brightness: 1.2 }), // Tăng sáng
        image.extract({
            left: 10,
            top: 10,
            width: width - 20,
            height: height - 20,
        }), // Zoom (crop nhẹ)
    ];

    // Chọn random N phép biến đổi để áp dụng
    for (let i = 0; i < AUGMENTATIONS_PER_IMAGE; i++) {
        const op = operations[Math.floor(Math.random() * operations.length)];
        const outputPath = path.join(
            outputDir,
            `${baseName}_aug${index}_${i}.jpg`
        );
        await op.toFile(outputPath);
    }
};

const runAugmentation = async () => {
    const classes = fs.readdirSync(DATASET_DIR);

    for (const className of classes) {
        const classDir = path.join(DATASET_DIR, className);
        const files = fs
            .readdirSync(classDir)
            .filter((f) => /\.(jpg|jpeg|png)$/i.test(f));

        console.log(
            `🔧 Đang tăng cường dữ liệu cho: ${className} (${files.length} ảnh)`
        );

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const filePath = path.join(classDir, file);
            const baseName = path.parse(file).name;

            await augmentImage(filePath, classDir, baseName, i);
        }
    }

    console.log("✅ Hoàn tất tăng cường dữ liệu!");
};

runAugmentation();
