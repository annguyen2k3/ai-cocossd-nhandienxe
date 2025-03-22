const fs = require("fs");
const path = require("path");

const DATASET_PATH = "./dataset";

const countImages = () => {
    const classes = fs.readdirSync(DATASET_PATH);
    console.log("=== Số lượng ảnh mỗi lớp: ===");

    classes.forEach((className) => {
        const classPath = path.join(DATASET_PATH, className);
        const files = fs.readdirSync(classPath);
        const imageFiles = files.filter((f) => /\.(jpg|jpeg|png)$/i.test(f));

        console.log(`- ${className}: ${imageFiles.length} ảnh`);
    });
};

countImages();
