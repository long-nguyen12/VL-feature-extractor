import argparse
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def read_info_from_npz(file_path):

    data = np.load(file_path, allow_pickle=True)

    info = data["info"].item()

    image_id = info["image_id"]
    h = info["image_h"]
    w = info["image_w"]
    objects_id = info["objects_id"]
    objects_conf = info["objects_conf"]
    attrs_id = info["attrs_id"]
    attrs_conf = info["attrs_conf"]
    num_boxes = info["num_boxes"]

    bbox = data["bbox"]
    print(bbox.shape)
    features = data["features"]

    return {
        "image_id": image_id,
        "image_h": h,
        "image_w": w,
        "objects_id": objects_id,
        "objects_conf": objects_conf,
        "attrs_id": attrs_id,
        "attrs_conf": attrs_conf,
        "num_boxes": num_boxes,
        "bbox": bbox,
        "features": features,
    }


def read(file_path):
    data = np.load(file_path, allow_pickle=True)

    # Đọc thông tin từ file npz
    info = data["info"].item()

    image_id = info["image_id"]
    h = info["image_h"]
    w = info["image_w"]
    num_boxes = info["num_boxes"]

    boxes = data["bbox"]

    scores = info["objects_conf"]

    classes = info["objects_id"]

    attr_scores = info["attrs_conf"]

    attr_classes = info["attrs_id"]

    return {
        "image_id": image_id,
        "image_h": h,
        "image_w": w,
        "boxes": boxes,
        "scores": scores,
        "classes": classes,
        "attr_scores": attr_scores,
        "attr_classes": attr_classes,
    }


def draw_bbox_on_image(image_path, info, scale=1.0):

    image = cv2.imread(image_path)

    w, h = image.shape[1], image.shape[0]
    # w, h = w*scale, h*scale
    # image = cv2.resize(image,(int(w), int(h)))

    for i in range(info["num_boxes"]):
        x1, y1, x2, y2 = info["bbox"][i]
        # x1, y1, x2, y2 = (x1*w)/scale, (y1*h)/scale, ((x2+1)*w)/scale, ((y2+1)*h)/scale
        label = str(info["objects_id"][i])
        confidence = info["objects_conf"][i]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # text = f"{label}: {confidence:.2f}"
        # cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(
        figsize=(scale * image_rgb.shape[1] / 100, scale * image_rgb.shape[0] / 100)
    )
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title("Image with Bbox")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes on image from npz file"
    )
    parser.add_argument(
        "--npz_folder",
        type=str,
        help="Folder containing npz files",
        default="feature_out",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        help="Folder containing images",
        default="data/Images",
    )
    parser.add_argument(
        "--filename",
        type=str,
        help="Filename of the npz file to process",
        required=True,
    )

    args = parser.parse_args()

    file_path = Path(os.path.join(args.npz_folder, args.filename)).with_suffix(".npz")
    image_path = Path(os.path.join(args.image_folder, args.filename)).with_suffix(
        ".jpg"
    )

    # im = cv2.imread(str)
    # im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # v = Visualizer(im[:, :, :], MetadataCatalog.get("vg"), scale=1.2)
    # pred = read(file_path)
    # v = v.draw_instance_predictions(pred)

    info = read_info_from_npz(file_path)

    draw_bbox_on_image(str(image_path), info, scale=1.2)
