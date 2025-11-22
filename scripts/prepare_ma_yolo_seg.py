import argparse
import json
import os
from pathlib import Path

def coco_to_yolo_seg(coco_json_path: Path, images_dir: Path, out_labels_split_dir: Path, class_id_map: dict):
    out_labels_split_dir.mkdir(parents=True, exist_ok=True)
    with coco_json_path.open("r") as f:
        coco = json.load(f)

    img_id_to_info = {img["id"]: img for img in coco["images"]}

    anns = coco.get("annotations", [])
    img_to_lines = {}

    for ann in anns:
        img_id = ann["image_id"]
        img = img_id_to_info.get(img_id)
        if img is None:
            continue
        w, h = img["width"], img["height"]
        cat_id = ann.get("category_id", 0)
        cls = class_id_map.get(cat_id, cat_id)

        seg = ann.get("segmentation")
        if seg is None:
            continue

        if isinstance(seg, dict):
            continue

        for poly in seg:
            if len(poly) < 6:
                continue
            coords = []
            for i in range(0, len(poly), 2):
                x = max(0.0, min(float(poly[i]) / w, 1.0))
                y = max(0.0, min(float(poly[i + 1]) / h, 1.0))
                coords.extend([x, y])
            line = str(int(cls)) + " " + " ".join(f"{v:.6f}" for v in coords)
            img_to_lines.setdefault(img["file_name"], []).append(line)

    for file_name, lines in img_to_lines.items():
        label_name = Path(file_name).with_suffix(".txt").name
        out_path = out_labels_split_dir / label_name
        with out_path.open("w") as f:
            f.write("\n".join(lines))

def make_yaml(path_root: Path, train_split: str, val_split: str, names: list, yaml_out: Path):
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    content = []
    content.append(f"path: {path_root}")
    content.append(f"train: {train_split}")
    content.append(f"val: {val_split}")
    content.append("names:")
    for i, n in enumerate(names):
        content.append(f"  {i}: {n}")
    yaml_out.write_text("\n".join(content))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", default="1536x")
    parser.add_argument("--subset", default="i")
    parser.add_argument("--root", default=str(Path.cwd() / "dataset"))
    parser.add_argument("--names", nargs="+", default=["MA"])
    parser.add_argument("--yaml_out", default=str(Path.cwd() / "configs" / "ma_seg_1536x_i.yaml"))
    args = parser.parse_args()

    path_root = Path(args.root) / args.res / args.subset
    images_train = path_root / "im_train"
    images_val = path_root / "im_val"
    labels_root = path_root / "labels"
    labels_train = labels_root / "im_train"
    labels_val = labels_root / "im_val"

    coco_train = path_root / "label" / "train.json"
    coco_val = path_root / "label" / "val.json"

    class_id_map = {}
    try:
        data = json.loads(coco_train.read_text())
        cats = data.get("categories")
        if cats:
            for i, c in enumerate(cats):
                class_id_map[c["id"]] = i
    except Exception:
        class_id_map = {1: 0}

    coco_to_yolo_seg(coco_train, images_train, labels_train, class_id_map)
    coco_to_yolo_seg(coco_val, images_val, labels_val, class_id_map)

    make_yaml(path_root, "im_train", "im_val", args.names, Path(args.yaml_out))

if __name__ == "__main__":
    def build_combined_all(root: Path):
        res_list = ["128x", "192x", "1536x"]
        subsets = ["i", "ii", "iii"]
        combined = root / "MA_all"
        images_train = combined / "images" / "train"
        images_val = combined / "images" / "val"
        labels_train = combined / "labels" / "train"
        labels_val = combined / "labels" / "val"
        for d in [images_train, images_val, labels_train, labels_val]:
            d.mkdir(parents=True, exist_ok=True)

        for res in res_list:
            for sub in subsets:
                base = root / res / sub
                coco_train = base / "label" / "train.json"
                coco_val = base / "label" / "val.json"
                if not coco_train.exists() or not coco_val.exists():
                    continue
                try:
                    data = json.loads(coco_train.read_text())
                    anns = data.get("annotations", [])
                    cats = data.get("categories", [])
                except Exception:
                    anns, cats = [], []
                if not anns:
                    continue
                class_id_map = {}
                for i, c in enumerate(cats):
                    class_id_map[c.get("id", i + 1)] = i

                coco_to_yolo_seg(coco_train, base / "im_train", base / "labels" / "im_train", class_id_map)
                coco_to_yolo_seg(coco_val, base / "im_val", base / "labels" / "im_val", class_id_map)

                # link with unique names
                tag = f"{res}_{sub}__"
                for img in (base / "im_train").glob("*"):
                    dst = images_train / (tag + img.name)
                    if not dst.exists():
                        dst.symlink_to(img)
                    lbl_src = base / "labels" / "im_train" / (img.with_suffix(".txt").name)
                    if lbl_src.exists():
                        lbl_dst = labels_train / (tag + lbl_src.name)
                        if not lbl_dst.exists():
                            lbl_dst.symlink_to(lbl_src)
                for img in (base / "im_val").glob("*"):
                    dst = images_val / (tag + img.name)
                    if not dst.exists():
                        dst.symlink_to(img)
                    lbl_src = base / "labels" / "im_val" / (img.with_suffix(".txt").name)
                    if lbl_src.exists():
                        lbl_dst = labels_val / (tag + lbl_src.name)
                        if not lbl_dst.exists():
                            lbl_dst.symlink_to(lbl_src)

        yaml_out = Path.cwd() / "configs" / "ma_seg_all.yaml"
        yaml_out.parent.mkdir(parents=True, exist_ok=True)
        yaml_out.write_text("\n".join([
            f"path: {combined}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: MA",
        ]))

    if os.environ.get("MA_ALL", "0") == "1":
        build_combined_all(Path.cwd() / "dataset")
    else:
        main()