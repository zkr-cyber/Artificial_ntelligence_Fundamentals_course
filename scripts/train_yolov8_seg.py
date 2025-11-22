import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(Path.cwd() / "configs" / "ma_seg_1536x_i.yaml"))
    p.add_argument("--model", default="yolo11n-seg.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--project", default="runs/seg_yolov8")
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device, project=args.project)

if __name__ == "__main__":
    main()