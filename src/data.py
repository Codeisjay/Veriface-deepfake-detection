import os, argparse, cv2, math, shutil, json
from pathlib import Path

def extract_frames(input_dir, output_dir, fps=5, label_map=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if label_map is None:
        label_map = {'real': 'real', 'fake': 'fake'}

    video_paths = list(input_dir.rglob("*.mp4")) + list(input_dir.rglob("*.avi")) + list(input_dir.rglob("*.mov"))
    for vp in video_paths:
        # infer class from parent dir name
        parent = vp.parent.name.lower()
        target_class = label_map.get(parent, None)
        if target_class is None:
            print(f"[skip] {vp} (unknown label from parent '{parent}')")
            continue

        out_dir = output_dir / target_class / vp.stem
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(vp))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(int(orig_fps // fps), 1)

        idx = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_interval == 0:
                img_path = out_dir / f"frame_{saved:05d}.jpg"
                cv2.imwrite(str(img_path), frame)
                saved += 1
            idx += 1
        cap.release()
        print(f"Extracted {saved} frames -> {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-frames", action="store_true", help="extract frames from videos")
    ap.add_argument("--input", type=str, default="data/raw_videos")
    ap.add_argument("--output", type=str, default="data/frames")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--label_map", type=str, default=None, help="python dict as string, e.g. {'real':'real','fake':'fake'}")
    args = ap.parse_args()

    label_map = None
    if args.label_map:
        label_map = eval(args.label_map)

    if args.extract_frames:
        extract_frames(args.input, args.output, fps=args.fps, label_map=label_map)
