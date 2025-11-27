import argparse, json, torch, torch.nn.functional as F, cv2, pandas as pd
from pathlib import Path
from model import build_model

def load_ckpt(ckpt_path):
    obj = torch.load(ckpt_path, map_location="cpu")
    return obj["model"], obj.get("classes", ["fake","real"]), obj.get("cfg", None)

def score_frame(model, frame, img_size, device):
    x = cv2.resize(frame, (img_size, img_size))
    x = torch.from_numpy(x).permute(2,0,1).float()/255.0
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = F.softmax(logits, dim=1)[0,1].item()  # prob(fake)
    return prob

def main(video_path, ckpt, fps, out_csv):
    state_dict, classes, cfg = load_ckpt(ckpt)
    img_size = cfg["data"]["img_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,
        dropout=cfg["model"]["dropout"]
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(int(orig_fps // fps), 1)

    rows = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            prob_fake = score_frame(model, frame, img_size, device)
            rows.append({"frame_index": idx, "prob_fake": prob_fake})
        idx += 1
    cap.release()

    df = pd.DataFrame(rows)
    if not df.empty:
        overall_conf = df["prob_fake"].mean()
    else:
        overall_conf = 0.5

    df.to_csv(out_csv, index=False)
    print(f"Saved per-frame scores -> {out_csv}")
    print(f"Overall confidence (mean prob_fake): {overall_conf:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True, help="path to best.ckpt from training")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--out_csv", type=str, default="outputs/per_frame_scores.csv")
    args = ap.parse_args()
    main(args.video, args.weights, args.fps, args.out_csv)
