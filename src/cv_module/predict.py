# src/cv_module/predict.py
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_checkpoint(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_state = ckpt["model_state"]
    # load labels
    base = Path(ckpt_path).parent
    with open(base / "labels.json", "r") as f:
        labels = json.load(f)["idx2label"]
    num_classes = len(labels)
    model = models.resnet50(pretrained=False)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(model_state)
    model.eval()
    return model, labels

def preprocess_image(img_path, size=224):
    img = Image.open(img_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return tf(img).unsqueeze(0), np.array(img)

# Simple Grad-CAM for ResNet last conv layer
def grad_cam(model, input_tensor, target_class=None, layer_name="layer4"):
    model.zero_grad()
    # forward hooks to capture activations
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations['value'] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    # register hooks
    for name, module in model.named_modules():
        if name == layer_name:
            h_f = module.register_forward_hook(forward_hook)
            h_b = module.register_backward_hook(backward_hook)
            break

    outputs = model(input_tensor)
    if target_class is None:
        target_class = outputs.argmax(dim=1).item()
    score = outputs[0, target_class]
    score.backward(retain_graph=True)

    activ = activations['value'][0].cpu().numpy()  # C x H x W
    grads = gradients['value'][0].cpu().numpy()  # C x H x W

    weights = np.mean(grads, axis=(1,2))  # C
    cam = np.zeros(activ.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activ[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
    cam = cam - cam.min()
    if cam.max() != 0:
        cam = cam / cam.max()
    # remove hooks
    h_f.remove(); h_b.remove()
    return cam

def save_cam_on_image(orig_img_np, cam, out_path, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32)/255.0
    img = orig_img_np.astype(np.float32)/255.0
    overlay = (1-alpha)*img + alpha*heatmap
    overlay = np.clip(overlay, 0, 1)
    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.imshow(overlay)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    model, labels = load_checkpoint(args.checkpoint, device=device)
    model.to(device)
    for p in args.inputs:
        inp_tensor, orig_np = preprocess_image(p, size=args.size)
        inp_tensor = inp_tensor.to(device)
        outputs = model(inp_tensor)
        probs = F.softmax(outputs, dim=1).cpu().detach().numpy()[0]
        topk = probs.argsort()[-args.topk:][::-1]
        print(f"Results for {p}:")
        for idx in topk:
            print(f"  {labels[str(idx)]}  {probs[idx]:.4f}")
        # grad-cam
        cam = grad_cam(model, inp_tensor, target_class=topk[0], layer_name=args.layer)
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_file = out / (Path(p).stem + "_cam.png")
        save_cam_on_image(orig_np, cam, str(out_file))
        print("Saved CAM to", out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/disease_model/best.pth")
    parser.add_argument("--inputs", nargs="+", required=True, help="image or folder of images")
    parser.add_argument("--out-dir", default="models/disease_model/cams")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--layer", default="layer4")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    # expand if inputs is a folder
    expanded = []
    for p in args.inputs:
        p = Path(p)
        if p.is_dir():
            for f in p.rglob("*"):
                if f.suffix.lower() in [".jpg",".jpeg",".png"]:
                    expanded.append(str(f))
        else:
            expanded.append(str(p))
    args.inputs = expanded
    main(args)
