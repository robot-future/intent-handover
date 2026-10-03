"""Optional original-weight Text2HOI coarse inference, without renderer/ROS/MANO.

Uses the original architecture and 1,000-step DDPM. Output retains the original
99D hand representation (translation + 16 six-dimensional rotations). The
prediction bridge handles conversion to MANO geometry.
"""
import numpy as np
from .artifacts import export_run, write_json
from .weights import sha256, verify_weights


def prepare_points(raw):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) < 4 or not np.isfinite(raw).all():
        raise ValueError("Point cloud must contain at least four finite Nx3 points in metres")
    # Deterministic farthest-point sampling; repeated indices allowed for small clouds.
    selected = np.empty(1024, dtype=int)
    distance = np.full(len(raw), np.inf)
    farthest = 0
    for i in range(min(1024, len(raw))):
        selected[i] = farthest
        distance = np.minimum(distance, np.sum((raw - raw[farthest])**2, axis=1))
        farthest = int(np.argmax(distance))
    if len(raw) < 1024:
        selected[len(raw):] = np.resize(selected[:len(raw)], 1024-len(raw))
    sampled = raw[selected]
    center = sampled.mean(axis=0)
    scale = float(np.linalg.norm(sampled-center, axis=1).max())
    if scale <= 1e-8:
        raise ValueError("Point cloud has zero spatial extent")
    return sampled, (sampled-center)/scale, center, scale


def predict(args):
    with export_run(args.output, ["metadata.json"]):
        return _predict(args)


def _predict(args):
    if args.frames < 1 or args.frames > 150:
        raise ValueError("Frames must be between 1 and 150")
    if args.hand not in ("left", "right") or not isinstance(args.prompt, str) or not args.prompt.strip():
        raise ValueError("Prediction requires hand=left/right and a nonempty prompt")
    verified = verify_weights(args.checkpoints)
    paths = {name: args.checkpoints / f"{name}.pth" for name in verified}
    sampled, normalized, center, scale = prepare_points(np.load(args.point_cloud, allow_pickle=False))
    import torch
    import clip
    from ._vendor.text2hoi.texthom import TextHOM
    from ._vendor.text2hoi.pointnet import PointNetfeat
    from ._vendor.text2hoi.cvae import CTCVAE
    from ._vendor.text2hoi.diffusion import Diffusion
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; pass --device cpu (slower)")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = TextHOM().to(device).eval()
    pointnet = PointNetfeat(global_feat=False).to(device).eval()
    contact = CTCVAE().to(device).eval()
    for name, network in [("texthom", model), ("pointfeat", pointnet), ("contact_estimator", contact)]:
        # Load tensor-only checkpoints; never silently use random weights.
        checkpoint = torch.load(paths[name], map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint)
        network.load_state_dict(state, strict=True)
    diffusion = Diffusion(None, None, T=1000).to(device).eval()
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    with torch.no_grad():
        # Match upstream's 20-token prompt limit plus start/end tokens.
        try:
            tokens = clip.tokenize([args.prompt], context_length=22, truncate=False).to(device)
        except RuntimeError as exc:
            raise ValueError("Prompt exceeds Text2HOI's 20 CLIP-token limit; shorten it (tokens are not words)") from exc
        tokens = torch.cat([tokens, torch.zeros((1,55), dtype=tokens.dtype, device=device)], dim=1)
        text = clip_model.encode_text(tokens).float()
        feature = pointnet(torch.from_numpy(normalized).unsqueeze(0).to(device))
        scales = torch.full((1,1024,1), scale, device=device)
        condition = torch.cat([scales, feature, text[:,None].expand(-1,1024,-1)], dim=2)
        contact_map = contact.decode(condition)[..., 0]
        obj_feature = torch.cat([feature[:,0,:1024], (contact_map>.5).float(),
                                 torch.tensor([[scale]], device=device),
                                 torch.from_numpy(center[None]).to(device)], dim=1)
        active = torch.ones((1,args.frames), dtype=torch.bool, device=device)
        inactive = torch.zeros_like(active)
        left, right, obj = diffusion.sampling(model, obj_feature, text, args.frames, 99, 9,
                                             active if args.hand == "left" else inactive,
                                             active if args.hand == "right" else inactive,
                                             active, device)
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = dict(left_hand=left.cpu().numpy(), right_hand=right.cpu().numpy(),
                  object_pose=obj.cpu().numpy(), sampled_object_points=sampled,
                  contact_probability=contact_map.cpu().numpy())
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Neural prediction contains nonfinite values")
    np.savez_compressed(args.output / "prediction.npz", **arrays)
    metadata = {"status": "succeeded", "backend": "original Text2HOI coarse DDPM", "refiner": False, "dataset": "h2o",
                "hand": args.hand, "prompt": args.prompt, "seed": args.seed,
                "frames": args.frames, "device": str(device),
                "representation": "translation(3) + 16 rotations(6); object translation(3) + rotation(6)",
                "checkpoints_sha256": {name: record["sha256"] for name, record in verified.items()},
                "prediction_sha256": sha256(args.output / "prediction.npz"),
                "input_point_cloud_sha256": sha256(args.point_cloud)}
    write_json(args.output / "metadata.json", metadata)
    print(f"Text2HOI prediction: {(args.output / 'prediction.npz').resolve()}")
