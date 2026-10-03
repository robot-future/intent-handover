"""Self-contained SVG/HTML result viewer, with no CDN or browser dependencies."""
import html
import json
from pathlib import Path
import numpy as np
from .geometry import corners, gripper_boxes, moved

EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if (i ^ j) in (1, 2, 4)]


def svg_scene(boxes, colors, cloud=None):
    projected = []
    for b in boxes:
        xyz = corners(b)
        projected.append(np.column_stack((xyz[:, 0] + .45 * xyz[:, 1],
                                          -xyz[:, 2] + .25 * xyz[:, 1])))
    cloud_xy = None
    if cloud is not None:
        p = np.asarray(cloud)[::max(1, len(cloud)//2048)]
        cloud_xy = np.column_stack((p[:, 0]+.45*p[:, 1], -p[:, 2]+.25*p[:, 1]))
    if not projected and cloud_xy is None:
        return ""
    all_points = np.concatenate(projected + ([cloud_xy] if cloud_xy is not None else []))
    lo, hi = all_points.min(0), all_points.max(0)
    scale = 330 / max(float((hi - lo).max()), .01)
    center = (lo + hi) / 2
    lines = []
    if cloud_xy is not None:
        for x, y in (cloud_xy-center)*scale+[270, 205]:
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.1" fill="#5dd8c0"/>')
    for xy, color in zip(projected, colors):
        xy = (xy - center) * scale + [270, 205]
        for i, j in EDGES:
            lines.append(f'<line x1="{xy[i,0]:.2f}" y1="{xy[i,1]:.2f}" x2="{xy[j,0]:.2f}" y2="{xy[j,1]:.2f}" stroke="{color}" stroke-width="2.5"/>')
    return '<svg viewBox="0 0 540 410" role="img" aria-label="Object, receiving hand and selected gripper">' + ''.join(lines) + '</svg>'


def write_report(scene, result, output):
    cloud = scene["object"].get("surface_points")
    objects = list(scene["object"]["boxes"]) if cloud is None else []
    boxes = objects + list(scene["receiving_hand"]["boxes"])
    colors = ["#5dd8c0"] * len(objects) + ["#f5b76b"] * len(scene["receiving_hand"]["boxes"])
    if result["selected"]:
        selected = result["selected"]
        boxes += [moved(b, selected["T_object_gripper"]) for b in gripper_boxes(selected["width_m"])]
        colors += ["#84a9ff"] * 3
    rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in
                   [r["id"], f'{r["width_m"]*1000:.1f}' if r["width_m"] is not None else 'n/a',
                    f'{r.get("feasibility_width_m", r["width_m"])*1000:.1f}' if r.get("feasibility_width_m", r["width_m"]) is not None else 'n/a',
                    f'{r["avoidance_cost"]:.4f}',
                    ', '.join(r["rejection_reasons"]) or 'valid']) + '</tr>' for r in result["candidates"])
    title = f'Intent-Handover / {scene["object"]["id"]} / {result["mode"]}'
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(title)}</title><style>
body{{background:#111b2d;color:#e9f0ff;font:16px system-ui;margin:36px auto;max-width:980px;padding:0 24px}}h1{{font-size:32px}}p{{line-height:1.6;color:#bccbe3}}svg{{width:100%;max-height:460px;background:#18263c;border-radius:16px}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #35445b}}code{{color:#5dd8c0}}footer{{margin-top:28px;color:#9aaac3}}
</style><h1>{html.escape(title)}</h1><p>{html.escape(scene['utterance'])}</p>
<p>Human region: <code>{html.escape(scene['intent']['human_region'])}</code> · Selected: <code>{html.escape(result['selected']['id'] if result['selected'] else 'none')}</code></p>
{svg_scene(boxes, colors, cloud)}<p>Green: {'source point cloud' if cloud is not None else 'object proxy'} · Orange: receiving hand proxy · Blue: selected gripper proxy</p>
<table><tr><th>Candidate</th><th>Aperture (mm)</th><th>Feasibility width (mm)</th><th>Avoidance cost ↓</th><th>Constraint result</th></tr>{rows}</table>
<footer>{html.escape(result['provenance'])}. Geometry is in metres. Lower avoidance cost is preferred.
Width policy: {html.escape(result.get('grasp_contract', {}).get('width_policy', 'global_projection'))}.
Feasibility: {html.escape(result.get('grasp_contract', {}).get('feasibility_width_policy', 'opening'))}.
Contact geometry follows the recorded width policy.</footer></html>'''
    Path(output).write_text(document)
