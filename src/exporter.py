"""Standalone HTML report generation for CT reconstruction outputs."""

import base64
import io
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_CSS = """
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: #0a0a0f;
  color: #e8e8ed;
  line-height: 1.6;
  padding: 40px 24px;
}
.container { max-width: 960px; margin: 0 auto; }
h1 {
  font-size: 2rem;
  font-weight: 700;
  background: linear-gradient(135deg, #f0f0f5, #a5b4fc, #fbbf24);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 8px;
}
h2 {
  font-size: 1.3rem;
  font-weight: 600;
  color: #a5b4fc;
  margin: 28px 0 12px;
}
h3 {
  font-size: 1rem;
  font-weight: 600;
  color: #fbbf24;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 8px;
}
.subtitle { color: #8a8a9a; margin-bottom: 24px; }
.card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 12px;
  padding: 20px 24px;
  margin-bottom: 20px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}
th {
  text-align: left;
  padding: 10px 16px;
  font-weight: 600;
  color: #8a8a9a;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  border-bottom: 1px solid rgba(255,255,255,0.08);
}
td {
  padding: 10px 16px;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  color: #c8c8d0;
}
tr:last-child td { border-bottom: none; }
.pass { color: #34d399; font-weight: 600; }
.fail { color: #f87171; font-weight: 600; }
.good { color: #34d399; }
.bad { color: #f87171; }
.plain { color: #fbbf24; font-weight: 600; }
img { max-width: 100%; border-radius: 8px; margin: 16px 0; }
.summary-pass { color: #34d399; font-size: 1.1rem; }
.summary-fail { color: #f87171; font-size: 1.1rem; }
.footer {
  margin-top: 32px;
  padding-top: 16px;
  border-top: 1px solid rgba(255,255,255,0.06);
  font-size: 0.8rem;
  color: #6a6a7a;
  text-align: center;
}
.tag {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.tag-green { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.2); }
.tag-red { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.2); }
.tag-blue { background: rgba(96,165,250,0.12); color: #60a5fa; border: 1px solid rgba(96,165,250,0.2); }
"""


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _metrics_rows(metrics):
    rows = ''
    for label, key, fmt, good_key in [
        ('RMSE', 'rmse', '{:.4f}', 'rmse'),
        ('PSNR', 'psnr', '{:.2f} dB', 'psnr'),
        ('SSIM', 'ssim', '{:.4f}', 'ssim'),
        ('Residual', 'residual', '{:.2e}', 'residual'),
    ]:
        val = metrics.get(key, '—')
        formatted = fmt.format(val) if isinstance(val, (int, float)) else str(val)
        cls = 'good' if _is_good(metrics, key) else 'bad'
        rows += f'<tr><td>{label}</td><td class="{cls}">{formatted}</td></tr>'
    return rows


def _is_good(metrics, key):
    if key == 'rmse':
        return metrics.get('rmse', 1) < 0.05
    if key == 'psnr':
        return metrics.get('psnr', 0) > 25
    if key == 'ssim':
        return metrics.get('ssim', 0) > 0.95
    if key == 'residual':
        return metrics.get('residual', 1) < 1e-3
    return True


def _validation_rows(pairs):
    rows = ''
    passed = 0
    for label, ok, detail in pairs:
        passed += int(ok)
        icon = '✓' if ok else '✗'
        cls = 'pass' if ok else 'fail'
        rows += f'<tr><td>{icon}</td><td>{label}</td><td class="{cls}">{detail}</td></tr>'
    total = len(pairs)
    return rows, passed, total


def _noise_table_rows(results, noise_levels):
    rows = ''
    for nl in noise_levels:
        m = results.get(nl, results.get(float(nl), {}))
        if not m:
            continue
        pct = f'{nl * 100:.0f}%'
        rmse = f'{m.get("rmse", "—"):.4f}'
        psnr = f'{m.get("psnr", "—"):.2f} dB'
        ssim = f'{m.get("ssim", "—"):.4f}'
        res = f'{m.get("residual", "—"):.2e}'
        rows += f'<tr><td>{pct}</td><td>{rmse}</td><td>{psnr}</td><td>{ssim}</td><td>{res}</td></tr>'
    return rows


def _build_page(title, body, timestamp=None):
    from datetime import datetime
    ts = timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
  <h1>{title}</h1>
  <p class="subtitle">Generated {ts}</p>
  {body}
  <div class="footer">
    CT Reconstruction using LU Decomposition &nbsp;·&nbsp; Python &nbsp;·&nbsp; NumPy/SciPy
  </div>
</div>
</body>
</html>'''


# ── Public API ──────────────────────────────────────────────────────────


def export_reconstruction_html(
    phantom: np.ndarray,
    reconstruction: np.ndarray,
    error_map: np.ndarray,
    metrics: dict,
    solver_info: dict,
    path: str,
    b: np.ndarray = None,
    source_label: str = 'Shepp-Logan Phantom',
):
    phantom = np.asarray(phantom)
    reconstruction = np.asarray(reconstruction)
    error_map = np.asarray(error_map)
    size = phantom.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    im0 = axes[0, 0].imshow(phantom, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(reconstruction, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Reconstruction  RMSE={metrics.get("rmse", 0):.4f}')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(error_map, cmap='hot', vmin=0, vmax=error_map.max() if error_map.max() > 0 else 1)
    axes[1, 0].set_title(f'Absolute Error  Max={error_map.max():.4f}')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    if b is not None:
        detectors = int(size * 1.4)
        sinogram = np.asarray(b).reshape(-1, detectors)
    else:
        sinogram = np.zeros((size, int(size * 1.4)))
    im3 = axes[1, 1].imshow(sinogram, cmap='gray', aspect='auto')
    axes[1, 1].set_title('Sinogram (Projection Data)')
    axes[1, 1].set_xlabel('Detector')
    axes[1, 1].set_ylabel('Angle')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    plt.tight_layout()

    img_b64 = _fig_to_b64(fig)

    method = solver_info.get('method', 'auto')
    factorization = solver_info.get('factorization', '—')
    residual = solver_info.get('residual', 0)
    noise_lvl = metrics.get('noise_level', None)
    reg = metrics.get('regularization', None)

    extras = f'Size {size}×{size}'
    if noise_lvl:
        extras += f' &nbsp;·&nbsp; Noise {noise_lvl*100:.0f}%'
    if reg:
        extras += f' &nbsp;·&nbsp; Tikhonov λ={reg}'
    else:
        extras += f' &nbsp;·&nbsp; No regularization'

    body = f'''
<div class="card">
  <h3>Source</h3>
  <p>{source_label} &nbsp;·&nbsp; {extras}</p>
</div>

<div class="card">
  <h3>Reconstruction Quality</h3>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{_metrics_rows(metrics)}</tbody>
  </table>
</div>

<div class="card">
  <h3>Comparison</h3>
  <p class="subtitle">Ground truth (left) vs reconstruction (right) with error map</p>
  <img src="data:image/png;base64,{img_b64}" alt="Reconstruction comparison">
</div>

<div class="card">
  <h3>Solver Details</h3>
  <table>
    <thead><tr><th>Property</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>Method</td><td>{method}</td></tr>
      <tr><td>Factorization</td><td>{factorization}</td></tr>
      <tr><td>Residual</td><td>{residual:.2e}</td></tr>
    </tbody>
  </table>
</div>
'''
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_build_page('CT Reconstruction Report', body), encoding='utf-8')
    return str(p)


def export_noise_html(
    results: dict,
    noise_levels: list,
    path: str,
    regularized: bool = False,
):
    rows = _noise_table_rows(results, noise_levels)
    label = 'with Tikhonov Regularization' if regularized else 'unregularized'

    body = f'''
<div class="card">
  <h3>Test Configuration</h3>
  <p><span class="tag tag-blue">{label}</span></p>
</div>

<div class="card">
  <h3>Noise Robustness Results</h3>
  <table>
    <thead><tr><th>Noise</th><th>RMSE</th><th>PSNR</th><th>SSIM</th><th>Residual</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>

<div class="card">
  <h3>Interpretation</h3>
  <p>As noise increases, RMSE rises and PSNR/SSIM degrade. Regularization helps contain the damage at higher noise levels by adding a penalty term that prevents noise amplification.</p>
  <p style="margin-top:8px;color:#8a8a9a;">Lower RMSE &nbsp;·&nbsp; Higher PSNR &nbsp;·&nbsp; Higher SSIM → better reconstruction</p>
</div>
'''
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_build_page('Noise Robustness Report', body), encoding='utf-8')
    return str(p)


def export_volume_html(
    volume_true: np.ndarray,
    volume_rec: np.ndarray,
    volume_error: np.ndarray,
    metrics: dict,
    path: str,
):
    """Export 3D volume reconstruction results to standalone HTML."""
    depth, size = volume_true.shape[0], volume_true.shape[1]
    slice_indices = [0]
    for frac in [0.25, 0.5, 0.75]:
        slice_indices.append(int(depth * frac))
    if depth > 1:
        slice_indices.append(depth - 1)
    slice_indices = sorted(set(slice_indices))

    n_slices = len(slice_indices)
    fig, axes = plt.subplots(3, n_slices, figsize=(3 * n_slices, 7))
    for col, z in enumerate(slice_indices):
        axes[0, col].imshow(volume_true[z], cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(f'Truth Z={z}')
        axes[0, col].axis('off')
        axes[1, col].imshow(volume_rec[z], cmap='gray', vmin=0, vmax=1)
        axes[1, col].set_title(f'Recon Z={z}')
        axes[1, col].axis('off')
        axes[2, col].imshow(volume_error[z], cmap='hot')
        axes[2, col].set_title(f'Error Z={z}')
        axes[2, col].axis('off')
    for row, label in enumerate(['Ground Truth', 'Reconstruction', 'Absolute Error']):
        axes[row, 0].set_ylabel(label, fontsize=10)
    plt.tight_layout()
    img_b64 = _fig_to_b64(fig)

    method = metrics.get('method', 'auto').upper()
    avg_rmse = metrics.get('rmse', 0)
    avg_psnr = metrics.get('psnr', 0)
    avg_ssim = metrics.get('ssim', 0)

    body = f'''
<div class="card">
  <h3>3D Volume Configuration</h3>
  <table>
    <thead><tr><th>Property</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>Depth</td><td>{depth} slices</td></tr>
      <tr><td>Size</td><td>{size}×{size}</td></tr>
      <tr><td>Method</td><td>{method}</td></tr>
      <tr><td>Avg RMSE</td><td class="{"good" if avg_rmse < 0.05 else "bad"}">{avg_rmse:.4f}</td></tr>
      <tr><td>Avg PSNR</td><td class="{"good" if avg_psnr > 25 else "bad"}">{avg_psnr:.2f} dB</td></tr>
      <tr><td>Avg SSIM</td><td class="{"good" if avg_ssim > 0.95 else "bad"}">{avg_ssim:.4f}</td></tr>
    </tbody>
  </table>
</div>

<div class="card">
  <h3>Selected Slices (Truth / Reconstruction / Error)</h3>
  <p class="subtitle">Showing {n_slices} representative slices through the volume</p>
  <img src="data:image/png;base64,{img_b64}" alt="3D volume slices">
</div>

<div class="card">
  <h3>Interpretation</h3>
  <p>Each slice was reconstructed independently. The volume shows how well the
  reconstruction preserves anatomical features across the entire depth.</p>
  <p style="margin-top:8px;color:#8a8a9a;">Lower RMSE &nbsp;·&nbsp; Higher PSNR &nbsp;·&nbsp; Higher SSIM → better reconstruction</p>
</div>
'''
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_build_page('3D Volume Reconstruction Report', body), encoding='utf-8')
    return str(p)


def export_validation_html(
    results: dict,
    path: str,
):
    all_rows = ''
    total_pass = 0
    total_count = 0

    for phase_name, pairs in results.items():
        rows, p, c = _validation_rows(pairs)
        total_pass += p
        total_count += c
        all_rows += f'''
<div class="card">
  <h3>{phase_name}</h3>
  <table>
    <thead><tr><th style="width:32px"></th><th>Check</th><th>Detail</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <p style="margin-top:8px;color:#8a8a9a;">{p}/{c} passed</p>
</div>'''

    summary_cls = 'summary-pass' if total_pass == total_count else 'summary-fail'
    summary_text = f'{total_pass}/{total_count} checks passed' if total_pass == total_count else f'{total_pass}/{total_count} checks passed ({total_count - total_pass} failed)'

    body = f'''
<div class="card" style="text-align:center;padding:24px;">
  <p class="{summary_cls}">{summary_text}</p>
</div>
{all_rows}
'''
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_build_page('Validation Report', body), encoding='utf-8')
    return str(p)
