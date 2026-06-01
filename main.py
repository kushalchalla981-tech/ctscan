#!/usr/bin/env python
"""CT Reconstruction using LU Decomposition — CLI entry point."""

import sys
import argparse
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich import box
from rich.text import Text
from rich.syntax import Syntax
from rich.columns import Columns

console = Console()

VERSION = '1.1.0'

BANNER = r"""
[bold cyan]
   ____ _____        _   _                                        _   _
  / ___|_   _|__    | | | | ___ _ __ ___  ___ _ __  _ __ ___  __| | | |_ ___  _ __ ___
 | |     | |/ __|   | |_| |/ _ | '__/ __|/ _ | '__|| '__/ _ \/ _` | | __/ _ \| '_ ` _ \
 | |____ | |\__ \   |  _  |  __| |  \__ |  __| |   | | |  __| (_| | | || (_) | | | | | |
  \____| |_||___/   |_| |_|\___|_|  |___/\___|_|   |_|  \___|\__,_|  \__\___/|_| |_| |_|

[/bold cyan]
"""

# ── Utilities ──────────────────────────────────────────────────────────────

def _banner():
    console.print(BANNER)

def _result_table(title, pairs, caption=None):
    t = Table(title=title, box=box.MINIMAL_HEAVY_HEAD, title_justify='left')
    t.add_column('Check', style='cyan')
    t.add_column('Status', width=10)
    t.add_column('Detail', style='white')
    passed = 0
    for label, ok, detail in pairs:
        status = '[bold green]PASS[/]' if ok else '[bold red]FAIL[/]'
        passed += int(ok)
        t.add_row(label, status, detail)
    t.caption = f'{caption or ""}  [{passed}/{len(pairs)}] passed'
    return t, passed, len(pairs)

def _metrics_table(metrics, title='[bold]Results[/]'):
    t = Table(box=box.SIMPLE_HEAVY, title=title)
    t.add_column('Metric', style='cyan')
    t.add_column('Value', justify='right')
    for k, v in [('RMSE', f'{metrics["rmse"]:.4f}'),
                 ('PSNR', f'{metrics["psnr"]:.2f} dB'),
                 ('SSIM', f'{metrics["ssim"]:.4f}'),
                 ('Residual', f'{metrics["residual"]:.2e}')]:
        t.add_row(k, str(v))
    return t

def _validate_size(size):
    if size < 8 or size > 256:
        console.print('[red]Error: --size must be between 8 and 256.[/]')
        sys.exit(1)
    return size

def _handle_error(e):
    console.print(f'[red]Error:[/] {e}')
    sys.exit(1)

def _pick_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title='Select an image file',
            filetypes=[
                ('Image files', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.dcm'),
                ('DICOM', '*.dcm'),
                ('All files', '*.*'),
            ]
        )
        root.destroy()
        return path if path else None
    except Exception:
        return None


# ── Rich Help ──────────────────────────────────────────────────────────────

class RichHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom formatter that is compatible with rich's syntax panel."""

def _show_help():
    _banner()
    console.print(Panel(
        '[bold]CT Reconstruction using LU Decomposition[/]\n\n'
        'Educational prototype for medical image reconstruction from CT projection data.\n'
        'Uses LU factorization and LSQR to reconstruct images from simulated X-ray projections.\n',
        border_style='blue',
    ))

    cmd_table = Table(box=box.SIMPLE)
    cmd_table.add_column('Command', style='cyan', no_wrap=True)
    cmd_table.add_column('Description')
    cmd_table.add_column('Usage')
    cmd_table.add_row(
        'reconstruct', 'Run reconstruction pipeline',
        'python main.py reconstruct [options]')
    cmd_table.add_row(
        'upload', 'Pick an image file → simulate CT',
        'python main.py upload')
    cmd_table.add_row(
        'validate', 'Run system validation checks',
        'python main.py validate --all')
    cmd_table.add_row(
        'noise', 'Test robustness under noise',
        'python main.py noise --levels 0 1 5 10')
    cmd_table.add_row(
        'interactive', 'Menu-driven interactive mode',
        'python main.py interactive')
    cmd_table.add_row(
        'info', 'Show project information',
        'python main.py info')
    console.print(cmd_table)

    console.print(Panel(
        '[bold yellow]Optional[/]\n'
        '  --version, -V    Show version\n'
        '  --help, -h       Show this help',
        border_style='yellow',
    ))

    console.print(Panel(
        '[bold]Examples[/]\n\n'
        '[dim]# Reconstruct the default Shepp-Logan phantom[/]\n'
        'python main.py reconstruct\n\n'
        '[dim]# Reconstruct from a DICOM scan[/]\n'
        'python main.py reconstruct --input samples/CT-brain.dcm --compare\n\n'
        '[dim]# Upload your own image[/]\n'
        'python main.py upload\n\n'
        '[dim]# Run all validations[/]\n'
        'python main.py validate --all\n\n'
        '[dim]# Noise robustness test with regularization[/]\n'
        'python main.py noise --levels 0 1 5 10 --regularize --plot noise.png\n\n'
        '[dim]# Interative menu[/]\n'
        'python main.py interactive',
        border_style='green',
    ))
    sys.exit(0)


# ── Subcommand: reconstruct ──────────────────────────────────────────────

def cmd_reconstruct(args):
    from src.reconstructor import reconstruct
    try:
        _validate_size(args.size)
        with console.status('[bold green]Reconstructing...', spinner='dots'):
            results = reconstruct(
                size=args.size, use_refinement=args.refine, method=args.method,
                input_image=args.input, compare_path=args.compare,
                save_metrics_path=args.save_metrics,
            )
        console.print()
        console.print(_metrics_table(results['metrics']))

        if args.output:
            import matplotlib.pyplot as plt
            fig = plt.gcf()
            fig.savefig(args.output, dpi=150, bbox_inches='tight')
            console.print(f'[green]Plot saved:[/] {args.output}')

        if args.input:
            samples = ['samples/CT-brain.dcm', 'samples/CT-chest.dcm', 'samples/CT-ankle.dcm']
            if args.input not in samples:
                console.print(f'\n[dim]Tip: Try python main.py upload for a file dialog.[/]')
    except Exception as e:
        _handle_error(e)


# ── Subcommand: upload ──────────────────────────────────────────────────

def cmd_upload(args):
    console.print(Panel(
        '[bold]Upload Image → CT Reconstruction[/]\n\n'
        'Pick an image file. We will:\n'
        '  [1] Resize it to use as a phantom (ground truth)\n'
        '  [2] Simulate X-ray projections from multiple angles\n'
        '  [3] Reconstruct the original from those projections\n'
        '  [4] Measure accuracy (RMSE, PSNR, SSIM)\n\n'
        '[dim]This simulates what a real CT scanner does —'
        ' it takes projection\nmeasurements and computes'
        ' the internal image using math.[/]',
        border_style='green',
    ))

    path = _pick_file() if not args.file else args.file
    if not path:
        console.print('[yellow]No file selected. Enter path manually:[/]')
        path = Prompt.ask('[bold]Image path[/]')
    if not path or not Path(path).exists():
        console.print('[red]File not found. Aborting.[/]')
        sys.exit(1)

    console.print(f'[green]Loading:[/] {path}')
    from src.reconstructor import reconstruct
    try:
        _validate_size(args.size)
        with console.status('[bold green]Reconstructing...', spinner='dots'):
            results = reconstruct(
                size=args.size, use_refinement=args.refine, method=args.method,
                input_image=path, compare_path=args.compare,
                save_metrics_path=args.save_metrics,
            )
    except Exception as e:
        _handle_error(e)

    m = results['metrics']
    console.print()
    console.print(_metrics_table(m))

    console.print(Panel(
        '[bold]Summary[/]\n\n'
        'Your image was used as the "ground truth". We simulated\n'
        'CT X-ray beams passing through it from multiple angles, then\n'
        f'solved a large system of equations to reconstruct it.\n\n'
        f'RMSE={m["rmse"]:.4f}, PSNR={m["psnr"]:.1f}dB, SSIM={m["ssim"]:.4f}\n'
        f'[dim]Lower RMSE / higher PSNR / SSIM → better reconstruction.[/]\n\n'
        f'[dim]Tip: Try --size 48 or --refine for different results.[/]',
        border_style='blue',
    ))


# ── Subcommand: validate ────────────────────────────────────────────────

def cmd_validate(args):
    from src.validate import validate_forward_model, validate_lu_solver, \
        validate_reconstruction, validate_noise_robustness, run_all

    phases = {
        1: ('Forward Model', validate_forward_model, {}, True),
        2: ('LU Solver', validate_lu_solver, {}, False),
        3: ('Reconstruction', validate_reconstruction, {}, True),
        4: ('Noise Robustness', validate_noise_robustness, {}, True),
    }

    try:
        if args.all:
            with console.status('[bold green]Running all validations...', spinner='dots'):
                all_results = run_all(size=_validate_size(args.size))
            total_pass = total_count = 0
            for name, pairs in all_results.items():
                t, p, c = _result_table(f'[bold]{name}[/]', pairs)
                console.print(t)
                total_pass += p
                total_count += c
            console.print(f'\n[bold]Total:[/] {total_pass}/{total_count} passed')
            return

        for pid, (name, func, kwargs, has_size) in phases.items():
            if args.phase == pid:
                if has_size:
                    kwargs['size'] = _validate_size(args.size)
                with console.status(f'[bold green]Validating {name}...', spinner='dots'):
                    pairs = func(**kwargs)
                t, p, c = _result_table(f'[bold]Phase {pid}: {name}[/]', pairs)
                console.print(t)
                return

        console.print('[red]Specify --phase (1-4) or --all[/]')
    except Exception as e:
        _handle_error(e)


# ── Subcommand: noise ───────────────────────────────────────────────────

def cmd_noise(args):
    from src.noise import noise_robustness_test, add_gaussian_noise
    from src.projector import build_system
    from src.lud_solver import solve_lu
    from src.metrics import compute_metrics, ssim

    noise_levels = [l / 100.0 for l in args.levels]
    label = 'Tikhonov (damp=2.0)' if args.regularize else 'unregularized'

    console.print(f'[bold]Noise Robustness —[/] [cyan]{label}[/]')
    console.print(f'[dim]Size: {args.size}×{args.size}[/]\n')

    try:
        _validate_size(args.size)
        with console.status('[bold green]Running noise sweep...', spinner='dots'):
            results = noise_robustness_test(
                size=args.size, noise_levels=noise_levels,
                use_regularization=args.regularize, seed=42
            )
    except Exception as e:
        _handle_error(e)

    t = Table(box=box.SIMPLE_HEAVY)
    t.add_column('Noise', justify='right')
    t.add_column('RMSE', justify='right')
    t.add_column('PSNR', justify='right')
    t.add_column('SSIM', justify='right')
    t.add_column('Residual', justify='right')

    for nl in noise_levels:
        m = results[nl]
        t.add_row(
            f'{nl*100:.0f}%',
            f'{m["rmse"]:.4f}',
            f'{m["psnr"]:.2f} dB',
            f'{m["ssim"]:.4f}',
            f'{m["residual"]:.2e}',
        )
    console.print(t)

    console.print(Panel(
        '[bold]Interpretation[/]\n\n'
        'As noise increases, RMSE goes up and PSNR/SSIM go down.\n'
        f'Regularization helps contain the damage at higher noise levels.\n'
        f'[dim]Compare: python main.py noise --levels 0 1 5 10 (without --regularize)[/]',
        border_style='yellow',
    ))

    if args.plot:
        import matplotlib.pyplot as plt
        import numpy as np
        A, b_clean, x_true = build_system(args.size)
        phantom = x_true.reshape(args.size, args.size)
        display = [nl for nl in noise_levels if nl in (0.0, 0.01, 0.05, 0.20)
                   or nl == noise_levels[-1]]
        fig, axes = plt.subplots(2, len(display), figsize=(4 * len(display), 8))
        for col, nl in enumerate(display):
            np.random.seed(42)
            b_noisy = add_gaussian_noise(b_clean, nl)
            x_rec, _ = solve_lu(A, b_noisy, regularization=4.0 if args.regularize else None)
            recon = x_rec.reshape(args.size, args.size)
            error = np.abs(phantom - recon)
            axes[0, col].imshow(recon, cmap='gray', vmin=0, vmax=1)
            axes[0, col].set_title(f'Noise {nl*100:.0f}%')
            axes[0, col].axis('off')
            axes[1, col].imshow(error, cmap='hot')
            axes[1, col].set_title(f'Max err {error.max():.3f}')
            axes[1, col].axis('off')
        axes[0, 0].set_ylabel('Reconstruction')
        axes[1, 0].set_ylabel('Error')
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150, bbox_inches='tight')
        console.print(f'[green]Plot saved:[/] {args.plot}')


# ── Subcommand: interactive ─────────────────────────────────────────────

def cmd_interactive(args=None):
    from src.validate import validate_forward_model, validate_lu_solver, \
        validate_reconstruction, validate_noise_robustness, run_all

    _banner()
    console.print(Panel(
        '[cyan]Interactive Menu[/]\n\n'
        'Choose an option to run a pipeline or validation.',
        border_style='green',
    ))

    while True:
        console.print()
        options = [
            ('[1]','Reconstruct (Shepp-Logan phantom)'),
            ('[2]','Reconstruct with refinement'),
            ('[3]','Upload image → reconstruct'),
            ('[4]','Validate — Forward Model'),
            ('[5]','Validate — LU Solver'),
            ('[6]','Validate — Reconstruction'),
            ('[7]','Validate — Noise Robustness'),
            ('[8]','Validate — All'),
            ('[9]','Noise sweep'),
            ('[10]','Project info'),
            ('[0]','Exit'),
        ]
        for num, desc in options:
            console.print(f'  [bold]{num}[/]  {desc}')

        choice = Prompt.ask('[bold yellow]Select[/]', choices=[str(i) for i in range(11)])

        if choice == '0':
            console.print('[green]Goodbye.[/]')
            break

        elif choice == '1':
            from src.reconstructor import reconstruct
            size = IntPrompt.ask('Phantom size', default=32)
            _validate_size(size)
            with console.status('[bold green]Reconstructing...', spinner='dots'):
                results = reconstruct(size=size, use_refinement=False)
            console.print(_metrics_table(results['metrics']))

        elif choice == '2':
            from src.reconstructor import reconstruct
            size = IntPrompt.ask('Phantom size', default=32)
            _validate_size(size)
            with console.status('[bold green]Reconstructing with refinement...', spinner='dots'):
                results = reconstruct(size=size, use_refinement=True)
            console.print(_metrics_table(results['metrics']))

        elif choice == '3':
            path = _pick_file()
            if not path:
                console.print('[yellow]No file selected. Enter path:[/]')
                path = Prompt.ask('[bold]Image path[/]')
            if not path or not Path(path).exists():
                console.print('[red]File not found.[/]')
                continue
            from src.reconstructor import reconstruct
            size = IntPrompt.ask('Phantom size', default=32)
            _validate_size(size)
            with console.status('[bold green]Reconstructing from image...', spinner='dots'):
                results = reconstruct(size=size, use_refinement=False, input_image=path)
            console.print(_metrics_table(results['metrics']))
            console.print(Panel(
                '[bold]Summary[/]\n\n'
                'Your image was used as ground truth. We simulated X-ray\n'
                'projections through it, then reconstructed it using math —\n'
                'just like a real CT scanner.',
                border_style='blue',
            ))

        elif choice == '4':
            size = IntPrompt.ask('Phantom size', default=32)
            with console.status('[bold green]Validating forward model...', spinner='dots'):
                pairs, _ = validate_forward_model(_validate_size(size))
            t, p, c = _result_table('[bold]Forward Model[/]', pairs)
            console.print(t)

        elif choice == '5':
            with console.status('[bold green]Validating LU solver...', spinner='dots'):
                pairs = validate_lu_solver()
            t, p, c = _result_table('[bold]LU Solver[/]', pairs)
            console.print(t)

        elif choice == '6':
            size = IntPrompt.ask('Phantom size', default=32)
            with console.status('[bold green]Validating reconstruction...', spinner='dots'):
                pairs = validate_reconstruction(_validate_size(size))
            t, p, c = _result_table('[bold]Reconstruction[/]', pairs)
            console.print(t)

        elif choice == '7':
            size = IntPrompt.ask('Phantom size', default=32)
            with console.status('[bold green]Validating noise robustness...', spinner='dots'):
                pairs = validate_noise_robustness(_validate_size(size))
            t, p, c = _result_table('[bold]Noise Robustness[/]', pairs)
            console.print(t)

        elif choice == '8':
            with console.status('[bold green]Running all validations...', spinner='dots'):
                all_results = run_all()
            total_pass = total_count = 0
            for name, pairs in all_results.items():
                t, p, c = _result_table(f'[bold]{name}[/]', pairs)
                console.print(t)
                total_pass += p; total_count += c
            console.print(f'\n[bold]Total:[/] {total_pass}/{total_count} passed')

        elif choice == '9':
            from src.noise import noise_robustness_test
            levels_input = Prompt.ask('Noise levels (%)', default='0 1 5 10')
            noise_levels = [float(x) / 100.0 for x in levels_input.split()]
            reg = Confirm.ask('Use regularization?', default=True)
            with console.status('[bold green]Running noise sweep...', spinner='dots'):
                results = noise_robustness_test(
                    noise_levels=noise_levels, use_regularization=reg, seed=42)
            t = Table(box=box.SIMPLE_HEAVY)
            t.add_column('Noise', justify='right'); t.add_column('RMSE', justify='right')
            t.add_column('PSNR', justify='right'); t.add_column('SSIM', justify='right')
            for nl in noise_levels:
                m = results[nl]
                t.add_row(f'{nl*100:.0f}%', f'{m["rmse"]:.4f}',
                          f'{m["psnr"]:.2f} dB', f'{m["ssim"]:.4f}')
            console.print(t)

        elif choice == '10':
            cmd_info()


# ── Subcommand: info ────────────────────────────────────────────────────

def cmd_info(args=None):
    from src.phantom import shepp_logan
    from src.projector import build_system, get_sparsity

    size = 32
    A, b, x_true = build_system(size)

    t = Table(box=box.ROUNDED)
    t.add_column('Property', style='cyan')
    t.add_column('Value')

    t.add_row('Phantom', f'Shepp-Logan ({size}×{size})')
    t.add_row('System matrix A', f'{A.shape[0]} measurements × {A.shape[1]} pixels')
    t.add_row('Matrix sparsity', f'{get_sparsity(A):.2%}')
    t.add_row('Non-zeros', f'{A.nnz:,}')
    t.add_row('Memory (A)', f'{A.data.nbytes / 1024**2:.2f} MB')
    t.add_row('Solver', 'LSQR (iterative least squares)')
    t.add_row('Regularization', 'Tikhonov (damp parameter)')
    t.add_row('CLI', 'python main.py reconstruct --help')

    console.print(Panel(t, title='[bold]CT Reconstruction Project[/]', border_style='blue'))
    console.print('\n[yellow]4 phases[/] · [cyan]22 source files[/] · [green]31 tests[/]')


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog='ct-reconstruction',
        description='CT Reconstruction using LU Decomposition',
        add_help=False,
    )
    parser.add_argument('--help', '-h', action='store_true', help='Show this help')
    parser.add_argument('--version', '-V', action='store_true', help='Show version')

    sub = parser.add_subparsers(dest='command')

    # reconstruct
    r = sub.add_parser('reconstruct', help='Reconstruct image from CT projections')
    r.add_argument('--size', type=int, default=32, help='Phantom size in pixels (8-256, default: 32)')
    r.add_argument('--refine', action='store_true', help='Apply iterative refinement')
    r.add_argument('--method', choices=['auto', 'sparse', 'dense'], default='auto',
                   help='Solver method (default: auto)')
    r.add_argument('--input', '-i', default=None,
                   help='Path to DICOM or image (default: Shepp-Logan phantom)')
    r.add_argument('--compare', '-c', default=None,
                   help='Save 4-panel comparison plot (truth, recon, error, sinogram)')
    r.add_argument('--save-metrics', '-m', default=None,
                   help='Export metrics to JSON file')
    r.add_argument('--output', '-o', help='Save final plot to file')

    # validate
    v = sub.add_parser('validate', help='Run system validation checks')
    v.add_argument('--phase', type=int, choices=[1, 2, 3, 4],
                   help='Run a single phase (1-4)')
    v.add_argument('--all', action='store_true', help='Run all validations')
    v.add_argument('--size', type=int, default=32,
                   help='Phantom size in pixels (8-256, default: 32)')

    # noise
    n = sub.add_parser('noise', help='Test reconstruction robustness under noise')
    n.add_argument('--levels', type=float, nargs='+', default=[0, 1, 5, 10, 20],
                   help='Noise levels in percent, e.g. 0 1 5 10')
    n.add_argument('--regularize', action='store_true',
                   help='Enable Tikhonov regularization')
    n.add_argument('--size', type=int, default=32,
                   help='Phantom size in pixels (8-256, default: 32)')
    n.add_argument('--plot', '-p', help='Save visual comparison plot to file')

    # upload
    u = sub.add_parser('upload', help='Pick an image and simulate CT reconstruction')
    u.add_argument('--file', '-f', default=None,
                   help='Path to image (opens dialog if omitted)')
    u.add_argument('--size', type=int, default=32,
                   help='Phantom size in pixels (8-256, default: 32)')
    u.add_argument('--refine', action='store_true', help='Apply iterative refinement')
    u.add_argument('--method', choices=['auto', 'sparse', 'dense'], default='auto',
                   help='Solver method (default: auto)')
    u.add_argument('--compare', '-c', default=None,
                   help='Save 4-panel comparison plot')
    u.add_argument('--save-metrics', '-m', default=None,
                   help='Export metrics to JSON file')

    # interactive
    sub.add_parser('interactive', help='Launch interactive menu-driven mode')

    # info
    sub.add_parser('info', help='Show project information')

    args = parser.parse_args()

    if args.help or args.version:
        if args.version:
            console.print(f'[cyan]ct-reconstruction[/] version [bold]{VERSION}[/]')
            sys.exit(0)
        _show_help()

    if not args.command:
        _show_help()

    dispatch = {
        'reconstruct': cmd_reconstruct,
        'validate': cmd_validate,
        'noise': cmd_noise,
        'upload': cmd_upload,
        'interactive': cmd_interactive,
        'info': cmd_info,
    }
    dispatch[args.command](args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        console.print('\n[yellow]Interrupted.[/]')
        sys.exit(0)
