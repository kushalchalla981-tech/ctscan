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
from rich.progress import track
from rich.prompt import Prompt, IntPrompt, Confirm
from rich import box

console = Console()


# ── Utility ──────────────────────────────────────────────────────────────

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


# ── Subcommand: info ──────────────────────────────────────────────────

def cmd_info(args=None):
    from src.phantom import shepp_logan
    from src.projector import build_system, get_sparsity

    size = 32
    A, b, x_true = build_system(size)

    t = Table(box=box.ROUNDED)
    t.add_column('Property', style='cyan')
    t.add_column('Value')

    t.add_row('Phantom', f'Shepp-Logan ({size}x{size})')
    t.add_row('System matrix A', f'{A.shape[0]} measurements × {A.shape[1]} pixels')
    t.add_row('Matrix sparsity', f'{get_sparsity(A):.2%}')
    t.add_row('Non-zeros', f'{A.nnz:,}')
    t.add_row('Memory (A)', f'{A.data.nbytes / 1024**2:.2f} MB')
    t.add_row('Solver', 'LSQR (iterative least squares)')
    t.add_row('Regularization', 'Tikhonov (damp parameter)')
    t.add_row('Cli', 'python main.py reconstruct --help')

    console.print(Panel(t, title='[bold]CT Reconstruction Project[/]', border_style='blue'))
    console.print('\n[yellow]4 phases[/] · [cyan]22 source files[/] · [green]31 tests[/]')


# ── Subcommand: reconstruct ────────────────────────────────────────────

def cmd_reconstruct(args):
    from src.reconstructor import reconstruct

    with console.status('[bold green]Reconstructing...') as status:
        results = reconstruct(size=args.size, use_refinement=args.refine, method=args.method)

    m = results['metrics']
    t = Table(box=box.SIMPLE_HEAVY, title='[bold]Results[/]')
    t.add_column('Metric', style='cyan')
    t.add_column('Value', justify='right')
    for k, v in [('RMSE', m['rmse']), ('PSNR', f'{m["psnr"]:.2f} dB'),
                 ('SSIM', m['ssim']), ('Residual', f'{m["residual"]:.2e}')]:
        t.add_row(k, str(v))
    console.print(t)

    if args.output:
        import matplotlib.pyplot as plt
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        console.print(f'[green]Plot saved:[/] {args.output}')


# ── Subcommand: validate ──────────────────────────────────────────────

def cmd_validate(args):
    from src.validate import validate_forward_model, validate_lu_solver, \
        validate_reconstruction, validate_noise_robustness, run_all

    phases = {
        1: ('Forward Model', validate_forward_model, {}, True),
        2: ('LU Solver', validate_lu_solver, {}, False),
        3: ('Reconstruction', validate_reconstruction, {}, True),
        4: ('Noise Robustness', validate_noise_robustness, {}, True),
    }

    if args.all:
        all_results = run_all(size=args.size)
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
                kwargs['size'] = args.size
            pairs = func(**kwargs)
            t, p, c = _result_table(f'[bold]Phase {pid}: {name}[/]', pairs)
            console.print(t)
            return

    console.print('[red]Specify --phase or --all[/]')


# ── Subcommand: noise ─────────────────────────────────────────────────

def cmd_noise(args):
    from src.noise import noise_robustness_test, add_gaussian_noise
    from src.projector import build_system
    from src.lud_solver import solve_lu
    from src.metrics import compute_metrics, ssim

    noise_levels = [l / 100.0 for l in args.levels]

    if args.regularize:
        console.print('[bold]Noise Robustness[/] (Tikhonov, damp=2.0)', style='cyan')
    else:
        console.print('[bold]Noise Robustness[/] (unregularized)', style='red')

    results = noise_robustness_test(
        size=args.size, noise_levels=noise_levels,
        use_regularization=args.regularize, seed=42
    )

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


# ── Subcommand: interactive ────────────────────────────────────────────

def cmd_interactive(args=None):
    from src.validate import validate_forward_model, validate_lu_solver, \
        validate_reconstruction, validate_noise_robustness, run_all

    console.print(Panel(
        '[cyan]CT Reconstruction Toolkit[/]\n\n'
        'Choose an option below to run a pipeline or validation.',
        title='[bold]Interactive Mode[/]', border_style='green',
    ))

    while True:
        console.print()
        console.print('[bold]Menu:[/]')
        console.print('  [1] Reconstruct (32×32)')
        console.print('  [2] Reconstruct with refinement')
        console.print('  [3] Validate — Forward Model')
        console.print('  [4] Validate — LU Solver')
        console.print('  [5] Validate — Reconstruction')
        console.print('  [6] Validate — Noise Robustness')
        console.print('  [7] Validate — All')
        console.print('  [8] Noise sweep')
        console.print('  [9] Project info')
        console.print('  [0] Exit')

        choice = Prompt.ask('[bold yellow]Select[/]', choices=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

        if choice == '0':
            console.print('[green]Goodbye.[/]')
            break

        elif choice == '1':
            from src.reconstructor import reconstruct
            size = IntPrompt.ask('Phantom size', default=32)
            with console.status('[bold green]Reconstructing...'):
                results = reconstruct(size=size, use_refinement=False)
            m = results['metrics']
            t = Table(box=box.SIMPLE_HEAVY)
            t.add_column('Metric', style='cyan'); t.add_column('Value', justify='right')
            for k, v in [('RMSE', m['rmse']), ('PSNR', f'{m["psnr"]:.2f} dB'),
                         ('SSIM', m['ssim']), ('Residual', f'{m["residual"]:.2e}')]:
                t.add_row(k, str(v))
            console.print(t)

        elif choice == '2':
            from src.reconstructor import reconstruct
            size = IntPrompt.ask('Phantom size', default=32)
            with console.status('[bold green]Reconstructing with refinement...'):
                results = reconstruct(size=size, use_refinement=True)
            m = results['metrics']
            t = Table(box=box.SIMPLE_HEAVY)
            t.add_column('Metric', style='cyan'); t.add_column('Value', justify='right')
            for k, v in [('RMSE', m['rmse']), ('PSNR', f'{m["psnr"]:.2f} dB'),
                         ('SSIM', m['ssim']), ('Residual', f'{m["residual"]:.2e}')]:
                t.add_row(k, str(v))
            console.print(t)

        elif choice == '3':
            size = IntPrompt.ask('Phantom size', default=32)
            with console.status('[bold green]Validating forward model...'):
                pairs, _ = validate_forward_model(size)
            t, p, c = _result_table('[bold]Forward Model[/]', pairs)
            console.print(t)

        elif choice == '4':
            with console.status('[bold green]Validating LU solver...'):
                pairs = validate_lu_solver()
            t, p, c = _result_table('[bold]LU Solver[/]', pairs)
            console.print(t)

        elif choice == '5':
            size = IntPrompt.ask('Phantom size', default=32)
            with console.status('[bold green]Validating reconstruction...'):
                pairs = validate_reconstruction(size)
            t, p, c = _result_table('[bold]Reconstruction[/]', pairs)
            console.print(t)

        elif choice == '6':
            size = IntPrompt.ask('Phantom size', default=32)
            with console.status('[bold green]Validating noise robustness...'):
                pairs = validate_noise_robustness(size)
            t, p, c = _result_table('[bold]Noise Robustness[/]', pairs)
            console.print(t)

        elif choice == '7':
            with console.status('[bold green]Running all validations...'):
                all_results = run_all()
            total_pass = total_count = 0
            for name, pairs in all_results.items():
                t, p, c = _result_table(f'[bold]{name}[/]', pairs)
                console.print(t)
                total_pass += p; total_count += c
            console.print(f'\n[bold]Total:[/] {total_pass}/{total_count} passed')

        elif choice == '8':
            from src.noise import noise_robustness_test
            levels_input = Prompt.ask('Noise levels (%)', default='0 1 5 10')
            noise_levels = [float(x) / 100.0 for x in levels_input.split()]
            reg = Confirm.ask('Use regularization?', default=True)
            with console.status('[bold green]Running noise sweep...'):
                results = noise_robustness_test(noise_levels=noise_levels,
                                                use_regularization=reg, seed=42)
            t = Table(box=box.SIMPLE_HEAVY)
            t.add_column('Noise', justify='right'); t.add_column('RMSE', justify='right')
            t.add_column('PSNR', justify='right'); t.add_column('SSIM', justify='right')
            for nl in noise_levels:
                m = results[nl]
                t.add_row(f'{nl*100:.0f}%', f'{m["rmse"]:.4f}',
                          f'{m["psnr"]:.2f} dB', f'{m["ssim"]:.4f}')
            console.print(t)

        elif choice == '9':
            cmd_info()


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='CT Reconstruction using LU Decomposition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python main.py reconstruct --size 32\n'
            '  python main.py reconstruct --size 32 --refine --output result.png\n'
            '  python main.py validate --all\n'
            '  python main.py validate --phase 2\n'
            '  python main.py noise --levels 0 1 5 10 --regularize --plot noise.png\n'
            '  python main.py interactive\n'
            '  python main.py info\n'
        )
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # reconstruct
    r = sub.add_parser('reconstruct', help='Reconstruct image from CT projections')
    r.add_argument('--size', type=int, default=32, help='Phantom size (default: 32)')
    r.add_argument('--refine', action='store_true', help='Use iterative refinement')
    r.add_argument('--method', choices=['auto', 'sparse', 'dense'], default='auto',
                   help='Solver method (default: auto)')
    r.add_argument('--output', '-o', help='Save plot to file')

    # validate
    v = sub.add_parser('validate', help='Run system validation checks')
    v.add_argument('--phase', type=int, choices=[1, 2, 3, 4],
                   help='Run a single phase validation')
    v.add_argument('--all', action='store_true', help='Run all validations')
    v.add_argument('--size', type=int, default=32, help='Phantom size (default: 32)')

    # noise
    n = sub.add_parser('noise', help='Test reconstruction robustness under noise')
    n.add_argument('--levels', type=float, nargs='+', default=[0, 1, 5, 10, 20],
                   help='Noise levels in percent (e.g. 0 1 5 10)')
    n.add_argument('--regularize', action='store_true', help='Use Tikhonov regularization')
    n.add_argument('--size', type=int, default=32, help='Phantom size (default: 32)')
    n.add_argument('--plot', '-p', help='Save visual comparison plot to file')

    # interactive
    sub.add_parser('interactive', help='Launch interactive menu-driven mode')

    # info
    sub.add_parser('info', help='Show project information')

    args = parser.parse_args()

    dispatch = {
        'reconstruct': cmd_reconstruct,
        'validate': cmd_validate,
        'noise': cmd_noise,
        'interactive': cmd_interactive,
        'info': cmd_info,
    }
    dispatch[args.command](args)


if __name__ == '__main__':
    main()
