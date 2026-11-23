
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import torch


def set_plotly_template(cfg):
    font_size = cfg.plotting.font.size
    tick_size = int(font_size * 0.75)
    
    pio.templates["my_style"] = go.layout.Template(
        data=dict(
            scatter=[go.Scatter(
                marker=dict(size=7),
                error_y=dict(thickness=2, width=5),
            )],
        ),
        layout=go.Layout(
            font=dict(family=cfg.plotting.font.family, size=font_size, color="black"),
            plot_bgcolor="white",
            showlegend=False,
            legend=dict(
                font=dict(size=font_size),
                title_font=dict(size=font_size),
            ),
            xaxis=dict(
                gridcolor="rgba(0,0,0,0.03)",
                title_font=dict(size=font_size),
                tickfont=dict(size=tick_size),
                zeroline=False,
                exponentformat='power',
            ),
            yaxis=dict(
                gridcolor="rgba(0,0,0,0.03)",
                title_font=dict(size=font_size),
                tickfont=dict(size=tick_size),
                zeroline=False,
                exponentformat='power',
            ),
        )
    )
    pio.templates.default = "my_style"

# def save_figure(fig, figures_dir, rootname, scale=2, file_exts=["pdf", "png"]):
def save_figure(fig, figures_dir, rootname, scale=2, file_exts=["pdf"]):
    figures_dir.mkdir(parents=True, exist_ok=True)
    for file_ext in file_exts:
        filename = f"{rootname}.{file_ext}"
        fig.write_image(figures_dir / filename, scale=scale)


def _merge_segments(segs):
    """Merge sorted adjacent segments like [(0,1),(1,2),(3,4)] -> [(0,2),(3,4)]."""
    segs.sort()
    merged = [segs[0]]
    for s, e in segs[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def mask_boundary_shapes(mask_arr, color="red", width=2):
    """Return Plotly shape dicts tracing the perimeter of the True region(s).
    
    Collinear boundary edges are merged into single lines for performance.
    """
    H, W = mask_arr.shape
    h_lines = {}  # y_coord -> list of (x_start, x_end)
    v_lines = {}  # x_coord -> list of (y_start, y_end)

    for i in range(H):
        for j in range(W):
            if not mask_arr[i, j]:
                continue
            # top edge
            if i == 0 or not mask_arr[i - 1, j]:
                h_lines.setdefault(i - 0.5, []).append((j - 0.5, j + 0.5))
            # bottom edge
            if i == H - 1 or not mask_arr[i + 1, j]:
                h_lines.setdefault(i + 0.5, []).append((j - 0.5, j + 0.5))
            # left edge
            if j == 0 or not mask_arr[i, j - 1]:
                v_lines.setdefault(j - 0.5, []).append((i - 0.5, i + 0.5))
            # right edge
            if j == W - 1 or not mask_arr[i, j + 1]:
                v_lines.setdefault(j + 0.5, []).append((i - 0.5, i + 0.5))

    shapes = []
    line_style = dict(color=color, width=width)

    for y, segs in h_lines.items():
        for start, end in _merge_segments(segs):
            shapes.append(dict(
                type="line", x0=start, x1=end, y0=y, y1=y,
                line=line_style,
            ))

    for x, segs in v_lines.items():
        for start, end in _merge_segments(segs):
            shapes.append(dict(
                type="line", x0=x, x1=x, y0=start, y1=end,
                line=line_style,
            ))

    return shapes


def denormalize_cifar10(img):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)