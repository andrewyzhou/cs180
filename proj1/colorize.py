import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import imageio.v3 as iio
import numpy as np
import skimage as sk
import skimage.io as skio
import tyro
from skimage.filters import gaussian

IMAGE_DIR = Path('proj1/images')
OUT_DIR = Path('proj1/out')

IMAGES = sorted(IMAGE_DIR.glob('*.jpg')) + sorted(IMAGE_DIR.glob('*.tif'))

Image = np.ndarray
Shift = tuple[int, int]  # (dy, dx), numpy row/col order
Metric = Callable[[Image, Image], float]


# --- metrics ---

def l2(a: Image, b: Image) -> float:
    """Sum of squared differences."""
    return float(np.sum((a - b) ** 2))


def ncc(a: Image, b: Image) -> float:
    """Negative normalized cross-correlation. Negative so that lower is better like L2 norm."""
    a = a - a.mean()
    b = b - b.mean()
    return float(-np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b)))


METRICS: dict[str, Metric] = {'l2': l2, 'ncc': ncc}


# --- helpers ---

def crop(img: Image, frac: float = 0.1) -> Image:
    """Crop a border of `frac` on all sides of `img` so edges are ignored."""
    h, w = img.shape
    dh, dw = int(h * frac), int(w * frac)
    return img[dh:h - dh, dw:w - dw]


def downsample(img: Image) -> Image:
    """Halve the image by applying a gaussian prefilter (for anti-aliasing) then sampling every other pixel."""
    return gaussian(img, sigma=1)[::2, ::2]


def build_pyramid(img: Image, min_size: int = 200) -> list[Image]:
    """Build pyramid from finest to coarsest, halving until the next level would be smaller than `min_size`."""
    levels = [img]
    while min(levels[-1].shape) // 2 >= min_size:
        levels.append(downsample(levels[-1]))
    return levels


# --- alignment ---

def search(moving: Image, fixed: Image, metric: Metric,
           center: Shift = (0, 0), radius: int = 15) -> Shift:
    """Score every shift within `radius` of `center`; return the best one."""
    fixed = crop(fixed)
    best_shift, best_score = center, np.inf
    for dy in range(center[0] - radius, center[0] + radius + 1):
        for dx in range(center[1] - radius, center[1] + radius + 1):
            score = metric(crop(np.roll(moving, (dy, dx), axis=(0, 1))), fixed)
            if score < best_score:
                best_shift, best_score = (dy, dx), score
    return best_shift


def align_single(moving: Image, fixed: Image, metric: Metric, radius: int = 15) -> Shift:
    """Single-scale alignment with one search at full resolution."""
    return search(moving, fixed, metric, radius=radius)


def align_pyramid(moving: Image, fixed: Image, metric: Metric,
                  radius: int = 15, refine_radius: int = 3) -> Shift:
    """Pyramid search starting at the coarsest level with shifts up to `radius`, then refining
    at further levels with shifts up to `refine_radius`."""
    moving_pyr = build_pyramid(moving)
    fixed_pyr = build_pyramid(fixed)

    shift = search(moving_pyr[-1], fixed_pyr[-1], metric, radius=radius)
    for mov, fix in zip(moving_pyr[-2::-1], fixed_pyr[-2::-1]):
        shift = (2 * shift[0], 2 * shift[1]) # double as shift was calculated for downsampled images
        shift = search(mov, fix, metric, center=shift, radius=refine_radius)
    return shift


# --- driver ---

def colorize(path: Path, metric: Metric, single: bool = False) -> tuple[Image, Shift, Shift]:
    """Split an image into B, G, R, align G and R to B, and return the RGB image and offsets."""
    img = sk.img_as_float(skio.imread(path))

    height = img.shape[0] // 3
    b = img[:height]
    g = img[height:2 * height]
    r = img[2 * height:3 * height]

    align = align_single if single else align_pyramid
    g_shift = align(g, b, metric)
    r_shift = align(r, b, metric)

    ag = np.roll(g, g_shift, axis=(0, 1))
    ar = np.roll(r, r_shift, axis=(0, 1))

    return np.dstack([ar, ag, b]), g_shift, r_shift


def main(metric: Literal['l2', 'ncc'] = 'ncc', single: bool = False,
         images: list[str] = []) -> None:
    """Colorize every image in and print offsets and timing.

    Args:
        metric: scoring function for the alignment search.
        single: single-scale only, for small jpgs.
        images: images to run, default to all.
    """
    mode = 'single' if single else 'pyramid'
    out_dir = OUT_DIR / f'{mode}_{metric}'
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    t_start = time.perf_counter()
    for path in [img for img in IMAGES if not images or img.stem in images]:
        t0 = time.perf_counter()
        im_out, g_shift, r_shift = colorize(path, METRICS[metric], single)
        iio.imwrite(out_dir / f'{path.stem}.jpg', sk.img_as_ubyte(im_out), quality=90)
        lines.append(f'{path.name:20s} G {str(g_shift):10s} R {str(r_shift):10s} {time.perf_counter() - t0:5.1f}s')
        print(lines[-1])
    lines.append(f'total {time.perf_counter() - t_start:.1f}s')
    print(lines[-1])
    (out_dir / 'out.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    tyro.cli(main)
