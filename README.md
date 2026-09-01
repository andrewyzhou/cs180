# cs180 portfolio

Project portfolio for UC Berkeley CS180 (Intro to Computer Vision & Computational
Photography), Fall 2026. Served under [andrewzhou.org](https://andrewzhou.org) but
kept as its own repo; `css/style.css` mirrors the design tokens of the
[personal-site](https://github.com/andrewyzhou/personal-site) repo (Funnel Sans,
oklch grayscale theme system).

## Structure

- `index.html` — portfolio landing page
- `css/style.css` — shared stylesheet (keep tokens in sync with personal-site)
- `js/theme.js` — dark/light toggle, dark default
- `projN/` — one folder per project, `index.html` + `media/`
- `incoming/` (gitignored) — drop zone for raw photos before processing
- `specs/` (gitignored) — downloaded course specs for reference

All asset links are relative so the site works locally, on a bare deployment
domain, and proxied under `andrewzhou.org/cs180/`.

## Project 0 — media

| file | content |
|---|---|
| `proj0/media/carolyn-close.jpg` | portrait close up, iPhone main lens (24mm equiv) |
| `proj0/media/carolyn-zoomed.jpg` | portrait stepped back, iPhone tele (77mm equiv) |
| `proj0/media/building-zoomed.jpg` | building from afar, 2x zoom (48mm equiv) |
| `proj0/media/building-close.jpg` | building up close, no zoom (24mm equiv) |
| `proj0/media/dolly/frame-0[1-9].jpg` | dolly zoom stills, 14→72mm equiv (iPhone 14 Pro + Canon M50 II) |
| `proj0/media/dolly-zoom.gif` | boomerang dolly zoom, 1440x1080 @ 3fps |

All images are full resolution, uncropped. HEIC originals were converted to
JPEG at maximum quality (`sips` `formatOptions best`); Canon JPGs are untouched
copies. GPS EXIF was removed losslessly from all iPhone shots before publishing
(`piexif`, GPS IFD only — orientation and the rest kept). Source originals live
in `incoming/` (gitignored).

## Submission checklist (proj0, due Sep 1 11:59pm)

1. Webpage live and photos swapped in, `todo` blocks replaced with writeups
2. URL → class gallery Google Form
3. Print page as PDF **with URL in header** → Gradescope (code G7EVRZ) —
   the print stylesheet forces light theme automatically
