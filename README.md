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

## Project 0 — media slots

| file | content |
|---|---|
| `proj0/media/selfie-close.jpg` | portrait from close up (distorted) |
| `proj0/media/selfie-zoomed.jpg` | portrait from several feet back, zoomed in |
| `proj0/media/arch-zoomed.jpg` | building from across a field, zoomed in |
| `proj0/media/arch-close.jpg` | building up close, no zoom |
| `proj0/media/dolly-zoom.gif` | animated dolly zoom (4–8+ stills) |

Current files are labeled placeholders. Drop raw photos (HEIC fine) in
`incoming/` and they get converted/resized/renamed into the slots above; dolly
zoom stills get assembled into the GIF with ffmpeg.

## Submission checklist (proj0, due Sep 1 11:59pm)

1. Webpage live and photos swapped in, `todo` blocks replaced with writeups
2. URL → class gallery Google Form
3. Print page as PDF **with URL in header** → Gradescope (code G7EVRZ) —
   the print stylesheet forces light theme automatically
