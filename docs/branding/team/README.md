# Team photos

Drop portrait photos here, named exactly as referenced from `website/index.qmd`:

- `pramita-bagchi.jpg` — Pramita Bagchi
- `sushovan-majhi.jpg`  — Sushovan Majhi
- `atish-mitra.jpg`     — Atish Mitra
- `ziga-virk.jpg`       — Žiga Virk

## Specs

- Format: JPG (lossy, smaller for web), PNG, or WebP
- Dimensions: square crop, ≥ 400×400 px (rendered at 64 px so 2× retina = 128 px minimum; 400 px is safe for future hero / paper-page reuse)
- Composition: head-and-shoulders, centred, neutral background preferred
- Colour: any (the `.avatar` circle handles the dark-mode framing)
- File size: aim for < 50 KB per portrait — use `cwebp -q 80` or `convert ... -quality 80 -resize 400x400` to compress

The `.avatar` element shows initials (`PB`, `SM`, `AM`, `ŽV`) as a fallback when the image is missing or fails to load. As soon as the file lands at the path above, the photo replaces the initials automatically.

After dropping new photos, re-render the site:

```bash
cd ../../website
quarto render
```

The `post-render.sh` hook copies `branding/team/` into `docs/branding/team/` so the GitHub Pages output sees them.
