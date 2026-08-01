/**
 * Pre-renders each series' 3x3 mosaic cover into a single image.
 * Reduces the gallery overview page from ~45 image requests to just 5.
 *
 * Runs automatically via npm prebuild (package.json) before astro build.
 * Output: public/covers/{series-slug}.jpg
 */
import sharp from 'sharp';
import { mkdir, readdir } from 'node:fs/promises';
import path from 'node:path';

const IMAGES_DIR = path.resolve('src/assets/images');
const OUT_DIR = path.resolve('public/covers');

// Cell and canvas dimensions
const CELL_SIZE = 320;
const GAP = 2;
const GRID_SIZE = CELL_SIZE * 3 + GAP * 2;
const BG = { r: 250, g: 250, b: 248 }; // matches --color-bg #FAFAF8

const seriesSlugs = ['if-not-human', 'broken-is-growing', 'dreams', 'city', 'sketches'];

await mkdir(OUT_DIR, { recursive: true });

for (const slug of seriesSlugs) {
  const seriesDir = path.join(IMAGES_DIR, slug);
  let files = [];
  try {
    files = (await readdir(seriesDir))
      .filter((f) => /\.jpe?g$/i.test(f))
      .sort()
      .slice(0, 9);
  } catch {
    console.warn(`Skipping ${slug}: directory not found`);
    continue;
  }

  // Resize each source to a square cell (center crop, no creative framing).
  const layers = [];
  for (let i = 0; i < 9; i++) {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const left = col * (CELL_SIZE + GAP);
    const top = row * (CELL_SIZE + GAP);

    if (i < files.length) {
      const resized = await sharp(path.join(seriesDir, files[i]))
        .resize(CELL_SIZE, CELL_SIZE, { fit: 'cover', position: 'centre' })
        .toBuffer();
      layers.push({ input: resized, left, top });
    } else {
      // Empty cell for series with fewer than 9 works
      const empty = await sharp({
        create: { width: CELL_SIZE, height: CELL_SIZE, channels: 3, background: BG },
      }).jpeg().toBuffer();
      layers.push({ input: empty, left, top });
    }
  }

  const outFile = path.join(OUT_DIR, `${slug}.jpg`);
  await sharp({
    create: { width: GRID_SIZE, height: GRID_SIZE, channels: 3, background: BG },
  })
    .composite(layers)
    .jpeg({ quality: 82, mozjpeg: true })
    .toFile(outFile);

  console.log(`Cover generated: ${slug}.jpg`);
}

console.log('All series covers ready.');
