# Slide media generators

Presentation-only code lives here, separate from the training and evaluation
launchers in `part2action/scripts/`. These scripts read recorded demonstrations,
completed evaluation outputs, and slide Markdown. They do not submit jobs or
run new policy rollouts.

| Script | Purpose |
| --- | --- |
| `export_project_gif.py` | Render the contact-marked expert demo used in the project overview and slide 1 |
| `build_media_20261012.py` | Assemble the seven selected slide GIFs, preview, source manifest, and ZIP |
| `export_tables_20261012.py` | Render the six Markdown tables as PNGs and refresh the preview and ZIP |
| `draw_unified_architecture.py` | Draw the simple unified architecture as PNG and editable SVG |
| `media_bundle.py` | Shared preview and ZIP writer used by the two slide exporters |
| `export_slide_tables.py` | Export tables for any dated deck, replacing visible Markdown tables with PNG previews and explicit image links |

Each slide uses one concise block of individual lines without leading bullet
dashes, before its images and download links. Do not split content into separate
blocks around tables.

For new decks, always render tables as PNGs and place them under
`media/YYYY-MM-DD/`. At the table's position in the slide document, embed the
PNG and add an explicit “Use this image” link naming the file. Keep all media
for that meeting in the same dated folder; do not recreate `slide_history/figures/`.

October 19 tables can be regenerated with:

```bash
/p/part2action/envs/part2action_train/bin/python part2action/docs/slide_history/scripts/export_slide_tables.py --slides part2action/docs/slide_history/2026-10-19_part2action_meeting_status.md
```

The first export reads the Markdown tables. After they are replaced by images,
the editable table cells are retained in `media/2026-10-19/tables.json`.
Edit that JSON and rerun to update the images, manifest, preview, and ZIP.
The date is inferred from the document filename for future decks.

Run these commands from the repository root on the UVA CS installation:

```bash
/p/part2action/envs/partgym_rollout/bin/python part2action/docs/slide_history/scripts/build_media_20261012.py
/p/part2action/envs/part2action_train/bin/python part2action/docs/slide_history/scripts/export_tables_20261012.py
/p/part2action/envs/part2action_train/bin/python part2action/docs/slide_history/scripts/draw_unified_architecture.py
```

The output is `part2action/docs/slide_history/media/2026-10-12/`. The GIF builder
reuses existing comparison GIFs after checking their provenance and checksum;
add `--rerender-comparisons` to render them again. It preserves PNG table entries
and does not recreate the removed distance-gate recap. The table exporter reads
the current Markdown cells and retains the GIFs. Both rebuild the same preview
and ZIP, including only the media listed in the manifest.

The diagram shows the shared two-frame MLP and its optional normalized contact
contribution. It groups internal encoder/hierarchy connections for readability.
The diagram generator checks the selected model config and exports a PNG for
slides plus an SVG for editing. Its metadata and both files are retained by the
GIF and table exporters.

Use `--out-dir /tmp/p2a_slide_media_preview` on both exporters to create a
separate preview. The table exporter also accepts `--slides PATH`; its table
selection and column layouts are specific to the October 12 deck.

The source GIFs, rollout JSON/video files, and the PartInstruct dataset must be
available through the repository's existing storage links. Fresh comparison
rendering uses the rollout environment; table export uses Pillow and installed
DejaVu fonts. Repository paths are resolved from the script location.

To regenerate the contact-marked project demo before rebuilding slide media:

```bash
/p/part2action/envs/part2action_train/bin/python part2action/docs/slide_history/scripts/export_project_gif.py
```

Its default inputs and outputs are the GIF and metadata under
`assets/part2action/`. That metadata records the demonstration, source frames,
projection, and color meanings. The demo illustrates recorded expert behavior;
it is not a learned-policy success result.

The October12 deck is consolidated into seven slides. Existing asset filenames
retain their original export numbers so previous links remain valid; the
manifest's `slide_mapping` and preview titles identify the current slide.
New timing media uses slide06. Keep future decks within six to seven slides.
