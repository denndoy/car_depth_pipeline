# Vehicle 3D assets

The three source `.blend` assets in this folder are used by
`scripts/12_generate_urban_car_scene.py`:

| File | Visual class | Embedded dimensions (approx.) |
|---|---|---:|
| `1990s-hatchback.blend` | Hatchback | 1.856 × 4.064 × 1.534 m |
| `1990s-minivan.blend` | MPV / minivan | 1.987 × 4.574 × 1.880 m |
| `1990s-sedan.blend` | Sedan | 1.896 × 4.674 × 1.500 m |

Blender root-object metadata identifies the set as **“Cozy Retro Cars Kit”**,
source **Grab3D**, license **CC0-1.0** (created 2026-10-19). The models are
generic illustrative 1990s vehicles, not representations of the study cars.
They are static meshes with packed PBR textures; the four wheels are separate
objects and can be animated around their local X axes if desired.

These files were opened, rendered, exported to GLB, and re-imported in Blender
5.2.2 LTS. GLB export reported a material warning where multiple image nodes
feed one sampler; inspect the final viewer render before publication.
