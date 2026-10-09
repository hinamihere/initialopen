# initialopen

An open-source touge racing game in lean C++20: late-90s camcorder look, Jolt
vehicle physics, and content converted from Assetto Corsa mods you already own.

**Stack:** SDL3 (window, input, audio) · sokol_gfx on OpenGL 4.1 · Jolt Physics
(`WheeledVehicleController`) · cgltf · nlohmann/json. All fetched by CMake.

## Content and licensing

This repository contains **no game content**. Assetto Corsa tracks and cars belong
to their authors, so they are converted locally on each player's machine and never
committed (`mods_raw/` and `assets/` are git-ignored).

## Build (Windows, MinGW-w64 GCC + CMake + Ninja)

```bash
cmake -S . -B build -G Ninja
```

```bash
cmake --build build
```

## Convert mods

Extract a mod archive anywhere, then point the converter at the AC folder.
It needs only Python 3, with no extra packages.

```bash
python tools/acconv/convert.py track mods_raw/track_gunma/assettocorsa/content/tracks/pk_gunma_cycle_sports_center gcsc_full_attack assets/tracks/pk_gunma_cycle_sports_center/gcsc_full_attack
```

```bash
python tools/acconv/convert.py car mods_raw/car_r33/nissan_r33_gtr_nords_spec assets/cars/nissan_r33_gtr_nords_spec --skin Midnight_Purple
```

What gets converted:

| From the mod | Into |
|---|---|
| `.kn5` models listed in `models_<layout>.ini` | `track.glb`, merged per material, DDS textures kept compressed |
| `1ROAD` / `1GRASS` / `1WALL`… physics meshes + `surfaces.ini` | collision meshes with friction |
| `AC_PIT_*`, `AC_HOTLAP_START_*`, `AC_AB_START_*` markers | spawn points |
| `ai/fast_lane.ai` | racing line (reset-to-road, autodrive, future AI) |
| CSP `ext_config.ini` light series | streetlight positions |
| car `.kn5` + `skins/<name>` | `car.glb` with BODY, WHEEL_*, SUSP_*, STEER_HR, light groups |
| `car.ini`, `suspensions.ini`, `tyres.ini`, `engine.ini` + `power.lut`, `drivetrain.ini`, `brakes.ini` | `car.json` physics spec |

Encrypted cars (`data.acd` only) are not supported yet.

## Run

```bash
./build/initialopen.exe
```

The game picks the first track and car found under `assets/`. Useful flags:
`--track DIR`, `--car DIR`, `--spawn AC_HOTLAP_START_0`, `--time 17.8`, `--chase`,
`--autodrive`, and `--screenshot out.bmp --frames N` for headless checks.

| Keys | Gamepad | Action |
|---|---|---|
| W / S, arrows | RT / LT | throttle / brake (hold brake when stopped to reverse) |
| A / D | left stick | steer |
| Space | A | handbrake |
| L (hold) | B | flash high beams |
| H | | headlights on/off |
| C | Y | cockpit / chase camera |
| T, Q / E | LB / RB | toggle AT/MT, shift down / up |
| R / P | Back | reset to road / back to spawn |
| drag slider, hold `[` `]` | | time of day |
| F1 / F2 | | help / time slider |
