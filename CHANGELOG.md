# Changelog

All notable changes to Anime Expeditions (VFERDZ-AE-MACRO) are documented here.

## [0.24.5] - 2026-10-16

### Improved
- **Team Loadout Process**: new update on macro team loadout due to game interface changes

## [0.24.4] - 2026-10-06

### Fixed
- Challenge Story Maps in Resource Tab are now real-time update when adding a new map
- Enhanced the detection mechanism for the "Include" button to retry detection and clicking based on the configured threshold in the Image Manager, rather than using a fixed value.

## [0.24.3] - 2026-10-04

### New
- **Progressive Mode**: added task for boss rush mode and map
- **Event Mode**: added the Infernal Cult Story Event Mode
- **Assets**: added new map, assets for the new update
- **Resource in Resource Tab**: added boss rush routes (user must manually record walk path per gates)
- **Macro Coordinates**: added new boss rush pick card coordinate for the boss rush progressive mode

### Improved
- Modified little bit on the GUI Task Builder for the new mode

## [0.24.0 - 0.24.2] - 2026-09-26

### New
- **New Update - Update 3 Endless Hellflame**: added new story mode, new assets, new mode, new folder map
- **New Story Map**: added flaming monastery map, new assets

### Fixed
- Fixed: Cant Manually Add New Maps -> Now automatically display/show the new map especially in the resource tab global map
- Fixed and Modified flaming monastery map due to spawn update/move

## [0.23.0] - 2026-09-11

### New
- **Added fishing mode**: add new fishhook and unwanted fish assets 

### Fixed
- **Fixed Target Priority Block**: fix on clicking unit priority orders, added new one and modified

## [0.22.0 - 0.22.2] - 2026-09-10

### New
- **New Update 2.5 Absolute Dream**: added the new update, new mode, story mode event, new maps
- **New Mode**: added the new story event mode; Golden Hour and Eclipse Infinite Stage 
- **New Assets**: eclipsed assets; sacrificed soul & redeemed soul, new mode assets; golden hour & eclipsed infinite story
- **Eclipse Card Selection**: add macro coordinate for the card selection and detection

### Improved
- Summer Event Mode & Portal Mode Task
- Modified the task process, ui, select options and etc
- Remove unused units
- Eclipsed Card Selection Process

## [0.21.0 - 0.21.6] - 2026-09-09

### New
- **Built-in Walk Path**: added new path built-in path for the new map: Crimson Shore, Snowy Castle, Summer Event Mode and Summer Portal
- **Logo**: replace the old logo
- **Macro Header Name**: replace macro header name to VFERDZ
-

### Improved
- **Ownership**: Repo Transfer Ownership
- Refactor workflows and update documentation for clarity and accuracy
- **Modified Files**: update README with new game features and installation instructions
- **Challenge Image Detection and Function**: fixed the image match
- **New/Replace Assets**: challenge, daily challenge, wave detect
- **Wave Detection/Monitor**: fixed to be accurate on detect when to stop

### Fixed
- Wave Check in Infinite 

## [0.20.0 - 0.20.3] - 2026-09-08

### New 
- **Macro changes**: replace Logo, rename GUI branding to VFERDZ macro, 
- **Creams Macro Labels to VFERDZ**: replace the remaining Creams Macro labels in the main dashboard, loading screen, welcome dialog, logs window, and wave monitor.
- **2 Summer Event Mode**: added event mode and portal mode

### Improved
- **Modified GUI**: Improve some interface
- **New Assets**: Added new assets to improve the detection
- **Modified Files**: readme, docs, user guide

## [0.19.1] - 2026-08-13

### Improved
- **Auto Fuel interval control**: the minutes/hours fields now remain available while Auto is selected, so a custom refill wait can be entered instead of being locked to the automatic 8-hour Max interval.
- **Expedition encounters**: added native encounter handling, routes, and screen references for East Town, Flower Forest, Rose Kingdom, and School Grounds.
- **Expedition reliability**: improved checkpoint, wave counter, Repeat Stage, Start Game, upgrade-card, and unit re-placement handling.
- **Navigation recovery**: widened card searches, added lobby re-sync, and exits the AFK Chamber when it blocks progress.
- **macOS**: stabilized code-signing identity across updates and fixed the Macro Manager panel rendering blank while the macro runs.

### Fixed
- Villian Invasion navigation now opens the event card before selecting the game mode.
- East Town is now available in the Challenge and Bounty Story map list.

## [0.19.0] - 2026-08-11

### New
- **East Town map**: added to the expedition and story map lists.
- **Tower game mode**: Play -> Tower -> Select Stage -> Start (solo). Wins advance floors with `Next_Floor`; defeats retry with `Repeat_Floor`. Supports Normal and Traitless Tower. No map dropdown in the builder; Rose Kingdom is the internal default.
- **Auto Fuel custom interval**: set any refill interval in minutes or hours (e.g. 30 minutes, 1 hour), or leave it on Auto to keep the per-amount default behavior.

### Improved
- **Event mode**: waits for the Event gamemode screen, clicks a user-configurable card coordinate, then image-clicks the Event Gamemode button.
- **Disconnect recovery**: kills a stuck Roblox client before deep-link relaunch, still respecting the multi-window guard.
- **File dialogs**: cancelled or failed native file dialogs now return clean results instead of rejected JS promises.
- **Auto Upgrade Unit**: bounded wait for the unit info panel before searching `priority_upgrade`, so slow-rendering panels are no longer skipped. New `quote_on` / `quote_off` reference images for user-built Detect conditions.
- **OCR overhaul**:
  - Auto Bounty wave OCR: wave-anchored parsing with card-local crops and contrast voting. Clipped wave numbers (`6`, `6C`) now resolve to `60` instead of ending the run at wave 6.
  - Optional RapidOCR engine layer (separate `requirements-rapidocr.txt`, Python 3.13-safe) with a RapidOCR -> Windows OCR -> Tesseract fallback chain.
  - Windows OCR output is always filtered by the config whitelist, preserving stats/wave/shop reads.
  - Daily Challenge map OCR keeps the HUD-anchored crop primary and adds a fixed relative top-right crop as fallback.
- **Packaging**: PyInstaller keeps `winsdk`/`winrt` collection and adds RapidOCR data only when installed.

### Fixed
- Auto Bounty no longer exits early on wave-60 bounties when OCR clips the trailing zero.
- Challenge map OCR recovers when the Daily Challenge HUD label is not found.
