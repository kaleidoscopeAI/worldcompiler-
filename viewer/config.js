/**
 * config.js — World Compiler Globe Viewer configuration.
 *
 * Edit this file to customise the viewer before running `python serve.py`.
 *
 * CESIUM_ION_TOKEN
 *   Optional Cesium ion access token (https://ion.cesium.com).
 *   Providing a token enables Cesium World Terrain and Cesium OSM Buildings.
 *   Leave as an empty string to run in free mode (flat terrain + OSM imagery).
 *
 * WORLD_OUTPUT_PATH
 *   URL (relative to the viewer) of the directory produced by the
 *   cesium_exporter pipeline.  Default: one level up from viewer/, which
 *   is correct when serve.py is run from the repository root.
 *
 * DEFAULT_ORIGIN
 *   Fallback camera position used when no tileset is loaded or the tileset
 *   has no valid bounding sphere.  Degrees (lat, lon) and altitude in metres.
 */

window.CESIUM_ION_TOKEN = "";          // "" = free mode (no ion services)
window.WORLD_OUTPUT_PATH = "../cesium_output";   // path to generated artifacts

window.DEFAULT_ORIGIN = {
  lat:  51.5074,   // London (same as CLI default)
  lon:  -0.1278,
  altitude: 1500,  // camera height in metres above origin
};
