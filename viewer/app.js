/**
 * app.js — World Compiler Globe Viewer application logic.
 *
 * Loads:
 *   1. A base globe (free OSM imagery + flat WGS-84 terrain by default; or
 *      Cesium World Terrain when an ion token is set in config.js).
 *   2. The generated 3D Tiles tileset from WORLD_OUTPUT_PATH/tileset.json.
 *   3. Entity metadata from WORLD_OUTPUT_PATH/entities.json for the info panel.
 *
 * Controls:
 *   Left-click + drag  — orbit
 *   Right-click + drag — zoom
 *   Middle-click + drag — pan
 *   Scroll wheel       — zoom
 *   Click an object    — show label info panel
 */

(async function () {
  "use strict";

  // -------------------------------------------------------------------------
  // Ion token (optional)
  // -------------------------------------------------------------------------
  const ionToken = window.CESIUM_ION_TOKEN || "";
  if (ionToken) {
    Cesium.Ion.defaultAccessToken = ionToken;
  }

  // -------------------------------------------------------------------------
  // Imagery provider — OSM by default (free, no token)
  // -------------------------------------------------------------------------
  const imageryProvider = new Cesium.UrlTemplateImageryProvider({
    url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    maximumLevel: 19,
    credit: new Cesium.Credit(
      '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      true
    ),
  });

  // -------------------------------------------------------------------------
  // Terrain provider — flat by default; Cesium World Terrain if ion token set
  // -------------------------------------------------------------------------
  let terrainProvider;
  if (ionToken) {
    try {
      terrainProvider = await Cesium.CesiumTerrainProvider.fromIonAssetId(1);
    } catch (e) {
      console.warn("Cesium World Terrain unavailable (ion token issue?), using flat terrain:", e.message);
      terrainProvider = new Cesium.EllipsoidTerrainProvider();
    }
  } else {
    terrainProvider = new Cesium.EllipsoidTerrainProvider();
  }

  // -------------------------------------------------------------------------
  // Create viewer
  // -------------------------------------------------------------------------
  const viewer = new Cesium.Viewer("cesiumContainer", {
    terrainProvider,
    imageryProvider,
    baseLayerPicker:      false,
    geocoder:             false,
    homeButton:           true,
    sceneModePicker:      false,
    navigationHelpButton: true,
    animation:            false,
    timeline:             false,
    fullscreenButton:     true,
    infoBox:              true,
    selectionIndicator:   true,
  });

  // Improve visual quality
  viewer.scene.globe.enableLighting = true;
  viewer.scene.atmosphere.show       = true;
  viewer.scene.fog.enabled           = false;   // disable fog for local demo
  viewer.resolutionScale             = window.devicePixelRatio || 1.0;

  // -------------------------------------------------------------------------
  // Load generated 3D Tiles tileset
  // -------------------------------------------------------------------------
  const outputPath = (window.WORLD_OUTPUT_PATH || "../cesium_output").replace(/\/$/, "");
  const tilesetUrl = `${outputPath}/tileset.json`;
  const entitiesUrl = `${outputPath}/entities.json`;

  let tileset = null;
  let entitiesData = null;

  // -- entities.json (for info panel) ---------------------------------------
  try {
    const res = await fetch(entitiesUrl);
    if (res.ok) {
      entitiesData = await res.json();
      updateSidebar(entitiesData);
    }
  } catch (e) {
    console.warn("Could not load entities.json:", e.message);
  }

  // -- 3D Tiles tileset -------------------------------------------------------
  try {
    tileset = await Cesium.Cesium3DTileset.fromUrl(tilesetUrl, {
      maximumScreenSpaceError: 2,
    });
    viewer.scene.primitives.add(tileset);

    // Style: add a subtle highlight on selection (no-op default style)
    tileset.style = new Cesium.Cesium3DTileStyle({
      color: "color('white')",
    });

    // Fly camera to the tileset
    await viewer.zoomTo(tileset);

  } catch (e) {
    console.warn(
      "Could not load tileset.json — flying to default origin instead.\n" +
      "Run `python -m cesium_exporter <input>` to generate assets first.\n" +
      "Error:", e.message
    );
    flyToDefault();
  }

  if (!tileset) {
    flyToDefault();
  }

  // -------------------------------------------------------------------------
  // Click handler — show entity info
  // -------------------------------------------------------------------------
  const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
  handler.setInputAction(function (movement) {
    const picked = viewer.scene.pick(movement.position);
    if (Cesium.defined(picked) && picked.content) {
      const batchId = picked.getProperty && picked.getProperty("id");
      if (entitiesData && batchId !== undefined) {
        const ent = entitiesData.entities.find(e => e.id === batchId);
        if (ent) showInfoPanel(ent);
      }
    }
  }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

  // -------------------------------------------------------------------------
  // Helper: fly to default origin when no tileset is available
  // -------------------------------------------------------------------------
  function flyToDefault() {
    const orig = window.DEFAULT_ORIGIN || { lat: 51.5074, lon: -0.1278, altitude: 1500 };
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(orig.lon, orig.lat, orig.altitude),
      orientation: {
        heading: Cesium.Math.toRadians(0),
        pitch:   Cesium.Math.toRadians(-45),
        roll:    0.0,
      },
    });
  }

  // -------------------------------------------------------------------------
  // Sidebar: populate entity list
  // -------------------------------------------------------------------------
  function updateSidebar(data) {
    const panel = document.getElementById("sidebarContent");
    if (!panel || !data) return;

    const title = document.getElementById("sceneTitle");
    if (title) title.textContent = data.title || "World Scene";

    const stats = data.stats || {};
    const statsEl = document.getElementById("sceneStats");
    if (statsEl) {
      statsEl.textContent =
        `${data.entities.length} entities · ` +
        `${stats.chunks || "?"} chunks · ` +
        `fingerprint: ${(data.fingerprint || "").slice(0, 12)}`;
    }

    const list = document.getElementById("entityList");
    if (!list) return;
    list.innerHTML = "";
    for (const ent of data.entities) {
      const li = document.createElement("li");
      li.className = "entity-item";
      const swatch = document.createElement("span");
      swatch.className = "entity-swatch";
      const [r, g, b] = ent.color;
      swatch.style.background =
        `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`;
      li.appendChild(swatch);
      const txt = document.createElement("span");
      txt.textContent = `[${ent.type}] ${ent.label.slice(0, 60)}`;
      li.appendChild(txt);
      li.title = ent.label;
      li.addEventListener("click", () => {
        flyToEntity(ent);
        showInfoPanel(ent);
      });
      list.appendChild(li);
    }
  }

  // -------------------------------------------------------------------------
  // Info panel: show selected entity details
  // -------------------------------------------------------------------------
  function showInfoPanel(ent) {
    const panel = document.getElementById("infoPanel");
    if (!panel) return;
    panel.style.display = "block";
    document.getElementById("infoId").textContent    = ent.id;
    document.getElementById("infoType").textContent  = ent.type;
    document.getElementById("infoShape").textContent = ent.shape;
    document.getElementById("infoMass").textContent  = (ent.mass * 100).toFixed(1) + "%";
    document.getElementById("infoLabel").textContent = ent.label;
    const pos = ent.position || {};
    document.getElementById("infoPos").textContent =
      `${(pos.lat || 0).toFixed(5)}°, ${(pos.lon || 0).toFixed(5)}° · ${(pos.height || 0).toFixed(0)} m`;
  }

  document.getElementById("infoPanelClose")?.addEventListener("click", () => {
    const p = document.getElementById("infoPanel");
    if (p) p.style.display = "none";
  });

  // -------------------------------------------------------------------------
  // Fly to a specific entity
  // -------------------------------------------------------------------------
  function flyToEntity(ent) {
    const pos = ent.position || {};
    const lat = pos.lat || 0;
    const lon = pos.lon || 0;
    const h   = pos.height || 0;
    const dist = (ent.scale_m || 50) * 8;
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(lon, lat, h + dist),
      orientation: {
        heading: Cesium.Math.toRadians(0),
        pitch:   Cesium.Math.toRadians(-30),
        roll:    0,
      },
    });
  }

}());
