const $ = (id) => document.getElementById(id);
const ui = Object.fromEntries([
  "startup", "app", "dataset-path", "source-path", "output-path", "fps",
  "total-frames", "progress-count", "frame-progress", "progress-bar",
  "progress-percent", "episode-list", "state-badge", "episode-title",
  "workflow-message", "sensor-status", "camera-select", "overlay-opacity",
  "reference-image", "reference-empty", "live-image", "live-empty",
  "overlay-reference", "overlay-live", "overlay-empty", "readiness",
  "beaver-summary", "beaver-grid",
  "rollback-button", "retry-button", "cancel-teach-button", "teach-button",
  "end-teach-button", "teach-collect-button", "stop-button", "replay-button",
  "command-message",
].map((id) => [id.replaceAll("-", "_"), $(id)]));

let state = null;
let previousEpisode = Symbol("initial");
let refreshBusy = false;
let beaverBusy = false;
let beaverGridKey = "";
let beaverSensorViews = [];
let streamTimer = null;

const BEAVER_COLORS = {
  contact: [255, 26, 26],
  near: [214, 234, 255],
  far: [8, 31, 102],
  over: [110, 110, 110],
  invalid: [52, 60, 73],
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = response.statusText;
    try { message = (await response.json()).detail || message; } catch (_) {}
    throw new Error(message);
  }
  return response.json();
}

function percent() {
  if (!state.total_episodes) return 100;
  const episodePart = state.completed_episodes / state.total_episodes;
  const framePart = state.current_episode_length
    ? state.replay_frame / state.current_episode_length / state.total_episodes
    : 0;
  return Math.min(100, (episodePart + framePart) * 100);
}

function renderEpisodes(episodeChanged) {
  ui.episode_list.replaceChildren(...state.episodes.map((episode) => {
    const row = document.createElement("div");
    row.className = `episode-row ${episode.status}${episode.replaced ? " replaced" : ""}`;
    const identity = document.createElement("span");
    identity.textContent = `Episode ${episode.source_episode_index}`;
    const length = document.createElement("span");
    const currentSafety = episode.status === "current"
      ? state.trajectory_safety
      : null;
    const safetyLabel = currentSafety?.repaired
      ? " · repaired"
      : currentSafety?.unsafe_original
        ? " · original"
        : "";
    const replacedLabel = episode.replaced
      ? ` · taught ${episode.recorded_length} frames`
      : "";
    length.textContent = `${episode.length} frames${safetyLabel}${replacedLabel}`;
    row.append(identity, length);
    return row;
  }));
  if (episodeChanged) {
    const current = ui.episode_list.querySelector(".current");
    if (current) {
      const listRect = ui.episode_list.getBoundingClientRect();
      const rowRect = current.getBoundingClientRect();
      if (rowRect.top < listRect.top) {
        ui.episode_list.scrollTop -= listRect.top - rowRect.top;
      } else if (rowRect.bottom > listRect.bottom) {
        ui.episode_list.scrollTop += rowRect.bottom - listRect.bottom;
      }
    }
  }
}

function setImage(image, empty, available, url) {
  if (!available) {
    image.classList.add("hidden");
    empty.classList.remove("hidden");
    image.removeAttribute("src");
    delete image.dataset.loadedUrl;
    return;
  }
  if (image.dataset.loadedUrl === url) {
    image.classList.remove("hidden");
    empty.classList.add("hidden");
    return;
  }
  image.onload = () => {
    image.classList.remove("hidden");
    empty.classList.add("hidden");
  };
  image.onerror = () => {
    image.classList.add("hidden");
    empty.classList.remove("hidden");
    delete image.dataset.loadedUrl;
  };
  image.dataset.loadedUrl = url;
  image.src = url;
}

function updateCameraSources() {
  const camera = ui.camera_select.value;
  if (!camera) return;
  const token = Date.now();
  const hasReference = state.reference_cameras.includes(camera);
  const hasLive = state.live_cameras.includes(camera);
  const referenceUrl = `/api/cameras/reference/${encodeURIComponent(camera)}.jpg?v=${state.reference_generation ?? state.current_source_episode}`;
  setImage(
    ui.reference_image,
    ui.reference_empty,
    hasReference,
    referenceUrl,
  );
  setImage(
    ui.overlay_reference,
    ui.overlay_empty,
    hasReference && hasLive,
    referenceUrl,
  );
  setImage(
    ui.live_image,
    ui.live_empty,
    hasLive,
    `/api/cameras/live/${encodeURIComponent(camera)}.jpg?t=${token}`,
  );
  setImage(
    ui.overlay_live,
    ui.overlay_empty,
    hasReference && hasLive,
    `/api/cameras/live/${encodeURIComponent(camera)}.jpg?t=${token}`,
  );
  ui.overlay_empty.classList.toggle("hidden", hasReference && hasLive);
}

function renderCameraSelector(episodeChanged) {
  const names = state.cameras;
  const previous = ui.camera_select.value;
  const changed = names.join("|") !== [...ui.camera_select.options].map((o) => o.value).join("|");
  if (changed) {
    ui.camera_select.replaceChildren(...names.map((name) => new Option(name, name)));
    ui.camera_select.value = names.includes(previous) ? previous : names[0];
  }
  updateCameraSources();
}

function beaverCellAppearance(value, valid, maxDistance) {
  if (!valid || !Number.isFinite(value) || value < 0) {
    return { background: BEAVER_COLORS.invalid, foreground: [230, 237, 242] };
  }
  if (value === 0) {
    return { background: BEAVER_COLORS.contact, foreground: [255, 255, 255] };
  }
  if (value > maxDistance) {
    return { background: BEAVER_COLORS.over, foreground: [255, 255, 255] };
  }
  const steps = Math.max(2, Math.round(maxDistance));
  const index = Math.max(0, Math.min(steps - 1, Math.floor(value - 1)));
  const mix = index / (steps - 1);
  const background = BEAVER_COLORS.near.map((channel, offset) =>
    Math.round(channel + (BEAVER_COLORS.far[offset] - channel) * mix));
  const luminance = (background[0] * 299 + background[1] * 587 + background[2] * 114) / 1000;
  return {
    background,
    foreground: luminance > 145 ? [7, 20, 35] : [245, 249, 255],
  };
}

function rgb(channels) {
  return `rgb(${channels.join(",")})`;
}

function ensureBeaverGrid(layout, gridWidth, sensorCount) {
  const key = JSON.stringify([layout, gridWidth, sensorCount]);
  if (key === beaverGridKey) return;

  const cards = [];
  beaverSensorViews = [];
  for (let slot = 0; slot < sensorCount; slot += 1) {
    const sensor = layout[slot] || [0, slot];
    const card = document.createElement("article");
    card.className = "beaver-sensor offline";

    const header = document.createElement("div");
    header.className = "beaver-sensor-head";
    const name = document.createElement("strong");
    name.textContent = `B${sensor[0]}S${sensor[1]}`;
    const sensorStatus = document.createElement("span");
    sensorStatus.textContent = "waiting";
    header.append(name, sensorStatus);

    const pixels = document.createElement("div");
    pixels.className = `beaver-pixels${gridWidth === 8 ? " grid-8" : ""}`;
    pixels.style.setProperty("--beaver-width", gridWidth);
    const cells = [];
    for (let index = 0; index < gridWidth * gridWidth; index += 1) {
      const cell = document.createElement("span");
      cell.className = "beaver-cell";
      pixels.append(cell);
      cells.push(cell);
    }

    const metrics = document.createElement("div");
    metrics.className = "beaver-metrics";
    const minimum = document.createElement("span");
    const average = document.createElement("span");
    metrics.append(minimum, average);
    card.append(header, pixels, metrics);
    cards.push(card);
    beaverSensorViews.push({
      card,
      sensor,
      sensorStatus,
      cells,
      minimum,
      average,
    });
  }
  ui.beaver_grid.replaceChildren(...cards);
  beaverGridKey = key;
}

function renderBeaver(beaver) {
  const layout = Array.isArray(beaver.sensor_layout) ? beaver.sensor_layout : [];
  const distances = Array.isArray(beaver.distance_mm) ? beaver.distance_mm : [];
  const statuses = Array.isArray(beaver.target_status) ? beaver.target_status : [];
  const present = Array.isArray(beaver.present) ? beaver.present : [];
  const gridWidth = Number(beaver.grid_width) || 4;
  const sensorCount = Math.max(layout.length, distances.length, present.length);
  const fresh = Boolean(beaver.connected) && !beaver.stale;
  const age = beaver.age_ms === null || beaver.age_ms === undefined
    ? ""
    : ` · ${Number(beaver.age_ms).toFixed(0)} ms`;
  ui.beaver_summary.textContent = fresh
    ? `${beaver.wire_distance_bits}-bit raw · ${gridWidth}×${gridWidth} · frame ${beaver.frame_count} · lost ${beaver.lost_frames}${age}`
    : `Waiting · ${beaver.error || "Beaver data is stale"}`;
  ui.beaver_summary.className = `beaver-summary ${fresh ? "ok" : "waiting"}`;

  if (!beaver.enabled || sensorCount === 0) {
    ui.beaver_grid.replaceChildren();
    beaverGridKey = "";
    beaverSensorViews = [];
    return;
  }

  ensureBeaverGrid(layout, gridWidth, sensorCount);
  const maxDistance = Math.max(1, Number(beaver.max_display_mm) || 400);
  for (let slot = 0; slot < sensorCount; slot += 1) {
    const view = beaverSensorViews[slot];
    const sensor = view.sensor;
    const online = Boolean(present[slot]);
    const sensorState = online ? (beaver.stale ? "stale" : "online") : "missing";
    view.card.className = `beaver-sensor ${online && !beaver.stale ? "online" : "offline"}`;
    view.sensorStatus.textContent = sensorState;
    const validValues = [];
    for (let row = 0; row < gridWidth; row += 1) {
      for (let column = 0; column < gridWidth; column += 1) {
        const value = Number(distances[slot]?.[row]?.[column]);
        const targetStatus = Number(statuses[slot]?.[row]?.[column]);
        const valid = online && (targetStatus === 5 || targetStatus === 9)
          && Number.isFinite(value) && value >= 0;
        if (valid) validValues.push(value);
        const appearance = beaverCellAppearance(value, valid, maxDistance);
        const cell = view.cells[row * gridWidth + column];
        cell.style.backgroundColor = rgb(appearance.background);
        cell.style.color = rgb(appearance.foreground);
        cell.textContent = valid ? String(value) : "—";
        cell.setAttribute(
          "aria-label",
          `B${sensor[0]}S${sensor[1]} row ${row + 1} column ${column + 1}: ${valid ? `${value} millimetres` : "invalid"}`,
        );
      }
    }

    view.minimum.textContent = validValues.length
      ? `min ${Math.min(...validValues).toFixed(1)} mm`
      : "min —";
    view.average.textContent = validValues.length
      ? `avg ${(validValues.reduce((sum, value) => sum + value, 0) / validValues.length).toFixed(1)} mm`
      : "avg —";
  }
}

async function refreshBeaver() {
  if (beaverBusy) return;
  beaverBusy = true;
  try {
    renderBeaver(await api("/api/beaver"));
  } catch (error) {
    ui.beaver_summary.textContent = `Could not read Beaver data: ${error.message}`;
    ui.beaver_summary.className = "beaver-summary waiting";
  } finally {
    beaverBusy = false;
  }
}

function render() {
  const episodeChanged = previousEpisode !== state.current_source_episode;
  previousEpisode = state.current_source_episode;
  ui.dataset_path.textContent = state.output_dataset;
  ui.source_path.textContent = state.source_dataset;
  ui.output_path.textContent = state.output_dataset;
  ui.fps.textContent = `${state.fps} fps`;
  ui.total_frames.textContent = `${state.total_frames} frames`;
  ui.progress_count.textContent = `${state.completed_episodes} / ${state.total_episodes} episodes`;
  const replacing = Boolean(state.collecting_replacement);
  ui.frame_progress.textContent = replacing && (state.state === "teaching" || state.state === "teach_ready")
    ? `taught ${state.taught_frames} frames · source ${state.current_episode_length}`
    : state.current_episode_length
      ? `frame ${state.replay_frame} / ${state.current_episode_length}`
      : "all frames exported";
  const progress = percent();
  ui.progress_bar.style.width = `${progress}%`;
  ui.progress_percent.textContent = `${progress.toFixed(1)}%`;
  ui.state_badge.textContent = state.state.replaceAll("_", " ");
  ui.state_badge.className = `state-badge state-${state.state}`;
  ui.episode_title.textContent = state.current_source_episode === null
    ? "Recollection complete"
    : replacing
      ? `Episode ${state.current_source_episode} · teach replacement (${state.taught_frames} frames)`
      : `Episode ${state.current_source_episode} · ${state.current_episode_length} frames`;
  ui.workflow_message.textContent = state.message;
  const beaver = state.beaver;
  const trajectorySafety = state.trajectory_safety || {};
  const beaverReady = beaver.connected && beaver.wire_distance_bits === 16;
  ui.sensor_status.innerHTML = `
    <span class="sensor ${beaverReady ? "ok" : "waiting"}">Beaver ${beaver.precision}</span>
    <span class="sensor ${state.live_cameras.length ? "ok" : "waiting"}">${state.live_cameras.length} live camera(s)</span>
    <span class="sensor">${beaver.frame_count || 0} Beaver frames · ${beaver.lost_frames || 0} lost</span>
    ${trajectorySafety.repaired
      ? `<span class="sensor waiting">Trajectory repaired · ${trajectorySafety.original_max_joint_speed.toFixed(3)} → ${trajectorySafety.replay_max_joint_speed.toFixed(3)} rad/s</span>`
      : trajectorySafety.unsafe_original
        ? `<span class="sensor waiting">Original trajectory · ${trajectorySafety.original_max_joint_speed.toFixed(3)} rad/s exceeds ${trajectorySafety.safety_limit.toFixed(3)} warning limit</span>`
        : ""}`;
  ui.readiness.textContent = state.readiness_issues.length
    ? `Waiting: ${state.readiness_issues.join(" · ")}`
    : state.state === "calibrating"
      ? "Sensors ready. Confirm the camera alignment, then press Enter to replay or T to teach a replacement."
      : state.state === "teaching"
        ? "Freedrive is on. Drag a new path, then press End teach. The next original episode is unchanged."
        : state.state === "teach_ready"
          ? "Arm is at the taught start. Set up the scene, then press Collect replacement. The next slot stays the original dataset episode."
          : "";
  ui.readiness.classList.toggle("blocked", state.readiness_issues.length > 0);
  ui.replay_button.disabled = !state.controls.replay_enabled;
  ui.replay_button.classList.toggle("hidden", state.state === "teaching" || state.state === "teach_ready");
  ui.teach_button.disabled = !state.controls.teach_enabled;
  ui.teach_button.classList.toggle("hidden", state.state === "teaching");
  ui.teach_button.textContent = state.state === "teach_ready" ? "Reteach replacement · T" : "Teach replacement · T";
  ui.end_teach_button.disabled = !state.controls.end_teach_enabled;
  ui.end_teach_button.classList.toggle("hidden", state.state !== "teaching");
  ui.teach_collect_button.disabled = !state.controls.teach_collect_enabled;
  ui.teach_collect_button.classList.toggle("hidden", state.state !== "teach_ready");
  ui.cancel_teach_button.disabled = !state.controls.cancel_teach_enabled;
  ui.cancel_teach_button.classList.toggle("hidden", !state.controls.cancel_teach_enabled);
  ui.retry_button.disabled = !state.controls.retry_enabled;
  ui.rollback_button.disabled = !state.controls.rollback_enabled;
  ui.stop_button.disabled = !state.controls.stop_enabled;
  renderEpisodes(episodeChanged);
  renderCameraSelector(episodeChanged);
}

async function refreshState() {
  if (refreshBusy) return;
  refreshBusy = true;
  try {
    state = await api("/api/state");
    ui.startup.classList.add("hidden");
    ui.app.classList.remove("hidden");
    render();
  } catch (error) {
    ui.startup.textContent = `Could not reach recollector: ${error.message}`;
    ui.startup.style.color = "var(--danger)";
  } finally {
    refreshBusy = false;
  }
}

async function command(name) {
  ui.command_message.classList.remove("error");
  ui.command_message.textContent = `Sending ${name}…`;
  try {
    await api(`/api/commands/${name}`, { method: "POST" });
    ui.command_message.textContent = `${name} accepted`;
    await refreshState();
  } catch (error) {
    ui.command_message.textContent = error.message;
    ui.command_message.classList.add("error");
  }
}

function bindEvents() {
  ui.replay_button.addEventListener("click", () => command("continue"));
  ui.teach_button.addEventListener("click", () => command(state?.state === "teach_ready" ? "reteach" : "teach"));
  ui.end_teach_button.addEventListener("click", () => command("end_teach"));
  ui.teach_collect_button.addEventListener("click", () => command("teach_collect"));
  ui.cancel_teach_button.addEventListener("click", () => command("cancel_teach"));
  ui.retry_button.addEventListener("click", () => command("retry"));
  ui.rollback_button.addEventListener("click", () => command("rollback"));
  ui.stop_button.addEventListener("click", () => command("stop"));
  ui.camera_select.addEventListener("change", () => updateCameraSources());
  ui.overlay_opacity.addEventListener("input", () => {
    ui.overlay_live.style.opacity = Number(ui.overlay_opacity.value) / 100;
  });
  document.addEventListener("keydown", (event) => {
    const tag = event.target.tagName?.toLowerCase();
    if (["input", "select", "button", "textarea"].includes(tag)) return;
    if (event.key === "Enter" && !event.repeat) {
      if (state?.controls.end_teach_enabled) {
        event.preventDefault();
        command("end_teach");
      } else if (state?.controls.teach_collect_enabled) {
        event.preventDefault();
        command("teach_collect");
      } else if (state?.controls.replay_enabled) {
        event.preventDefault();
        command("continue");
      }
    } else if (event.key.toLowerCase() === "t" && !event.repeat) {
      if (state?.state === "teach_ready" && state?.controls.teach_enabled) {
        event.preventDefault();
        command("reteach");
      } else if (state?.controls.teach_enabled) {
        event.preventDefault();
        command("teach");
      }
    } else if (event.key.toLowerCase() === "r" && state?.controls.retry_enabled && !event.repeat) {
      event.preventDefault();
      command("retry");
    } else if (event.key === "Escape" && state?.controls.stop_enabled) {
      event.preventDefault();
      command("stop");
    }
  });
}

bindEvents();
refreshState();
refreshBeaver();
window.setInterval(refreshState, 300);
window.setInterval(refreshBeaver, 100);
streamTimer = window.setInterval(() => {
  if (state && ["calibrating", "teaching", "teach_ready", "replaying", "moving_to_start"].includes(state.state)) {
    updateCameraSources();
  }
}, 120);
