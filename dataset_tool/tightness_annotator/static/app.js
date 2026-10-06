"use strict";

const ui = Object.fromEntries(
  [
    "startup", "app", "dataset-name", "dataset-path", "progress-count", "remaining-count",
    "progress-bar", "progress-percent", "episode-jump", "jump-button", "episode-list",
    "episode-title", "annotation-status", "camera-select", "video-frame", "frame-loading",
    "frame-error", "frame-count", "time-readout", "fps-readout", "timeline-shell",
    "label-track", "transition-marker", "timeline", "timeline-end", "play-button",
    "selection-title", "selection-summary", "mark-button", "no-tight-button", "clear-button",
    "save-message", "save-button", "save-next-button", "previous-episode", "next-episode",
  ].map((id) => [id.replaceAll("-", "_"), document.getElementById(id)])
);

let state = null;
let episode = null;
let currentFrame = 0;
let selection = { mode: "unset", frame: null };
let savedSelection = { mode: "unset", frame: null };
let dirty = false;
let playing = false;
let frameRequest = null;
let frameToken = 0;
let displayedObjectUrl = null;
let displayedFrame = null;

const selectionKey = (value) => `${value.mode}:${value.frame ?? ""}`;

async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).detail || message; } catch (_) { /* response is not JSON */ }
    throw new Error(message);
  }
  return response.json();
}

function annotationToSelection(annotation) {
  if (!annotation.annotated) return { mode: "unset", frame: null };
  if (!annotation.has_tight_grasp) return { mode: "no-tight", frame: null };
  return { mode: "tight", frame: annotation.tight_frame_index };
}

function setDirty() {
  dirty = selectionKey(selection) !== selectionKey(savedSelection);
  renderAnnotation();
}

function renderProgress() {
  ui.progress_count.textContent = `${state.annotated_count} / ${state.total_episodes} annotated`;
  ui.remaining_count.textContent = `${state.remaining_count} remaining`;
  ui.progress_percent.textContent = `${state.progress_percent.toFixed(1)}%`;
  ui.progress_bar.style.width = `${state.progress_percent}%`;
}

function renderEpisodeList() {
  ui.episode_list.replaceChildren();
  state.episodes.forEach((record) => {
    const button = document.createElement("button");
    button.className = `episode-item ${record.annotated ? "annotated" : ""}`;
    if (episode && record.episode_index === episode.episode_index) button.classList.add("active");
    button.innerHTML = `<span class="episode-index"><span class="episode-dot"></span>${record.episode_index}</span><span class="episode-state">${record.annotated ? "Done" : "Open"}</span>`;
    button.addEventListener("click", () => navigateTo(record.episode_index));
    ui.episode_list.append(button);
  });
  requestAnimationFrame(() => ui.episode_list.querySelector(".active")?.scrollIntoView({ block: "nearest" }));
}

function renderFrameReadout() {
  if (!episode) return;
  const last = episode.last_frame_index;
  const timestamp = episode.timestamps[currentFrame];
  ui.frame_count.textContent = `Frame ${currentFrame} / ${last} (${episode.frame_count} frames)`;
  ui.time_readout.textContent = `${timestamp.toFixed(3)} s`;
  ui.fps_readout.textContent = `${episode.fps} FPS`;
  ui.timeline.value = currentFrame;
  ui.timeline.setAttribute("aria-valuetext", `Frame ${currentFrame} of ${last}, ${timestamp.toFixed(3)} seconds`);
}

function renderTimelineLabels() {
  ui.timeline_shell.classList.remove("neutral", "tight", "no-tight");
  ui.transition_marker.classList.add("hidden");
  if (selection.mode === "tight") {
    ui.timeline_shell.classList.add("tight");
    const boundary = selection.frame === 0 || episode.last_frame_index === 0
      ? 0
      : ((selection.frame - 0.5) / episode.last_frame_index) * 100;
    const marker = episode.last_frame_index === 0 ? 0 : (selection.frame / episode.last_frame_index) * 100;
    ui.timeline_shell.style.setProperty("--boundary", `${boundary}%`);
    ui.timeline_shell.style.setProperty("--marker", `${marker}%`);
    ui.transition_marker.classList.remove("hidden");
  } else if (selection.mode === "no-tight") {
    ui.timeline_shell.classList.add("no-tight");
  } else {
    ui.timeline_shell.classList.add("neutral");
  }
}

function renderAnnotation() {
  if (!episode) return;
  const annotation = episode.annotation;
  ui.annotation_status.textContent = annotation.annotated ? "Annotated" : "Not annotated";
  ui.annotation_status.className = `status-badge ${annotation.annotated ? "annotated" : "unannotated"}`;
  if (selection.mode === "tight") {
    const k = selection.frame;
    const time = episode.timestamps[k];
    ui.selection_title.textContent = `Transition at frame ${k}`;
    ui.selection_summary.textContent = k === 0
      ? `At ${time.toFixed(3)} s · label 1 → frames 0–${episode.last_frame_index} (no label-0 frames)`
      : `At ${time.toFixed(3)} s · label 0 → frames 0–${k - 1} · label 1 → frames ${k}–${episode.last_frame_index}`;
  } else if (selection.mode === "no-tight") {
    ui.selection_title.textContent = "No tight grasp";
    ui.selection_summary.textContent = `All ${episode.frame_count} frames will be labeled 0. This is saved explicitly, not treated as unfinished.`;
  } else {
    ui.selection_title.textContent = "Choose a transition";
    ui.selection_summary.textContent = "Scrub to the first securely tightened frame, then mark it, or choose No tight grasp.";
  }
  const canSave = selection.mode !== "unset" && (dirty || !annotation.annotated);
  ui.mark_button.disabled = displayedFrame === null;
  ui.save_button.disabled = !canSave;
  ui.save_next_button.disabled = selection.mode === "unset";
  ui.clear_button.disabled = selection.mode === "unset";
  renderTimelineLabels();
}

async function showFrame() {
  if (!episode) return false;
  const token = ++frameToken;
  const requestedFrame = currentFrame;
  frameRequest?.abort();
  frameRequest = new AbortController();
  ui.frame_loading.classList.remove("hidden");
  ui.frame_error.classList.add("hidden");
  const camera = ui.camera_select.value;
  const url = `/api/episodes/${episode.episode_index}/frames/${currentFrame}?camera=${encodeURIComponent(camera)}`;
  try {
    const response = await fetch(url, { signal: frameRequest.signal });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    const blob = await response.blob();
    if (token !== frameToken) return false;
    const objectUrl = URL.createObjectURL(blob);
    ui.video_frame.src = objectUrl;
    await ui.video_frame.decode();
    if (token !== frameToken) {
      URL.revokeObjectURL(objectUrl);
      return false;
    }
    if (displayedObjectUrl) URL.revokeObjectURL(displayedObjectUrl);
    displayedObjectUrl = objectUrl;
    displayedFrame = requestedFrame;
    ui.frame_loading.classList.add("hidden");
    renderFrameReadout();
    renderAnnotation();
    return true;
  } catch (error) {
    if (error.name === "AbortError") return false;
    if (token === frameToken) {
      ui.frame_loading.classList.add("hidden");
      ui.frame_error.textContent = `Could not decode this frame: ${error.message}`;
      ui.frame_error.classList.remove("hidden");
    }
    return false;
  }
}

async function setFrame(value, { stop = true } = {}) {
  if (!episode) return;
  if (stop) stopPlayback();
  currentFrame = Math.max(0, Math.min(episode.last_frame_index, Number(value)));
  ui.timeline.value = currentFrame;
  await showFrame();
}

function stopPlayback() {
  playing = false;
  ui.play_button.textContent = "▶ Play";
}

async function playbackLoop() {
  if (!playing || !episode) return;
  if (currentFrame >= episode.last_frame_index) {
    stopPlayback();
    return;
  }
  const started = performance.now();
  currentFrame += 1;
  await showFrame();
  if (!playing) return;
  const remaining = Math.max(0, 1000 / episode.fps - (performance.now() - started));
  window.setTimeout(playbackLoop, remaining);
}

function togglePlayback() {
  if (playing) {
    stopPlayback();
    return;
  }
  if (currentFrame >= episode.last_frame_index) currentFrame = 0;
  playing = true;
  ui.play_button.textContent = "❚❚ Pause";
  playbackLoop();
}

function markCurrent() {
  if (displayedFrame === null) return;
  stopPlayback();
  selection = { mode: "tight", frame: displayedFrame };
  ui.save_message.textContent = "";
  setDirty();
}

function markNoTight() {
  stopPlayback();
  selection = { mode: "no-tight", frame: null };
  ui.save_message.textContent = "";
  setDirty();
}

function clearDraft() {
  stopPlayback();
  selection = { mode: "unset", frame: null };
  ui.save_message.textContent = episode.annotation.annotated
    ? "Saved annotation is unchanged until you choose and save a replacement."
    : "";
  setDirty();
}

async function saveAnnotation({ next = false } = {}) {
  if (selection.mode === "unset") {
    ui.save_message.textContent = "Choose a transition or No tight grasp before saving.";
    ui.save_message.classList.add("error");
    return false;
  }
  stopPlayback();
  ui.save_button.disabled = true;
  ui.save_next_button.disabled = true;
  ui.save_message.classList.remove("error");
  ui.save_message.textContent = "Saving to dataset…";
  try {
    const result = await api(`/api/episodes/${episode.episode_index}/annotation`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        has_tight_grasp: selection.mode === "tight",
        tight_frame_index: selection.mode === "tight" ? selection.frame : null,
      }),
    });
    episode.annotation = result.annotation;
    savedSelection = { ...selection };
    dirty = false;
    state.annotated_count = result.annotated_count;
    state.remaining_count = result.remaining_count;
    state.progress_percent = result.progress_percent;
    state.resume_episode = result.resume_episode;
    state.episodes[episode.episode_index] = result.annotation;
    ui.save_message.textContent = `Saved episode ${episode.episode_index}`;
    renderProgress();
    renderEpisodeList();
    renderAnnotation();
    if (next && episode.episode_index < state.total_episodes - 1) {
      await loadEpisode(episode.episode_index + 1, { skipDirtyCheck: true });
    }
    return true;
  } catch (error) {
    ui.save_message.textContent = `Save failed: ${error.message}`;
    ui.save_message.classList.add("error");
    renderAnnotation();
    return false;
  }
}

async function loadEpisode(index, { skipDirtyCheck = false } = {}) {
  index = Number(index);
  if (!Number.isInteger(index) || index < 0 || index >= state.total_episodes) return;
  if (!skipDirtyCheck && dirty && !window.confirm("Discard the unsaved annotation draft?")) return;
  stopPlayback();
  frameToken += 1;
  frameRequest?.abort();
  displayedFrame = null;
  episode = await api(`/api/episodes/${index}`);
  selection = annotationToSelection(episode.annotation);
  savedSelection = { ...selection };
  dirty = false;
  currentFrame = selection.mode === "tight" ? selection.frame : 0;
  ui.episode_title.textContent = `Episode ${index} / ${state.total_episodes - 1}`;
  ui.episode_jump.value = index;
  ui.timeline.max = episode.last_frame_index;
  ui.timeline.value = currentFrame;
  ui.timeline_end.textContent = episode.last_frame_index;
  ui.previous_episode.disabled = index === 0;
  ui.next_episode.disabled = index === state.total_episodes - 1;
  ui.save_message.textContent = "";
  ui.save_message.classList.remove("error");
  ui.camera_select.replaceChildren(
    ...episode.cameras.map((key) => new Option(key, key))
  );
  renderEpisodeList();
  renderAnnotation();
  await showFrame();
}

async function navigateTo(index) {
  try { await loadEpisode(index); }
  catch (error) {
    ui.save_message.textContent = `Could not load episode: ${error.message}`;
    ui.save_message.classList.add("error");
  }
}

function isEditableTarget(target) {
  if (target.isContentEditable) return true;
  const tag = target.tagName?.toLowerCase();
  if (tag === "textarea" || tag === "select" || tag === "button") return true;
  return tag === "input" && target.type !== "range";
}

function bindEvents() {
  ui.timeline.addEventListener("input", () => setFrame(ui.timeline.value));
  document.querySelectorAll("[data-step]").forEach((button) =>
    button.addEventListener("click", () => setFrame(currentFrame + Number(button.dataset.step)))
  );
  ui.play_button.addEventListener("click", togglePlayback);
  ui.mark_button.addEventListener("click", markCurrent);
  ui.no_tight_button.addEventListener("click", markNoTight);
  ui.clear_button.addEventListener("click", clearDraft);
  ui.save_button.addEventListener("click", () => saveAnnotation());
  ui.save_next_button.addEventListener("click", () => saveAnnotation({ next: true }));
  ui.previous_episode.addEventListener("click", () => navigateTo(episode.episode_index - 1));
  ui.next_episode.addEventListener("click", () => navigateTo(episode.episode_index + 1));
  ui.jump_button.addEventListener("click", () => navigateTo(ui.episode_jump.value));
  ui.episode_jump.addEventListener("keydown", (event) => {
    if (event.key === "Enter") navigateTo(ui.episode_jump.value);
  });
  ui.camera_select.addEventListener("change", () => showFrame());
  window.addEventListener("beforeunload", (event) => {
    if (!dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });
  document.addEventListener("keydown", (event) => {
    if (isEditableTarget(event.target)) return;
    const key = event.key.toLowerCase();
    if (["t", "s", "enter", "n", "p"].includes(key) && event.repeat) return;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      setFrame(currentFrame - (event.shiftKey ? 10 : 1));
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      setFrame(currentFrame + (event.shiftKey ? 10 : 1));
    } else if (event.code === "Space") {
      event.preventDefault();
      togglePlayback();
    } else if (key === "t") {
      event.preventDefault();
      markCurrent();
    } else if (key === "s") {
      event.preventDefault();
      saveAnnotation();
    } else if (event.key === "Enter") {
      event.preventDefault();
      saveAnnotation({ next: true });
    } else if (key === "n") {
      event.preventDefault();
      navigateTo(episode.episode_index + 1);
    } else if (key === "p") {
      event.preventDefault();
      navigateTo(episode.episode_index - 1);
    }
  });
}

async function initialize() {
  try {
    state = await api("/api/state");
    if (!state.cameras.length) throw new Error("Dataset has no camera/image observation keys");
    ui.dataset_name.textContent = state.dataset_name;
    ui.dataset_path.textContent = state.output_dataset;
    ui.episode_jump.max = state.total_episodes - 1;
    renderProgress();
    bindEvents();
    ui.startup.classList.add("hidden");
    ui.app.classList.remove("hidden");
    await loadEpisode(state.resume_episode, { skipDirtyCheck: true });
  } catch (error) {
    ui.startup.textContent = `Could not start annotator: ${error.message}`;
    ui.startup.style.color = "var(--danger)";
  }
}

initialize();
