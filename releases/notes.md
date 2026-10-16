Fix a CTranslate2 native crash while generating Whisper word timings for a
window with fewer than two usable mel frames. The guard preserves normal word
timings and the existing CUDA/cuDNN dependencies.

The Jellyfin dashboard now has a Refresh log button below each expanded job log.
Queue dub appears disabled until a library video and an available director are
selected. Plugin and worker updates continue to install together only when idle.
