Show active runtime and estimated time remaining on Jellyfin job cards. Runtime
excludes queue waits, pauses, review, and worker downtime. Earlier jobs recover
an approximate runtime from logs. Estimates learn each stage and requested voice
separately, including QA rounds, and show a stage estimate while later stages
are still learning. Final assembly is identified separately. Saved clocks and
completed work survive pause/resume and idle updates.
