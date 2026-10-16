# Optional media-server integrations

The worker, headless directing adapters, language setup, and release tooling
live here. A normal request to dub a local video uses `scripts/dub_anything.py`;
it does not need this package or Jellyfin installed.

From the repository root, with VoiceStudio's Python:

```powershell
& $py scripts/integrations service serve
& $py scripts/integrations service status
& $py scripts/integrations service stop
& $py scripts/integrations updates once --install work/service/install.json
& $py scripts/integrations build-release
```

Each component accepts `--help`. The directory launcher also works when called
by absolute path from another working directory. The worker keeps the existing
`work/service/` queue, settings, and job progress.

Windows installation is handled by `setup-jellyfin.ps1` in this directory;
follow [docs/jellyfin.md](../../docs/jellyfin.md) for the complete setup.
`director.py` and `language_setup.py` are imported helpers, not extra services.
The shared dubbing tools remain in the parent `scripts/` directory.

Published runtime bundles also contain tiny launchers at the former
`scripts/dub_service.py` and `scripts/dub_updates.py` paths. They delegate to this
package so older installed updaters keep working. These compatibility files are
generated during packaging and do not clutter the source checkout's script root.
