Move optional worker, director, language-setup, installer, and release tooling
into scripts/integrations/. Direct dubs continue to use scripts/dub_anything.py.
Update documentation, tests, installation commands, and release automation for
the layout. Bundled compatibility launchers let older installed updaters upgrade
and roll back safely; settings, queue, and working files keep their paths.

Includes the CTranslate2 short-window crash guard, Refresh log control, and
disabled Queue dub styling introduced in the preceding patch releases.
