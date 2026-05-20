# oxideav-vorbis

Pure-Rust Vorbis I audio codec.

## Status — 2026-05-20

**Orphan-rebuild scaffold.** The crate's prior implementation was
retired under the workspace clean-room policy: provenance for several
core modules could not be defended against the "no external library
source as reference" rule that governs every crate in this workspace.

Per workspace policy, the only acceptable response is a full
clean-room re-implementation against the Vorbis I standards documents
and black-box validator binaries. That work has not yet been
scheduled.

Every public entry point currently returns `Error::NotImplemented`.

## Planned clean-room sources

The clean-room rebuild will consult only:

* Vorbis I Specification (xiph.org / IETF) — the authoritative format
  spec.
* RFC 5215 — RTP Payload Format for Vorbis (where the framing
  intersects).
* Black-box invocations of `oggdec` / `oggenc` (the binaries — not
  their source) as opaque validators.

No external library source — libvorbis, etc. — is permitted as a
reference under the workspace clean-room policy.

## License

MIT. See `LICENSE`.
