# BUGS — what broke, and what caught it

| # | Phase | What broke | What caught it | Fix |
|---|---|---|---|---|
| 1 | 0 | The MPS model rate kept rising from run to run (185 → 233 → 288 fps). One warm-up batch was not enough, so the first runs were timing GPU warm-up rather than steady state. | The timed runs climbed steadily instead of scattering around a mean. | The warm-up is now one full untimed pass over all batches. The later runs scatter at 206–255 fps with no trend. |
| 2 | 1 | The corpus download aborted entirely when the Internet Archive returned HTTP 500/504 for one title. One exception in the thread pool stopped every download. | The background download exited with code 1 partway through. | Each title now fails independently. curl retries server errors (`--retry-all-errors`, 15 s delay), resumes partial files, and the script lists the failed titles at the end. Re-running resumes the download. |
| 3 | 1 | Python HTTPS requests to archive.org failed certificate verification, although `curl` worked. Something on the network path re-signs TLS certificates, and Python checks against its own bundle rather than the OS store. | `CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain` when building the manifest. | Use `truststore`, which makes Python verify against the macOS certificate store. Verification was never switched off. |
