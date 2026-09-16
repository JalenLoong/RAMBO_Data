---
id: CURRENT-PUBLICATION-V2
type: current
status: accepted
source_map: []
---
# QLM-Bench publication policy

The user designated [dontKnow23456/QLM-Bench](https://huggingface.co/datasets/dontKnow23456/QLM-Bench)
as the continuing destination for authorized quadrupedal loco-manipulation synthesis batches.
The latest instruction limits uploads to **usable demonstrations only**: their Raw, Canonical,
provenance and replay. Pilots, failures and quarantined attempts remain local, including their
media/evidence payloads. This supersedes the initial full-archive proposal. No additional
collection, randomization, training or GitHub publication is implied.

Public visibility and `license: unknown` are preserved. The dataset card states that the data
license is pending and credits third-party assets. Asset source files, model weights, WAM
model-specific caches and training artifacts are not uploaded. This is scripted-expert data,
not manual teleoperation or learned-policy performance evidence.

## Ownership and layout

RAMBO owns publication policy/config, explicit allowlist preparation, immutable payload upload
and remote verification. WAM consumes a pinned Hub revision plus a Canonical subdirectory.
Neither imports the other repository. The underlying dataset contract is unchanged.

```text
README.md
releases.json
raw/v2/<task>/<collection>/<episode>/
canonical/lerobot_v2_1/<release>/
releases/<release>/{manifest.json,checksums.json,evidence/,review/}
```

Original bytes and historical local paths are preserved inside source manifests. The release
manifest provides separate portable paths, episode identity, split, source fingerprints and
source-to-Hub mapping. Flattening DATA-005's local `demonstrations/` grouping is recorded in that
mapping; no two attempts may share a destination. Observer is only Raw/review, never a model key.

## Publishing and consuming

`configs/qlm_bench_publication_v1.json` fixes the destination and exclusions.
`scripts/rambo/prepare_release_v2.py` requires an explicit eligible-episode allowlist, exact
Canonical membership/split and completed local acceptance; it refuses an existing staging tree.
`scripts/rambo/publish_dataset_v2.py` offers check/upload/verify/index operations over an explicit
inventory. Identical remote files are skipped on resume; differing content at an existing
payload path is an error. Commits use the observed remote parent to reject concurrent updates.

Upload the immutable payload first, compare every remote object against SHA256 (LFS) or Git blob
hash, and download Canonical at the resulting commit. Run existing contract/media checks plus
LeRobot/WAM readback without Raw or model caches. Only then update the root README/release index.
Keep previous indexed releases unchanged. An interrupted upload is not a completed release.
Record the final Hub SHA and verify the index at that SHA before reporting publication complete.

Authentication uses an existing HF login; credentials never appear in configs, logs or datasets.
Use the installed environment's CLI/API. Hub visibility is never changed by the publisher.

## First release

DATA-006-20260916T075451Z contains the original ten DATA-005 demonstrations and forty new usable
demonstrations. The original8/1/1 split is retained; additions32/4/4 give40/5/5. demo_11-a2 replaces
the locally quarantined a1; the successful authorized a4s for10/13/25 are included. Publication
evidence is tracked by [DATA-007](../work/archive/DATA-007/plan.md). The first release is now published and verified; see the receipt below.

## Published first release

QLM-Bench publication and fixed-revision readback passed. Final Hub revision `4b5e0ed9aa04cbc1143774798f63171b27d838a7`; payload revision `21b69e2d775cd77d6565e25d1b0b99ad725b54b3`. Exactly1616 payload files/2150407479 bytes,50 usable Raw/Canonical demonstrations, split40/5/5. All remote file hashes match;516 Canonical/release files were downloaded at the payload revision and independently validated through RAMBO and WAM/LeRobot0.3.3. Root README/index and exact remote membership were verified at the final revision; public/license unknown unchanged. No pilots/failures/quarantine/model-cache/asset binaries uploaded. Publication and source handoff receipts: workspace runs/audit/v2/DATA-007. No training or remote jobs.
