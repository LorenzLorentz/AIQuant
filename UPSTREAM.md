# Upstream sources

This monorepo was assembled from two upstream projects on 2026-05-25.
If you ever need to pull upstream changes, the original git histories are
preserved in `<subdir>/.git.bak` and the anchor commits are recorded below.

## DeepMarket
- Upstream: https://github.com/LeonardoBerti00/DeepMarket
- Anchor commit: 8f1f89b (Update citation) on `main`
- Backup of original `.git` kept at `DeepMarket/.git.bak`

## lob_bench
- Upstream: https://github.com/peernagy/lob_bench
- Anchor commit: 276baf3 (Changes to impact to allow for different periods ...) on `main`
- Backup of original `.git` kept at `lob_bench/.git.bak`

## Re-establishing upstream
If you ever want to merge upstream updates back in:
```bash
# Inside the subdir, restore its old .git temporarily:
mv .git .git.AIQuant && mv .git.bak .git
git fetch origin
# cherry-pick / merge / diff as needed, then:
mv .git .git.bak && mv .git.AIQuant .git
```
