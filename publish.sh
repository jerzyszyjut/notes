#!/usr/bin/env bash
# Copy every vault note with `publish: true` in its frontmatter into content/,
# then commit and push so GitHub Actions rebuilds the site.
# Usage: ./publish.sh            (sync + push)
#        ./publish.sh --dry-run  (only list what would be published)
set -euo pipefail

VAULT="$HOME/Obsidian/vault"
QUARTZ="$(cd "$(dirname "$0")" && pwd)"
CONTENT="$QUARTZ/content"

is_published() {
  # Only look inside the leading frontmatter block.
  awk 'NR==1 && $0!="---" {exit 1}
       NR>1 && $0=="---" {exit found ? 0 : 1}
       NR>1 && /^publish:[[:space:]]*true[[:space:]]*$/ {found=1}
       END {exit found ? 0 : 1}' "$1"
}

mapfile -d '' notes < <(cd "$VAULT" && find . -name '*.md' -not -path './.obsidian/*' -not -path './Templates/*' -print0)

published=()
for n in "${notes[@]}"; do
  is_published "$VAULT/$n" && published+=("${n#./}")
done

echo "Notes to publish (${#published[@]}):"
printf '  %s\n' "${published[@]}"
[[ "${1:-}" == "--dry-run" ]] && exit 0

# Rebuild content/ from scratch (keeping the homepage) so unpublished notes disappear.
find "$CONTENT" -mindepth 1 -not -name index.md -delete
for n in "${published[@]}"; do
  mkdir -p "$CONTENT/$(dirname "$n")"
  cp "$VAULT/$n" "$CONTENT/$n"
done

cd "$QUARTZ"
git add -A content
if git diff --cached --quiet; then
  echo "Nothing changed."
else
  git commit -q -m "Publish notes $(date +%F)"
  git push -q
  echo "Pushed. The site updates in a minute or two."
fi
