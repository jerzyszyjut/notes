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

# Folder names on the site (the vault keeps its own names).
declare -A SITE_DIR=(
  [00-Dashboard]=Dashboard [01-Dziennik]=Journal [02-Zycie]=Life
  [Listy_zakupow]="Shopping lists" [03-Uczelnia]=University
  [04-Wieden]=Vienna [05-Niemiecki]=German [Ćwiczenia]=Exercises
  [Gramatyka]=Grammar [Lekcje]=Lessons [Materiały]=Materials [strony]=Pages
)

site_path() {
  local out="" seg
  IFS=/ read -ra segs <<< "$(dirname "$1")"
  for seg in "${segs[@]}"; do
    [[ "$seg" == "." ]] && continue
    out+="${SITE_DIR[$seg]:-$seg}/"
  done
  echo "$out$(basename "$1")"
}

published=()
for n in "${notes[@]}"; do
  is_published "$VAULT/$n" && published+=("${n#./}")
done

echo "Notes to publish (${#published[@]}):"
for n in "${published[@]}"; do echo "  $n  ->  $(site_path "$n")"; done
[[ "${1:-}" == "--dry-run" ]] && exit 0

# Rebuild content/ from scratch (keeping the homepage) so unpublished notes disappear.
find "$CONTENT" -mindepth 1 -not -name index.md -delete
for n in "${published[@]}"; do
  dest="$CONTENT/$(site_path "$n")"
  mkdir -p "$(dirname "$dest")"
  cp "$VAULT/$n" "$dest"
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
