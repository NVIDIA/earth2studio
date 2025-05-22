#!/bin/bash

# Get the current version using uv and hatch
HATCH_VERSION=$(uv run --only-group dev hatch version)
VERSION=$(echo ${HATCH_VERSION} | sed 's/rc0$//')

PREV_VERSION=$(grep -oP '(?<=earth2studio\.git@)[0-9]+\.[0-9]+\.[0-9]+' docs/userguide/about/install.md | head -n 1)

if [ -z "$PREV_VERSION" ]; then
  echo "Error: Could not extract previous version from install.md."
  exit 1
fi

echo "Current version (from hatch): ${VERSION}"
echo "Previous version (from install.md): ${PREV_VERSION}"

sed -i "s/earth2studio\.git@${PREV_VERSION}/earth2studio\.git@${VERSION}/g" docs/userguide/about/install.md
sed -i "s/${PREV_VERSION}/${VERSION}/g" recipes/template/pyproject.toml
sed -i "s/${PREV_VERSION}/${VERSION}/g" recipes/template/README.md
sed -i "s/${PREV_VERSION}/${VERSION}/g" .github/ISSUE_TEMPLATE/bug_report.yml

echo "Replaced all occurrences of ${PREV_VERSION} with ${VERSION}"
