#!/usr/bin/env sh

set -e

npm install --save-dev @commitlint/{cli,config-conventional}

echo "export default { extends: ['@commitlint/config-conventional'] };" > commitlint.config.js

npm install --save-dev husky

npx husky init

# Add commit message linting to commit-msg hook
echo "npx --no -- commitlint --edit \$1" > .husky/commit-msg

# Windows users should use ` to escape dollar signs
# echo "npx --no commitlint --edit `$1" > .husky/commit-msg
