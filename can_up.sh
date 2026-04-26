#!/bin/bash
# 兼容入口：保留旧命令 bash can_up.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[Notice] can_up.sh is deprecated. Use: bash interface_up.sh [8chips|4chips|crab|none]"
exec bash "$SCRIPT_DIR/interface_up.sh" none
