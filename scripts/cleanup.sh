#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$BASE_DIR"
echo "Cleaning up intermediate files..."
# Remove llamafactory_data directory only
if [ -d "llamafactory_data" ]; then
    rm -rf llamafactory_data
    echo "  Removed llamafactory_data/"
fi
echo "Cleanup complete. Data, adapters, and results are preserved."
