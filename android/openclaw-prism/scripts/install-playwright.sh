#!/usr/bin/env bash
# install-playwright.sh - Install Playwright for browser automation
# Usage: bash install-playwright.sh [install|update]
#
# What it does:
#   1. Ensure Chromium is installed (dependency)
#   2. Install playwright-core via npm global
#   3. Set Playwright environment variables in .bashrc
#   4. Print usage guide
#
# This script is WARN-level: failure does not abort the parent installer.
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

MODE="${1:-install}"

# ── Helper ────────────────────────────────────

fail_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    exit 0
}

# ── Detect Chromium binary path ───────────────

detect_chromium_bin() {
    for bin in "$PREFIX/bin/chromium-browser" "$PREFIX/bin/chromium"; do
        if [ -x "$bin" ]; then
            echo "$bin"
            return 0
        fi
    done
    return 1
}

# ── Pre-checks ────────────────────────────────

if [ -z "${PREFIX:-}" ]; then
    fail_warn "Not running in Termux (\$PREFIX not set)"
fi

if ! command -v npm &>/dev/null; then
    fail_warn "npm not found — Node.js is required for Playwright"
fi

# ── Step 1: Ensure Chromium is installed ──────

if ! CHROMIUM_BIN=$(detect_chromium_bin); then
    echo "Chromium is required for Playwright. Installing..."
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    if [ -f "$SCRIPT_DIR/install-chromium.sh" ]; then
        if ! bash "$SCRIPT_DIR/install-chromium.sh" install; then
            fail_warn "Chromium installation failed — cannot proceed with Playwright"
        fi
    else
        fail_warn "install-chromium.sh not found — install Chromium first"
    fi

    if ! CHROMIUM_BIN=$(detect_chromium_bin); then
        fail_warn "Chromium binary not found after installation"
    fi
fi

echo -e "${GREEN}[OK]${NC}   Chromium found: $CHROMIUM_BIN"

# ── Step 2: Install playwright-core ───────────

if [ "$MODE" = "install" ]; then
    if npm list -g playwright-core &>/dev/null; then
        echo -e "${GREEN}[SKIP]${NC} playwright-core already installed"
    else
        echo "Installing playwright-core..."
        if ! npm install -g playwright-core; then
            fail_warn "Failed to install playwright-core"
        fi
        echo -e "${GREEN}[OK]${NC}   playwright-core installed"
    fi
elif [ "$MODE" = "update" ]; then
    echo "Updating playwright-core..."
    if ! npm install -g playwright-core@latest; then
        fail_warn "Failed to update playwright-core"
    fi
    echo -e "${GREEN}[OK]${NC}   playwright-core updated"
fi

# ── Step 3: Set environment variables ─────────

BASHRC="$HOME/.bashrc"
PW_MARKER_START="# >>> Playwright >>>"
PW_MARKER_END="# <<< Playwright <<<"

PW_BLOCK="${PW_MARKER_START}
export PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=\"$CHROMIUM_BIN\"
export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
${PW_MARKER_END}"

touch "$BASHRC"
if grep -qF "$PW_MARKER_START" "$BASHRC"; then
    sed -i "/${PW_MARKER_START//\//\\/}/,/${PW_MARKER_END//\//\\/}/d" "$BASHRC"
fi
echo "" >> "$BASHRC"
echo "$PW_BLOCK" >> "$BASHRC"

echo -e "${GREEN}[OK]${NC}   Environment variables set in .bashrc"
echo "       PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=$CHROMIUM_BIN"
echo "       PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1"

# ── Step 4: Usage guide ──────────────────────

echo ""
echo -e "${BOLD}  Playwright is ready!${NC}"
echo ""
echo "  To use in your project:"
echo ""
echo "    npm install playwright-core    # add to your project"
echo ""
echo "  Example code:"
echo ""
echo "    const { chromium } = require('playwright-core');"
echo ""
echo "    const browser = await chromium.launch();"
echo "    const page = await browser.newPage();"
echo "    await page.goto('https://example.com');"
echo "    await page.screenshot({ path: 'screenshot.png' });"
echo "    await browser.close();"
echo ""
echo -e "  ${YELLOW}[NOTE]${NC} Environment variables are set. No need to specify"
echo "         executablePath or --no-sandbox manually."
echo ""
echo "  To apply environment variables in current session:"
echo "    source ~/.bashrc"
echo ""
