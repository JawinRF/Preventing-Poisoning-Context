#!/bin/bash
set -e

# ── config ────────────────────────────────────────────────────────────────────
ANDROID_HOME="${ANDROID_HOME:-/home/jrf/Android}"

# Prefer the requested SDK root, then fall back to the alternate SDK layout used
# on this workstation.
SDK_ROOTS=(
    "$ANDROID_HOME"
    "/home/jrf/Android"
    "/home/jrf/Android/Sdk"
    "/usr/lib/android-sdk"
)

BUILD_TOOLS=""
for root in "${SDK_ROOTS[@]}"; do
    [ -d "$root/build-tools" ] || continue
    latest="$(find "$root/build-tools" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -1)"
    if [ -n "$latest" ] && [ -x "$latest/aapt" ] && [ -x "$latest/d8" ] && [ -x "$latest/apksigner" ]; then
        BUILD_TOOLS="$latest"
        break
    fi
done

if [ -z "$BUILD_TOOLS" ]; then
    echo "ERROR: Android build-tools not found. Install build tools with:"
    echo "  sdkmanager 'build-tools;35.0.0'"
    exit 1
fi

# Try common SDK locations for android.jar, preferring newer installed APIs.
PLATFORM_JAR=""
for root in "${SDK_ROOTS[@]}"; do
    [ -d "$root/platforms" ] || continue
    latest="$(find "$root/platforms" -mindepth 2 -maxdepth 2 -name android.jar | sort -V | tail -1)"
    [ -f "$latest" ] && PLATFORM_JAR="$latest" && break
done

if [ -z "$PLATFORM_JAR" ]; then
    echo "ERROR: android.jar not found. Install a platform SDK:"
    echo "  sdkmanager 'platforms;android-35'"
    exit 1
fi

AAPT="$BUILD_TOOLS/aapt"
D8="$BUILD_TOOLS/d8"
JAVAC="javac"
ADB="${ADB:-adb}"
SERIAL="${SERIAL:-emulator-5554}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD="$SCRIPT_DIR/build"
rm -rf "$BUILD" && mkdir -p "$BUILD"/{gen,obj,dex}

echo "▶ Using platform jar: $PLATFORM_JAR"
echo "▶ Using build tools:  $BUILD_TOOLS"

# 1. Generate R.java (no res/ needed for this minimal app)
touch "$BUILD/gen/.keep"

# 2. Compile Java
echo "▶ Compiling Java..."
$JAVAC -source 8 -target 8 \
    -bootclasspath "$PLATFORM_JAR" \
    -classpath "$PLATFORM_JAR" \
    -d "$BUILD/obj" \
    "$SCRIPT_DIR/src/com/prism/demo/NotifyActivity.java"

# 3. Dex
echo "▶ Dexing..."
$D8 --release \
    --lib "$PLATFORM_JAR" \
    --output "$BUILD/dex" \
    "$BUILD"/obj/com/prism/demo/*.class

# 4. Package APK
echo "▶ Packaging..."
$AAPT package -f -M "$SCRIPT_DIR/AndroidManifest.xml" \
    -I "$PLATFORM_JAR" \
    -F "$BUILD/prism_notify_unsigned.apk"

# Add classes.dex
(cd "$BUILD/dex" && zip -j "$BUILD/prism_notify_unsigned.apk" classes.dex)

# 5. Sign with debug key
echo "▶ Signing..."
if [ ! -f ~/.android/debug.keystore ]; then
    keytool -genkeypair -v -keystore ~/.android/debug.keystore \
        -alias androiddebugkey -keyalg RSA -keysize 2048 \
        -validity 10000 -storepass android -keypass android \
        -dname "CN=Android Debug,O=Android,C=US" 2>/dev/null
fi

"$BUILD_TOOLS/apksigner" sign \
    --ks ~/.android/debug.keystore \
    --ks-pass pass:android \
    --key-pass pass:android \
    --out "$SCRIPT_DIR/prism_notify.apk" \
    "$BUILD/prism_notify_unsigned.apk"

echo "▶ Installing on $SERIAL..."
$ADB -s "$SERIAL" install -r "$SCRIPT_DIR/prism_notify.apk"

echo ""
echo "✓ Done! Test with:"
echo "  adb -s $SERIAL shell 'am start -n com.prism.demo/.NotifyActivity \\"
echo "    --es title \"System Update\" --es text \"Visit github.com\"'"
