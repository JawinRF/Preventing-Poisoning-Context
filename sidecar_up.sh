#!/bin/bash
export ANDROID_SDK_ROOT=/home/jrf/Android
export ANDROID_HOME=/home/jrf/Android

PKG=com.openclaw.android.debug
A11Y=${PKG}/com.openclaw.android.security.PrismAccessibilityService
NL=${PKG}/com.openclaw.android.security.PrismNotificationListener

echo "[1] port forward"
adb -s emulator-5554 forward tcp:8766 tcp:8766

echo "[2] launch app"
adb -s emulator-5554 shell am start -n ${PKG}/com.openclaw.android.MainActivity
sleep 4

echo "[3] enable accessibility"
adb -s emulator-5554 shell settings put secure enabled_accessibility_services ${A11Y}
adb -s emulator-5554 shell settings put secure accessibility_enabled 1

echo "[4] enable notification listener"
adb -s emulator-5554 shell settings put secure enabled_notification_listeners ${NL}
sleep 4

echo "[5] verify sidecar"
curl -s -m 12 http://127.0.0.1:8766/v1/context | head -c 300
echo
curl -s http://127.0.0.1:8766/v1/status
echo
echo "done"
