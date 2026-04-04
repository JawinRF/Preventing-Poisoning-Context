/**
 * JsBridge wrapper — typed interface to window.OpenClaw (§2.6).
 * All Kotlin @JavascriptInterface methods return JSON strings.
 */

interface OpenClawBridge {
  showTerminal(): void
  showWebView(): void
  createSession(): string
  switchSession(id: string): void
  closeSession(id: string): void
  getTerminalSessions(): string
  writeToTerminal(id: string, data: string): void
  getSetupStatus(): string
  getBootstrapStatus(): string
  startSetup(): void
  saveToolSelections(json: string): void
  getAvailablePlatforms(): string
  getInstalledPlatforms(): string
  installPlatform(id: string): void
  uninstallPlatform(id: string): void
  switchPlatform(id: string): void
  getActivePlatform(): string
  getInstalledTools(): string
  installTool(id: string): void
  uninstallTool(id: string): void
  isToolInstalled(id: string): string
  runCommand(cmd: string): string
  runCommandAsync(callbackId: string, cmd: string): void
  checkForUpdates(): string
  applyUpdate(component: string): void
  getApkUpdateInfo(): string
  getAppInfo(): string
  getBatteryOptimizationStatus(): string
  requestBatteryOptimizationExclusion(): void
  openSystemSettings(page: string): void
  copyToClipboard(text: string): void
  getStorageInfo(): string
  clearCache(): void
  openUrl(url: string): void

  // PRISM Security
  getSecurityStatus(): string
  getAuditFeed(): string
  getSidecarHealth(): string
  getPermissionStatus(): string
}

// PRISM Security types
export interface SecurityStatus {
  blocked: number
  allowed: number
  total: number
  sidecarPort: number
}

export interface AuditEntry {
  id: number
  path: string
  snippet: string
  verdict: string
  layer1Score: number
  layer2Prob: number
  matchedRules: string
  timestamp: number
}

export interface SidecarHealth {
  python_sidecar: 'up' | 'down'
  android_sidecar: 'up' | 'down'
}

export interface PermissionStatus {
  sms: boolean
  contacts: boolean
  calendar: boolean
}

declare global {
  interface Window {
    OpenClaw?: OpenClawBridge
    __oc?: { emit(type: string, data: unknown): void }
  }
}

export function isAvailable(): boolean {
  return typeof window.OpenClaw !== 'undefined'
}

export function call<K extends keyof OpenClawBridge>(
  method: K,
  ...args: Parameters<OpenClawBridge[K]>
): ReturnType<OpenClawBridge[K]> | null {
  if (window.OpenClaw && typeof window.OpenClaw[method] === 'function') {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (window.OpenClaw[method] as (...a: any[]) => any)(...args)
  }
  console.warn('[bridge] OpenClaw not available:', method)
  return null
}

export function callJson<T>(
  method: keyof OpenClawBridge,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ...args: any[]
): T | null {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raw = (call as any)(method, ...args)
  if (raw == null) return null
  try {
    return JSON.parse(raw as string) as T
  } catch {
    return raw as unknown as T
  }
}

export const bridge = { isAvailable, call, callJson }
