import type { OpenClawPluginApi } from "openclaw/plugin-sdk/core";
import { PrismContextEngine } from "./context_engine.js";
import { registerAfterToolCallHook } from "./hooks/after_tool_call.js";
import { registerBeforeToolCallHook } from "./hooks/before_tool_call.js";
import { registerMessageReceivedHook } from "./hooks/message_received.js";
import type { PluginConfig } from "./types.js";

export const defaultConfig: PluginConfig = {
  sidecarUrl: "http://127.0.0.1:8765",
  fallbackSidecarUrl: "http://127.0.0.1:8766",
  timeoutMs: 500,
  failClosed: true,
  quarantineMode: "exclude",
  protectSources: {
    messages: true,
    toolResults: true,
    memory: true,
    androidUi: true,
  },
};

const pluginConfigSchema = {
  type: "object",
  additionalProperties: false,
  properties: {
    sidecarUrl: { type: "string", default: defaultConfig.sidecarUrl },
    fallbackSidecarUrl: { type: "string", default: defaultConfig.fallbackSidecarUrl },
    timeoutMs: { type: "integer", default: defaultConfig.timeoutMs, minimum: 1 },
    failClosed: { type: "boolean", default: true },
    secret: { type: "string" },
    quarantineMode: { type: "string", enum: ["exclude", "placeholder"], default: "exclude" },
    protectSources: {
      type: "object",
      additionalProperties: false,
      properties: {
        messages: { type: "boolean", default: true },
        toolResults: { type: "boolean", default: true },
        memory: { type: "boolean", default: true },
        androidUi: { type: "boolean", default: true },
      },
    },
  },
};

function resolvePluginConfig(raw?: Record<string, unknown>): PluginConfig {
  const protectSources =
    raw?.protectSources && typeof raw.protectSources === "object"
      ? (raw.protectSources as Record<string, unknown>)
      : {};

  return {
    sidecarUrl:
      typeof raw?.sidecarUrl === "string" && raw.sidecarUrl.trim()
        ? raw.sidecarUrl
        : defaultConfig.sidecarUrl,
    fallbackSidecarUrl:
      typeof raw?.fallbackSidecarUrl === "string" && raw.fallbackSidecarUrl.trim()
        ? raw.fallbackSidecarUrl
        : defaultConfig.fallbackSidecarUrl,
    timeoutMs:
      typeof raw?.timeoutMs === "number" && Number.isFinite(raw.timeoutMs) && raw.timeoutMs > 0
        ? Math.floor(raw.timeoutMs)
        : defaultConfig.timeoutMs,
    failClosed: raw?.failClosed !== false,
    secret: typeof raw?.secret === "string" ? raw.secret : undefined,
    quarantineMode: raw?.quarantineMode === "placeholder" ? "placeholder" : "exclude",
    protectSources: {
      messages: protectSources.messages !== false,
      toolResults: protectSources.toolResults !== false,
      memory: protectSources.memory !== false,
      androidUi: protectSources.androidUi !== false,
    },
  };
}

const plugin = {
  id: "openclaw-prism",
  name: "OpenClaw PRISM",
  description: "PRISM Shield context defense for OpenClaw",
  version: "0.1.0",
  configSchema: pluginConfigSchema,
  register(api: OpenClawPluginApi) {
    const config = resolvePluginConfig(api.pluginConfig);

    api.registerContextEngine("prism-context", () => new PrismContextEngine(config, api.logger));
    registerBeforeToolCallHook(api, config);
    registerMessageReceivedHook(api, config);
    registerAfterToolCallHook(api, config);
  },
};

export default plugin;
