import { PluginConfig, PrismRequest, PrismResponse, SidecarHealth } from "./types.js";

function syntheticResponse(verdict: "ALLOW" | "BLOCK", reason: string): PrismResponse {
  return {
    verdict,
    confidence: verdict === "BLOCK" ? 1 : 0,
    reason,
    layer_triggered: "SidecarClient",
    normalized_text: "",
    ticket_id: null,
    placeholder:
      verdict === "BLOCK"
        ? "[PRISM_BLOCKED untrusted context removed before model assembly]"
        : null,
    audit: {
      path: "network_responses",
      source_type: "sidecar_client",
      degraded: true,
    },
  };
}

function buildHeaders(config: PluginConfig): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (config.secret) {
    headers["X-PRISM-Secret"] = config.secret;
  }
  return headers;
}

async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Inspect with failover: try primary sidecar (Python :8765), on failure
 * try fallback sidecar (Android :8766). Both share the same JSON schema.
 */
export async function inspect(
  request: PrismRequest,
  config: PluginConfig,
): Promise<PrismResponse> {
  const headers = buildHeaders(config);
  const body = JSON.stringify(request);
  const urls = [config.sidecarUrl];
  if (config.fallbackSidecarUrl) {
    urls.push(config.fallbackSidecarUrl);
  }

  for (const baseUrl of urls) {
    try {
      const response = await fetchWithTimeout(
        `${baseUrl}/v1/inspect`,
        { method: "POST", headers, body },
        config.timeoutMs,
      );
      if (!response.ok) {
        throw new Error(`sidecar_${response.status}`);
      }
      return (await response.json()) as PrismResponse;
    } catch {
      // Try next sidecar
    }
  }

  // All sidecars failed
  return config.failClosed
    ? syntheticResponse("BLOCK", "all_sidecars_unreachable")
    : syntheticResponse("ALLOW", "all_sidecars_unreachable_allow");
}

export async function getTicket(ticketId: string, config: PluginConfig): Promise<unknown | null> {
  const headers = buildHeaders(config);
  const urls = [config.sidecarUrl];
  if (config.fallbackSidecarUrl) {
    urls.push(config.fallbackSidecarUrl);
  }

  for (const baseUrl of urls) {
    try {
      const response = await fetchWithTimeout(
        `${baseUrl}/v1/ticket/${ticketId}`,
        { method: "GET", headers },
        config.timeoutMs,
      );
      if (response.ok) {
        return (await response.json()) as unknown;
      }
    } catch {
      // Try next sidecar
    }
  }
  return null;
}

export async function getHealth(config: PluginConfig): Promise<SidecarHealth | null> {
  const urls = [config.sidecarUrl];
  if (config.fallbackSidecarUrl) {
    urls.push(config.fallbackSidecarUrl);
  }

  for (const baseUrl of urls) {
    try {
      const response = await fetch(`${baseUrl}/health`);
      if (response.ok) {
        return (await response.json()) as SidecarHealth;
      }
    } catch {
      // Try next sidecar
    }
  }
  return null;
}
