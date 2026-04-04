export type Verdict = "ALLOW" | "BLOCK" | "QUARANTINE";

export interface PrismRequest {
  entry_id: string;
  text: string;
  ingestion_path: string;
  source_type: string;
  source_name: string;
  session_id: string;
  run_id: string;
  metadata: Record<string, unknown>;
}

export interface PrismResponse {
  verdict: Verdict;
  confidence: number;
  reason: string;
  layer_triggered: string;
  normalized_text: string;
  ticket_id: string | null;
  placeholder: string | null;
  audit: Record<string, unknown>;
}

export interface SidecarHealth {
  status: string;
  uptime?: number;
  version?: string;
}

export interface PluginConfig {
  sidecarUrl: string;
  fallbackSidecarUrl: string;
  timeoutMs: number;
  failClosed: boolean;
  secret?: string;
  quarantineMode: "exclude" | "placeholder";
  protectSources: {
    messages: boolean;
    toolResults: boolean;
    memory: boolean;
    androidUi: boolean;
  };
}

