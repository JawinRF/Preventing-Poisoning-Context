import { useState, useEffect } from 'react'
import { bridge } from '../lib/bridge'
import { t } from '../i18n'

interface BootstrapStatus {
  installed: boolean
  prefixPath?: string
}

interface PlatformInfo {
  id: string
  name: string
}

export function Dashboard() {
  const [status, setStatus] = useState<BootstrapStatus | null>(null)
  const [platform, setPlatform] = useState<PlatformInfo | null>(null)
  const [runtimeInfo, setRuntimeInfo] = useState<Record<string, string>>({})

  useEffect(() => {
    const bs = bridge.callJson<BootstrapStatus>('getBootstrapStatus')
    if (bs) setStatus(bs)

    const ap = bridge.callJson<PlatformInfo>('getActivePlatform')
    if (ap) setPlatform(ap)

    const nodeV = bridge.callJson<{ stdout: string }>('runCommand', 'node -v 2>/dev/null')
    const ocV   = bridge.callJson<{ stdout: string }>('runCommand', 'openclaw --version 2>/dev/null')
    setRuntimeInfo({
      'Node.js':   nodeV?.stdout?.trim() || '—',
      'openclaw':  ocV?.stdout?.trim()   || '—',
    })
  }, [])

  if (!status?.installed) {
    return (
      <div className="page">
        <div className="setup-container" style={{ minHeight: 'calc(100vh - 80px)' }}>
          <div className="setup-title">{t('dash_setup_required')}</div>
          <div className="setup-subtitle">{t('dash_setup_desc')}</div>
        </div>
      </div>
    )
  }

  return (
    <div className="page">
      <div style={{ marginBottom: 28 }}>
        <div style={{ fontSize: 12, fontWeight: 600, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--text-secondary)', marginBottom: 6 }}>
          Agent
        </div>
        <div style={{ fontSize: 30, fontWeight: 700, lineHeight: 1 }}>
          {platform?.name || 'OpenClaw'}
        </div>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="info-row" style={{ borderBottom: '1px solid var(--border)' }}>
          <span className="label">Status</span>
          <span style={{ color: 'var(--success)', fontWeight: 600 }}>Running</span>
        </div>
        <div className="info-row" style={{ borderBottom: '1px solid var(--border)' }}>
          <span className="label">Node.js</span>
          <span>{runtimeInfo['Node.js']}</span>
        </div>
        <div className="info-row">
          <span className="label">openclaw</span>
          <span>{runtimeInfo['openclaw']}</span>
        </div>
      </div>
    </div>
  )
}
