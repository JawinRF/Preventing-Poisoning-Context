import { useState, useEffect } from 'react'
import { bridge, type SecurityStatus, type AuditEntry } from '../lib/bridge'
import { t } from '../i18n'

export function SecurityDashboard() {
  const [status, setStatus] = useState<SecurityStatus>({ blocked: 0, allowed: 0, total: 0, sidecarPort: 8766 })
  const [feed, setFeed] = useState<AuditEntry[]>([])

  useEffect(() => {
    function refresh() {
      const s = bridge.callJson<SecurityStatus>('getSecurityStatus')
      if (s) setStatus(s)
      const f = bridge.callJson<AuditEntry[]>('getAuditFeed')
      if (f) setFeed(f)
    }
    refresh()
    const interval = setInterval(refresh, 2000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="page">
      {/* Header */}
      <div className="security-header">
        <div className="security-title">{t('sec_title')}</div>
        <div className="status-pill active">{t('sec_active')}</div>
      </div>
      <div className="security-subtitle">
        localhost:{status.sidecarPort} &bull; {t('sec_on_device')}
      </div>

      {/* Stat cards */}
      <div className="stat-row">
        <div className="stat-card stat-blocked">
          <div className="stat-value">{status.blocked}</div>
          <div className="stat-label">{t('sec_blocked')}</div>
        </div>
        <div className="stat-card stat-allowed">
          <div className="stat-value">{status.allowed}</div>
          <div className="stat-label">{t('sec_allowed')}</div>
        </div>
        <div className="stat-card stat-total">
          <div className="stat-value">{status.total}</div>
          <div className="stat-label">{t('sec_total')}</div>
        </div>
      </div>

      {/* Live threat feed */}
      <div className="section-title">{t('sec_live_feed')}</div>

      {feed.length === 0 ? (
        <div className="feed-empty">{t('sec_no_events')}</div>
      ) : (
        <div className="feed-list">
          {feed.map((entry) => (
            <LogRow key={entry.id} entry={entry} />
          ))}
        </div>
      )}
    </div>
  )
}

function LogRow({ entry }: { entry: AuditEntry }) {
  const isBlock = entry.verdict === 'BLOCK'
  const icon = isBlock ? '\u2717' : '\u2713'
  const time = new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })

  return (
    <div className={`log-row ${isBlock ? 'log-block' : 'log-allow'}`}>
      <span className="log-icon">{icon}</span>
      <div className="log-content">
        <div className="log-meta">
          <span className={`log-path-badge ${isBlock ? 'badge-block' : 'badge-allow'}`}>
            {entry.path}
          </span>
          <span className="log-time">{time}</span>
        </div>
        <div className="log-snippet">{entry.snippet.slice(0, 90)}</div>
        {isBlock && (
          <div className="log-scores">
            <span className="score-badge score-l1">L1={entry.layer1Score.toFixed(2)}</span>
            {entry.layer2Prob > 0 && (
              <span className="score-badge score-l2">L2={entry.layer2Prob.toFixed(2)}</span>
            )}
            {entry.matchedRules && (
              <span className="score-badge score-rule">{entry.matchedRules.split(',')[0]}</span>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
