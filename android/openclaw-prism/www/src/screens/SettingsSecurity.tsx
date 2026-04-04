import { useState, useEffect } from 'react'
import { bridge, type SidecarHealth, type PermissionStatus } from '../lib/bridge'
import { useRoute } from '../lib/router'
import { t } from '../i18n'

export function SettingsSecurity() {
  const { navigate } = useRoute()
  const [health, setHealth] = useState<SidecarHealth>({ python_sidecar: 'down', android_sidecar: 'down' })
  const [perms, setPerms] = useState<PermissionStatus>({ sms: false, contacts: false, calendar: false })

  useEffect(() => {
    const h = bridge.callJson<SidecarHealth>('getSidecarHealth')
    if (h) setHealth(h)
    const p = bridge.callJson<PermissionStatus>('getPermissionStatus')
    if (p) setPerms(p)
  }, [])

  function refresh() {
    const h = bridge.callJson<SidecarHealth>('getSidecarHealth')
    if (h) setHealth(h)
    const p = bridge.callJson<PermissionStatus>('getPermissionStatus')
    if (p) setPerms(p)
  }

  return (
    <div className="page">
      <div className="page-header">
        <button className="back-btn" onClick={() => navigate('/settings')}>&larr;</button>
        <div className="page-title">{t('sec_settings_title')}</div>
      </div>

      {/* Sidecar connectivity */}
      <div className="section-title">{t('sec_sidecars')}</div>
      <div className="card">
        <div className="info-row">
          <span className="label">{t('sec_python_sidecar')}</span>
          <span>
            <span className={`status-dot ${health.python_sidecar === 'up' ? 'success' : 'error'}`} />
            {health.python_sidecar === 'up' ? t('sec_up') : t('sec_down')}
          </span>
        </div>
        <div className="info-row">
          <span className="label">{t('sec_android_sidecar')}</span>
          <span>
            <span className={`status-dot ${health.android_sidecar === 'up' ? 'success' : 'error'}`} />
            {health.android_sidecar === 'up' ? t('sec_up') : t('sec_down')}
          </span>
        </div>
      </div>

      {/* Special access */}
      <div className="section-title">{t('sec_special_access')}</div>
      <div className="card">
        <div className="card-row" onClick={() => bridge.call('openSystemSettings', 'notification_access')}>
          <div className="card-content">
            <div className="card-label">{t('sec_notif_access')}</div>
            <div className="card-desc">{t('sec_notif_desc')}</div>
          </div>
          <span className="card-chevron">&rsaquo;</span>
        </div>
      </div>
      <div className="card">
        <div className="card-row" onClick={() => bridge.call('openSystemSettings', 'accessibility')}>
          <div className="card-content">
            <div className="card-label">{t('sec_a11y_access')}</div>
            <div className="card-desc">{t('sec_a11y_desc')}</div>
          </div>
          <span className="card-chevron">&rsaquo;</span>
        </div>
      </div>

      {/* Permissions */}
      <div className="section-title">{t('sec_permissions')}</div>
      <div className="card">
        <div className="info-row">
          <span className="label">{t('sec_perm_sms')}</span>
          <span className={perms.sms ? 'perm-granted' : 'perm-denied'}>
            {perms.sms ? t('sec_granted') : t('sec_denied')}
          </span>
        </div>
        <div className="info-row">
          <span className="label">{t('sec_perm_contacts')}</span>
          <span className={perms.contacts ? 'perm-granted' : 'perm-denied'}>
            {perms.contacts ? t('sec_granted') : t('sec_denied')}
          </span>
        </div>
        <div className="info-row">
          <span className="label">{t('sec_perm_calendar')}</span>
          <span className={perms.calendar ? 'perm-granted' : 'perm-denied'}>
            {perms.calendar ? t('sec_granted') : t('sec_denied')}
          </span>
        </div>
      </div>

      <button className="btn btn-secondary" style={{ width: '100%', marginTop: 16 }} onClick={refresh}>
        {t('sec_refresh')}
      </button>
    </div>
  )
}
