import { useRoute } from '../lib/router'
import { t } from '../i18n'

interface MenuItem {
  icon: string
  label: string
  desc: string
  route: string
  badge?: boolean
}

function getMenu(): MenuItem[] {
  return [
    { icon: 'PL', label: t('settings_platforms'), desc: t('settings_platforms_desc'), route: '/settings/platforms' },
    { icon: 'UP', label: t('settings_updates'), desc: t('settings_updates_desc'), route: '/settings/updates', badge: false },
    { icon: 'KA', label: t('settings_keep_alive'), desc: t('settings_keep_alive_desc'), route: '/settings/keep-alive' },
    { icon: 'SC', label: t('settings_security'), desc: t('settings_security_desc'), route: '/settings/security' },
    { icon: 'ST', label: t('settings_storage'), desc: t('settings_storage_desc'), route: '/settings/storage' },
    { icon: 'AB', label: t('settings_about'), desc: t('settings_about_desc'), route: '/settings/about' },
  ]
}

export function Settings() {
  const { navigate } = useRoute()

  return (
    <div className="page">
      <div className="page-title" style={{ marginBottom: 24 }}>{t('settings_title')}</div>
      {getMenu().map(item => (
        <div key={item.route} className="card" onClick={() => navigate(item.route)}>
          <div className="card-row">
            <span className="card-icon">{item.icon}</span>
            <div className="card-content">
              <div className="card-label">{item.label}</div>
              <div className="card-desc">{item.desc}</div>
            </div>
            {item.badge && <span className="card-badge" />}
            <span className="card-chevron">›</span>
          </div>
        </div>
      ))}
    </div>
  )
}
