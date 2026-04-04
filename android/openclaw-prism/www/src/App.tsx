import { useState, useEffect, useCallback } from 'react'
import { Route, useRoute } from './lib/router'
import { bridge } from './lib/bridge'
import { useNativeEvent } from './lib/useNativeEvent'
import { t } from './i18n'
import { Setup } from './screens/Setup'
import { Dashboard } from './screens/Dashboard'
import { SecurityDashboard } from './screens/SecurityDashboard'
import { Settings } from './screens/Settings'
import { SettingsKeepAlive } from './screens/SettingsKeepAlive'
import { SettingsStorage } from './screens/SettingsStorage'
import { SettingsAbout } from './screens/SettingsAbout'
import { SettingsUpdates } from './screens/SettingsUpdates'
import { SettingsPlatforms } from './screens/SettingsPlatforms'
import { SettingsSecurity } from './screens/SettingsSecurity'

type Tab = 'terminal' | 'dashboard' | 'security' | 'settings'

export function App() {
  const { path, navigate } = useRoute()
  const [hasUpdates, setHasUpdates] = useState(false)

  // Check setup status on mount
  const [setupDone, setSetupDone] = useState<boolean | null>(null)

  useEffect(() => {
    const status = bridge.callJson<{ bootstrapInstalled?: boolean; platformInstalled?: string }>(
      'getSetupStatus'
    )
    if (status) {
      setSetupDone(!!status.bootstrapInstalled && !!status.platformInstalled)
    } else {
      // Bridge not available (dev mode) — assume setup done
      setSetupDone(true)
    }

    // Check for updates
    const updates = bridge.callJson<unknown[]>('checkForUpdates')
    if (updates && updates.length > 0) setHasUpdates(true)
  }, [])

  const onUpdateAvailable = useCallback(() => {
    setHasUpdates(true)
  }, [])
  useNativeEvent('update_available', onUpdateAvailable)

  // Determine active tab from path
  const activeTab: Tab = path.startsWith('/security')
    ? 'security'
    : path.startsWith('/settings')
      ? 'settings'
      : path.startsWith('/setup')
        ? 'settings'
        : 'dashboard'

  function handleTabClick(tab: Tab) {
    if (tab === 'terminal') {
      bridge.call('showTerminal')
      return
    }
    bridge.call('showWebView')
    if (tab === 'dashboard') navigate('/dashboard')
    if (tab === 'security') navigate('/security')
    if (tab === 'settings') navigate('/settings')
  }

  // Show setup flow if not completed (Security and Dashboard are always accessible)
  if (setupDone === null) return null // loading
  if (!setupDone && !path.startsWith('/setup') && !path.startsWith('/security') && !path.startsWith('/dashboard')) {
    navigate('/setup')
  }

  return (
    <>
      {/* Tab bar */}
      <nav className="tab-bar">
        <button
          className="tab-bar-item"
          onClick={() => handleTabClick('terminal')}
        >
          {t('tab_terminal')}
        </button>
        <button
          className={`tab-bar-item ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => handleTabClick('dashboard')}
        >
          {t('tab_dashboard')}
        </button>
        <button
          className={`tab-bar-item ${activeTab === 'security' ? 'active' : ''}`}
          onClick={() => handleTabClick('security')}
        >
          {t('tab_security')}
        </button>
        <button
          className={`tab-bar-item ${activeTab === 'settings' ? 'active' : ''}`}
          onClick={() => handleTabClick('settings')}
        >
          {t('tab_settings')}
          {hasUpdates && <span className="badge" />}
        </button>
      </nav>

      {/* Routes */}
      <Route path="/setup">
        <Setup onComplete={() => { setSetupDone(true); navigate('/dashboard') }} />
      </Route>
      <Route path="/dashboard">
        <Dashboard />
      </Route>
      <Route path="/security">
        <SecurityDashboard />
      </Route>
      <Route path="/settings">
        <SettingsRouter />
      </Route>
    </>
  )
}

function SettingsRouter() {
  const { path } = useRoute()
  if (path === '/settings') return <Settings />
  if (path === '/settings/keep-alive') return <SettingsKeepAlive />
  if (path === '/settings/storage') return <SettingsStorage />
  if (path === '/settings/about') return <SettingsAbout />
  if (path === '/settings/updates') return <SettingsUpdates />
  if (path === '/settings/platforms') return <SettingsPlatforms />
  if (path === '/settings/security') return <SettingsSecurity />
  return <Settings />
}
