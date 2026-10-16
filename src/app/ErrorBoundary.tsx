import { Component, type ErrorInfo, type ReactNode } from 'react'

// Last line of defence: a render error shows this page instead of an empty screen. The reset button
// clears the current view and session (the usual culprits when saved state is bad); "My projects" and
// notes are kept.
const sessionKeys = ['atlas-ui-v1', 'model-atlas-project-v2']

export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(error, info.componentStack)
  }

  private reset = () => {
    try { sessionKeys.forEach((key) => window.localStorage.removeItem(key)) } catch { /* storage may be unavailable */ }
    window.location.replace(window.location.pathname)
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children
    return (
      <div role="alert" style={{ minHeight: '100vh', display: 'grid', placeItems: 'center', padding: 24, background: 'var(--app-bg)', color: 'var(--text)', fontFamily: 'var(--sans)' }}>
        <div style={{ maxWidth: 520, lineHeight: 1.6 }}>
          <h1 style={{ fontSize: 20, margin: '0 0 8px' }}>出了点问题 · Something went wrong</h1>
          <p style={{ margin: '0 0 16px', color: 'var(--muted)' }}>
            页面遇到错误，没能显示出来。可以先重新加载；如果仍然出错，就重置界面状态，"我的项目"和笔记会保留。
            <br />The page hit an error. Try reloading; if it happens again, reset the view. Your projects and notes are kept.
          </p>
          <div className="dialog-actions" style={{ justifyContent: 'flex-start', flexWrap: 'wrap', marginBottom: 16 }}>
            <button type="button" className="primary" onClick={() => window.location.reload()}>重新加载 · Reload</button>
            <button type="button" onClick={this.reset}>重置界面 · Reset view</button>
          </div>
          <details style={{ color: 'var(--muted)', fontSize: 12 }}>
            <summary>错误详情 · Details</summary>
            <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'var(--mono)' }}>{String(error.stack ?? error.message)}</pre>
          </details>
        </div>
      </div>
    )
  }
}
