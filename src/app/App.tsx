import { useEffect } from 'react'
import { AppShell } from '../components/layout/AppShell'
import { openSharedLink } from '../components/layout/projectActions'

export default function App() {
  useEffect(() => {
    void openSharedLink()
    const onHash = () => { void openSharedLink() }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  return <AppShell />
}
