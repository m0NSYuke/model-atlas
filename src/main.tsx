import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './app/App'
import './styles/index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)

// Development only: stores on window for end-to-end checks.
if (import.meta.env.DEV) {
  void Promise.all([import('./store/modelStore'), import('./store/studyStore'), import('./store/uiStore'), import('./components/layout/projectActions')]).then(([model, study, ui, actions]) => {
    Object.assign(window, { __atlas: { model: model.useModelStore, study: study.useStudyStore, ui: ui.useUiStore, share: actions.createShareLink } })
  })
}
