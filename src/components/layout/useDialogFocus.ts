import { useEffect, useRef, type RefObject } from 'react'

const focusable = 'a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

/**
 * Keyboard behaviour of a modal dialog: focus moves into it when it opens, Tab and Shift+Tab cycle inside it,
 * Escape closes it, and focus goes back to whatever opened it once it closes.
 */
export function useDialogFocus(ref: RefObject<HTMLElement>, onClose: () => void) {
  const close = useRef(onClose)
  close.current = onClose
  useEffect(() => {
    const dialog = ref.current
    if (!dialog) return
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const items = () => [...dialog.querySelectorAll<HTMLElement>(focusable)].filter((element) => element.getClientRects().length > 0)
    if (!dialog.contains(document.activeElement)) dialog.focus({ preventScroll: true })
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        if (event.defaultPrevented) return
        event.preventDefault()
        close.current()
        return
      }
      if (event.key !== 'Tab') return
      const list = items()
      if (!list.length) { event.preventDefault(); return }
      const active = document.activeElement
      const inside = dialog.contains(active) && active !== dialog
      if (event.shiftKey && (!inside || active === list[0])) { event.preventDefault(); list[list.length - 1].focus() }
      else if (!event.shiftKey && (!inside || active === list[list.length - 1])) { event.preventDefault(); list[0].focus() }
    }
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      if (opener?.isConnected) opener.focus({ preventScroll: true })
    }
  }, [ref])
}
