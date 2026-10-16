import { useMemo } from 'react'
import { buildViewGraph, type LayoutMetrics, type ViewGraph } from '../../core/layout/viewGraph'
import { collapsedPredicate, useModelStore } from '../../store/modelStore'

/** The visible graph laid out for the 3D scene, or for other metrics (the flat diagram); null skips the work. */
export function useViewGraph(): ViewGraph
export function useViewGraph(metrics: LayoutMetrics | null): ViewGraph | null
export function useViewGraph(metrics?: LayoutMetrics | null): ViewGraph | null {
  const model = useModelStore((state) => state.modelIR)
  const depth = useModelStore((state) => state.depth)
  const toggles = useModelStore((state) => state.toggles)
  return useMemo(() => metrics === null ? null : buildViewGraph(model, collapsedPredicate({ modelIR: model, depth, toggles }), metrics), [model, depth, toggles, metrics])
}

export const tr = (locale: string, en: string, zh: string) => locale === 'zh' ? zh : en

export { STUB_LENGTH, edgeCurve } from '../../core/layout/edgeCurve'
