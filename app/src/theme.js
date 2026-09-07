import { Platform } from 'react-native';

/**
 * Grounded in EEG chart paper: the gridded fanfold stock clinical traces are
 * printed on. Pale mineral-green ground, printed grid rule, ink-dark trace,
 * and a single red reserved for the event marker.
 */
export const color = {
  paper: '#E9EBE4',
  paperEdge: '#DFE3DA',
  grid: '#C9D1C4',
  gridMajor: '#B4BFAE',
  ink: '#16221C',
  mute: '#6B7A6E',
  mark: '#B4331F',
};

export const space = [0, 4, 8, 12, 16, 24, 32, 48];

// Modular scale, ~1.33.
export const size = {
  fine: 11,
  small: 13,
  body: 15,
  large: 20,
  display: 27,
};

export const font = {
  sans: Platform.select({ ios: 'System', android: 'sans-serif', default: 'System' }),
  // Reserved for measured values, where tabular figures are functional.
  mono: Platform.select({ ios: 'Menlo', android: 'monospace', default: 'monospace' }),
};
