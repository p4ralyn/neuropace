import React, { useMemo, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import Svg, { Line, Polyline } from 'react-native-svg';

import { color, font, size, space } from '../theme';

const GRID = 16; // px per minor division, as on printed chart stock

/**
 * Multi-channel trace drawn on chart paper.
 *
 * Purely presentational: it takes a frame shaped [samples][channels] and the
 * names of the channels it should draw.
 */
export default function EEGTrace({ frame, names, height = 260 }) {
  // Measure the container: an Svg sized only by its viewBox reports that
  // viewBox as its intrinsic width on web and drags the page wider than the
  // viewport.
  const [measured, setMeasured] = useState(0);
  const width = 1000; // viewBox units
  const lanes = names.length;
  const laneHeight = height / lanes;

  const grid = useMemo(() => {
    const lines = [];
    for (let x = 0; x <= width; x += GRID) {
      lines.push({ key: `v${x}`, x1: x, y1: 0, x2: x, y2: height, major: x % (GRID * 5) === 0 });
    }
    for (let y = 0; y <= height; y += GRID) {
      lines.push({ key: `h${y}`, x1: 0, y1: y, x2: width, y2: y, major: y % (GRID * 5) === 0 });
    }
    return lines;
  }, [height]);

  const traces = useMemo(() => {
    if (!frame?.length) return [];
    const step = width / (frame.length - 1 || 1);

    return names.map((_, channel) => {
      const values = frame.map((row) => row[channel] ?? 0);
      const peak = Math.max(...values.map(Math.abs), 1e-6);
      const centre = laneHeight * channel + laneHeight / 2;
      const scale = (laneHeight * 0.38) / peak;

      return values
        .map((value, i) => `${(i * step).toFixed(1)},${(centre - value * scale).toFixed(1)}`)
        .join(' ');
    });
  }, [frame, names, laneHeight]);

  return (
    <View
      style={[styles.paper, { height }]}
      onLayout={(event) => setMeasured(event.nativeEvent.layout.width)}
    >
      <Svg
        width={measured || '100%'}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
      >
        {grid.map(({ key, major, ...line }) => (
          <Line
            {...line}
            key={key}
            stroke={major ? color.gridMajor : color.grid}
            strokeWidth={major ? 1.2 : 0.6}
          />
        ))}
        {traces.map((points, i) => (
          <Polyline
            key={names[i]}
            points={points}
            fill="none"
            stroke={color.ink}
            strokeWidth={1.4}
            strokeLinejoin="round"
          />
        ))}
      </Svg>

      <View style={styles.margin} pointerEvents="none">
        {names.map((name) => (
          <View key={name} style={[styles.lane, { height: laneHeight }]}>
            <Text style={styles.channel}>{name}</Text>
          </View>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  paper: {
    width: '100%',
    overflow: 'hidden',
    backgroundColor: color.paper,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderColor: color.gridMajor,
  },
  margin: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'flex-start',
  },
  lane: { justifyContent: 'center', paddingLeft: space[3] },
  channel: {
    fontFamily: font.mono,
    fontSize: size.fine,
    color: color.mute,
  },
});
