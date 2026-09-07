import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { color, font, size, space } from '../theme';

/**
 * One classified head, set as a row on the record rather than a card: name in
 * the margin, value in the measure, confidence as a filled rule beneath.
 */
export default function Reading({ head, label, confidence }) {
  const filled = Math.max(0, Math.min(1, confidence ?? 0));

  return (
    <View style={styles.row}>
      <Text style={styles.head}>{head}</Text>

      <View style={styles.measure}>
        <View style={styles.line}>
          <Text
            style={[styles.value, !label && styles.valueUnread]}
            numberOfLines={1}
          >
            {label ?? 'unread'}
          </Text>
          <Text style={styles.number}>
            {label ? filled.toFixed(2) : '—'}
          </Text>
        </View>

        <View style={styles.track}>
          <View style={[styles.fill, { width: `${filled * 100}%` }]} />
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingVertical: space[4],
    borderTopWidth: StyleSheet.hairlineWidth,
    borderColor: color.grid,
  },
  head: {
    width: 78,
    fontFamily: font.sans,
    fontSize: size.small,
    color: color.mute,
    paddingTop: 2,
  },
  measure: { flex: 1 },
  line: {
    flexDirection: 'row',
    alignItems: 'baseline',
    justifyContent: 'space-between',
  },
  value: {
    flex: 1,
    fontFamily: font.sans,
    fontSize: size.large,
    fontWeight: '600',
    letterSpacing: -0.3,
    color: color.ink,
  },
  valueUnread: { fontWeight: '400', color: color.mute },
  number: {
    fontFamily: font.mono,
    fontSize: size.small,
    color: color.mute,
    marginLeft: space[3],
  },
  track: {
    height: 3,
    backgroundColor: color.grid,
    marginTop: space[2],
  },
  fill: { height: 3, backgroundColor: color.ink },
});
