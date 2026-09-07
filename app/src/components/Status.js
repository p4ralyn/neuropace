import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { color, font, size, space } from '../theme';

/** Connection state, stated plainly. */
export default function Status({ tone, children }) {
  return (
    <View style={styles.row}>
      <View style={[styles.dot, tone === 'mark' && styles.dotMark]} />
      <Text style={styles.text}>{children}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: 'row', alignItems: 'center' },
  dot: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: color.ink,
    marginRight: space[2],
  },
  dotMark: { backgroundColor: color.mark },
  text: {
    fontFamily: font.sans,
    fontSize: size.small,
    color: color.mute,
  },
});
