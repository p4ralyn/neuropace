import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Pressable,
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  View,
} from 'react-native';

import EEGTrace from './src/components/EEGTrace';
import Reading from './src/components/Reading';
import Status from './src/components/Status';
import { API_BASE, getHealth, postPredict } from './src/api';
import { createSignal } from './src/signal';
import { color, font, size, space } from './src/theme';

const HEADS = ['ailment', 'stress', 'mood'];
const VISIBLE_CHANNELS = ['Fp1', 'F3', 'Cz', 'P4', 'O2'];
const TRACE_SAMPLES = 220; // what we draw, not what we send
const FRAME_MS = 120;

export default function App() {
  const [health, setHealth] = useState(null);
  const [connection, setConnection] = useState('checking');
  const [frame, setFrame] = useState([]);
  const [readings, setReadings] = useState(null);
  const [problem, setProblem] = useState(null);
  const [reading, setReading] = useState(false);

  const shape = health?.expected_shape ?? [1250, 19];
  const [windowSamples, channels] = shape;

  const signal = useMemo(
    () => createSignal(channels, Math.round(windowSamples / 5)),
    [channels, windowSamples],
  );
  const buffer = useRef([]);

  useEffect(() => {
    let cancelled = false;
    getHealth().then((result) => {
      if (cancelled) return;
      if (result.ok) {
        setHealth(result.body);
        setConnection(result.body.model_loaded ? 'ready' : 'no-model');
      } else {
        setConnection('unreachable');
        setProblem(result.detail);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const timer = setInterval(() => {
      const next = signal.next(12);
      buffer.current = [...buffer.current, ...next].slice(-TRACE_SAMPLES);
      setFrame(buffer.current);
    }, FRAME_MS);
    return () => clearInterval(timer);
  }, [signal]);

  const read = useCallback(async () => {
    setReading(true);
    setProblem(null);

    const window = createSignal(channels, Math.round(windowSamples / 5)).next(
      windowSamples,
    );
    const result = await postPredict(window);

    if (result.ok) {
      setReadings(result.body);
      setConnection('ready');
    } else {
      setReadings(null);
      setProblem(result.detail);
      setConnection(result.status === 503 ? 'no-model' : 'unreachable');
    }
    setReading(false);
  }, [channels, windowSamples]);

  const statusLine = {
    checking: 'Checking the service',
    ready: `Connected to ${API_BASE}`,
    'no-model': 'Service running, no trained model',
    unreachable: 'Service unreachable',
  }[connection];

  return (
    <SafeAreaView style={styles.screen}>
      <StatusBar barStyle="dark-content" backgroundColor={color.paper} />

      <View style={styles.header}>
        <Text style={styles.wordmark}>NeuroPace</Text>
        <Status tone={connection === 'ready' ? 'ink' : 'mark'}>{statusLine}</Status>
      </View>

      <EEGTrace frame={frame} names={VISIBLE_CHANNELS} height={260} />

      <View style={styles.caption}>
        <Text style={styles.captionText}>
          Simulated signal, {channels} channels at {Math.round(windowSamples / 5)} Hz.
          Showing {VISIBLE_CHANNELS.length}.
        </Text>
      </View>

      <ScrollView contentContainerStyle={styles.body}>
        {HEADS.map((head) => (
          <Reading
            key={head}
            head={head}
            label={readings?.[head]?.label}
            confidence={readings?.[head]?.confidence}
          />
        ))}

        {problem ? (
          <View style={styles.problem}>
            <Text style={styles.problemText}>{problem}</Text>
          </View>
        ) : (
          !readings && (
            <Text style={styles.empty}>
              No reading yet. Capture a window to classify it.
            </Text>
          )
        )}
      </ScrollView>

      <Pressable
        onPress={read}
        disabled={reading}
        style={({ pressed }) => [styles.action, pressed && styles.actionPressed]}
        accessibilityRole="button"
        accessibilityLabel="Read this window"
      >
        {reading ? (
          <ActivityIndicator color={color.paper} />
        ) : (
          <Text style={styles.actionText}>Read this window</Text>
        )}
      </Pressable>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, overflow: 'hidden', backgroundColor: color.paperEdge },
  header: {
    paddingHorizontal: space[5],
    paddingTop: space[4],
    paddingBottom: space[4],
    gap: space[2],
  },
  wordmark: {
    fontFamily: font.sans,
    fontSize: size.display,
    fontWeight: '700',
    letterSpacing: -0.8,
    color: color.ink,
  },
  caption: {
    paddingHorizontal: space[5],
    paddingVertical: space[3],
  },
  captionText: {
    fontFamily: font.sans,
    fontSize: size.fine,
    color: color.mute,
  },
  body: { paddingHorizontal: space[5], paddingBottom: space[5] },
  empty: {
    fontFamily: font.sans,
    fontSize: size.small,
    color: color.mute,
    paddingTop: space[5],
    borderTopWidth: StyleSheet.hairlineWidth,
    borderColor: color.grid,
  },
  problem: {
    marginTop: space[5],
    paddingTop: space[4],
    borderTopWidth: 2,
    borderColor: color.mark,
  },
  problemText: {
    fontFamily: font.sans,
    fontSize: size.small,
    lineHeight: 21,
    color: color.ink,
  },
  action: {
    margin: space[5],
    paddingVertical: space[4],
    alignItems: 'center',
    backgroundColor: color.mark,
  },
  actionPressed: { opacity: 0.85 },
  actionText: {
    fontFamily: font.sans,
    fontSize: size.body,
    fontWeight: '600',
    color: color.paper,
  },
});
