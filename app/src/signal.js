/**
 * Simulated EEG, generated on the device.
 *
 * There is no electrode hardware behind this app. The buffer is a sum of
 * band-limited sinusoids plus noise -- the same shape the training data takes
 * -- so the trace and the classify action have something real to operate on.
 * The interface says so rather than implying a live patient.
 */

const BANDS = [
  { low: 0.5, high: 4, amp: 1.1 },
  { low: 4, high: 8, amp: 0.7 },
  { low: 8, high: 13, amp: 1.0 },
  { low: 13, high: 30, amp: 0.5 },
  { low: 30, high: 45, amp: 0.2 },
];

function channelPhases(channels) {
  return Array.from({ length: channels }, () =>
    BANDS.map((band) => ({
      frequency: band.low + Math.random() * (band.high - band.low),
      phase: Math.random() * Math.PI * 2,
      amplitude: band.amp * (0.7 + Math.random() * 0.6),
    })),
  );
}

/** A generator producing successive frames of a continuous signal. */
export function createSignal(channels, samplingRate) {
  const components = channelPhases(channels);
  let cursor = 0;

  return {
    /** Next `count` samples, shaped [samples][channels]. */
    next(count) {
      const frame = [];
      for (let i = 0; i < count; i += 1) {
        const t = (cursor + i) / samplingRate;
        frame.push(
          components.map((parts) =>
            parts.reduce(
              (sum, { frequency, phase, amplitude }) =>
                sum + amplitude * Math.sin(2 * Math.PI * frequency * t + phase),
              (Math.random() - 0.5) * 0.2,
            ),
          ),
        );
      }
      cursor += count;
      return frame;
    },
  };
}
