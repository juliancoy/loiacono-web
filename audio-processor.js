// AudioWorkletProcessor for streaming audio data
class AudioStreamProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.port.onmessage = (event) => {
      // Handle messages from main thread if needed
    };
  }

  process(inputs, outputs, parameters) {
    // We only care about the first input (mono)
    const input = inputs[0];
    if (input && input.length > 0) {
      const channelData = input[0];
      // Send audio data to main thread
      this.port.postMessage({
        type: 'audioData',
        data: channelData.slice() // Copy the data
      });
    }
    return true; // Keep processor alive
  }
}

registerProcessor('audio-stream-processor', AudioStreamProcessor);
