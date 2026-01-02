import argparse
import os
import sys
import time

import cv2
import librosa
import numpy as np
import tqdm

# === Setup Import Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..')))
import vulkanese as ve  # Uses Vulkanese-based Loiacono transform
import vulkan as vk

sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', 'sinode')))


def parse_args():
    parser = argparse.ArgumentParser(description='Render Loiacono spectrograms with Vulkan-based GPU DSP')
    parser.add_argument('audio_file', help='Path to the input WAV/MP3 file')
    parser.add_argument('--realtime', action='store_true', help='Show a live OpenCV preview window')
    parser.add_argument('--fps', type=int, default=60, help='Frames-per-second for the output video')
    parser.add_argument('--buffer-frames', type=int, default=120, help='Temporal buffer depth shown in the video')
    parser.add_argument('--num-bins', type=int, default=512, help='Frequency bins in the spectrum')
    return parser.parse_args()


def create_loiacono(device, fprime, multiple, dtftlen):
    return ve.math.signals.loiacono_gpu.Loiacono_GPU(
        device=device,
        parent=device,
        fprime=fprime.astype(np.float32),
        multiple=multiple,
        signalLength=dtftlen,
        DEBUG=False,
    )


def main() -> None:
    args = parse_args()
    audio_file = args.audio_file
    fps = args.fps
    buffer_frames = args.buffer_frames
    num_bins = args.num_bins

    y, sr = librosa.load(audio_file, sr=None)
    frame_duration = 1.0 / fps
    hop_size = int(frame_duration * sr)

    multiple = 15
    minfreq = 100
    maxfreq = 12000
    fprime = np.logspace(np.log10(minfreq / sr), np.log10(maxfreq / sr), num_bins)
    max_samples_per_period = multiple * sr / minfreq
    dtftlen = 2 ** int(np.ceil(np.log2(max_samples_per_period)))
    print(f'Using FFT size of {dtftlen} samples (original hop size: {hop_size})')

    instance = ve.instance.Instance(verbose=False)
    device = instance.getDevice(0)
    loiacono = create_loiacono(device, fprime, multiple, dtftlen)

    video_size = (buffer_frames * 6, int(num_bins * 0.25))
    video_file = audio_file.replace('.wav', '.mp4').replace('.mp3', '.mp4')

    buffer = np.zeros((buffer_frames, num_bins), dtype=np.float32)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_file, fourcc, fps, video_size, isColor=True)

    bg_color = (0x61, 0x2A, 0x00)
    bg_color = tuple(reversed(bg_color))
    bg_color_np = np.array(bg_color, dtype=np.float32)
    white_color = np.array([255, 255, 255], dtype=np.float32)
    red_color = np.array([0, 0, 255], dtype=np.float32)

    realtime = args.realtime
    if realtime:
        cv2.namedWindow('Spectral Peaks', cv2.WINDOW_NORMAL)
        screen_width = cv2.getWindowImageRect('Spectral Peaks')[2]
    else:
        screen_width = 0

    middle_start = buffer_frames // 2 - buffer_frames // 20
    middle_end = buffer_frames // 2 + buffer_frames // 20

    num_frames = len(y) // hop_size
    print(f'Processing {num_frames} frames with Loiacono transform...')

    try:
        for i in tqdm.tqdm(range(num_frames)):
            start = i * hop_size
            end = start + dtftlen
            frame = y[start:end] if end <= len(y) else np.zeros(dtftlen)
            spectrum = loiacono.run(frame, blocking=True)

            mean_val = np.mean(spectrum)
            std_val = np.std(spectrum)
            threshold = mean_val + 1.5 * std_val

            peak_range = np.max(spectrum) - threshold
            if peak_range <= 0:
                normalized = np.zeros_like(spectrum)
            else:
                normalized = np.clip((spectrum - threshold) / peak_range, 0, 1)

            buffer = np.roll(buffer, -1, axis=0)
            buffer[-1] = normalized

            buffer_3d = np.repeat(buffer[:, :, np.newaxis], 3, axis=2)
            img = bg_color_np * (1 - buffer_3d) + white_color * buffer_3d
            img[middle_start:middle_end, :] = (
                bg_color_np * (1 - buffer_3d[middle_start:middle_end, :]) +
                red_color * buffer_3d[middle_start:middle_end, :]
            )
            img = img.astype(np.uint8)
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = cv2.resize(img, video_size, interpolation=cv2.INTER_NEAREST)

            writer.write(img)

            if realtime:
                if screen_width > 0:
                    cv2.moveWindow('Spectral Peaks', (screen_width - num_bins) // 2, 100)
                cv2.imshow('Spectral Peaks', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('Interrupted by user.')
                    break
    finally:
        writer.release()
        cv2.destroyAllWindows()
        loiacono.release()
        device.release()
        instance.release()

    print(f'Video written to {video_file}')


if __name__ == '__main__':
    main()
