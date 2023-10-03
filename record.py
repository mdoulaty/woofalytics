import logging
import time
import threading
import subprocess
import datetime

import pyaudio
import wave
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import requests

from pyargus.directionEstimation import (
    gen_ula_scanning_vectors,
    corr_matrix_estimate,
    DOA_Bartlett,
    DOA_Capon,
    DOA_MEM,
)

from event_filter import EventFilter

# Replace with your unique event name and IFTTT Webhooks API key
IFTTT_EVENT_NAME = "woof"
IFTTT_KEY = "YOUR_IFTTT_WEBHOOKS_KEY"

last_preds = []


class Woofalytics:
    def __init__(self, clip_past_context_seconds=15, clip_future_context_seconds=15):
        self._logger = logging.getLogger("Woofalytics")
        self._recording_device_index = self.find_andrea_mic_array()

        self._chunk = 441
        self._sample_format = pyaudio.paInt16  # 16 bits per sample
        self._channels = 2
        self._fs = 44_100
        self._model_sample_rate = 16_000

        self._clip_past_context_seconds = clip_past_context_seconds
        self._clip_future_context_seconds = clip_future_context_seconds

        self._store_flag = False
        self._stop_flag = False

        self._buffer = []

        self._worker_thread = None

        # self.set_mic_volume()

        self._model = torch.jit.load("./models/traced_model.pt")
        self._model.eval()
        self._model_window_size = 6
        self._model_window_overlap = 3
        self._model_last_pred = {
            "datetime": datetime.datetime.now().isoformat(),
            "bark_probability": [],
        }
        self._pred_lock = threading.Lock()

        self.ef = EventFilter()

        self._bark_prob_threshold = 0.88

        # DOA
        d = 0.1  # Inter element spacing [lambda]
        M = 2  # number of antenna elements in the antenna system (ULA)
        array_alignment = np.arange(0, M, 1) * d
        incident_angles = np.arange(0, 181, 1)
        self.ula_scanning_vectors = gen_ula_scanning_vectors(
            array_alignment, incident_angles
        )

    def find_andrea_mic_array(self) -> int:
        p = pyaudio.PyAudio()  # Create an interface to PortAudio
        # Get the list of input devices
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get("deviceCount")

        for i in range(0, numdevices):
            device_info = p.get_device_info_by_index(i)
            if device_info.get("maxInputChannels") > 0:
                name = device_info.get("name")
                self._logger.debug(f"Device index {i}: {name}")
                if name.startswith("Andrea PureAudio"):
                    self._logger.info(f"Found {name} at index {i}")
                    return i

        self._logger.warning(
            "Couldn't find 'Andrea PureAudio' recording device. Please make sure it's attached and functioning."
        )
        return -1

    def set_mic_volume(self, volume_percentage: int = 75):
        command = "amixer get Capture".split(" ")
        output = subprocess.check_output(command, text=True)
        self._logger.debug(output)

        command = f"amixer set Capture {volume_percentage}% unmute".split(" ")
        output = subprocess.check_output(command, text=True)
        self._logger.info(output)

        command = "amixer get Capture".split(" ")
        output = subprocess.check_output(command, text=True)
        self._logger.debug(output)

    def start(self):
        self._worker_thread = threading.Thread(target=self._recording_worker)
        self._worker_thread.start()

    def _recording_worker(self):
        past_frames_count = int(
            self._fs / self._chunk * self._clip_past_context_seconds * self._channels
        )
        future_frames_count = int(
            self._fs / self._chunk * self._clip_future_context_seconds * self._channels
        )

        self._logger.info("Starting recording loop...")
        self._logger.debug(
            f"Clip past context seconds: {self._clip_past_context_seconds}, number of frames: {past_frames_count}"
        )
        self._logger.debug(
            f"Clip future context seconds: {self._clip_future_context_seconds}, number of frames: {future_frames_count}"
        )

        p = pyaudio.PyAudio()
        stream = p.open(
            format=self._sample_format,
            channels=self._channels,
            rate=self._fs,
            frames_per_buffer=self._chunk,
            input=True,
            input_device_index=self._recording_device_index,
        )

        self._sample_size = p.get_sample_size(self._sample_format)

        record_buffer = []
        infer_buffer = []

        # how many samples for window length of 6?
        window_len_samples = int(self._fs * self._model_window_size / 1000.0)
        window_shift_samples = int(self._fs * self._model_window_overlap / 1000.0)
        self._logger.info(
            f"Window len #samples: {window_len_samples}, overlap #samples: {window_shift_samples}"
        )

        while not self._stop_flag:
            try:
                data = stream.read(self._chunk)
            except OSError as ex:
                self._logger.exception(ex)
                # Terminate the PortAudio interface
                p.terminate()

                p = pyaudio.PyAudio()
                stream = p.open(
                    format=self._sample_format,
                    channels=self._channels,
                    rate=self._fs,
                    frames_per_buffer=self._chunk,
                    input=True,
                    input_device_index=self._recording_device_index,
                )

                data = stream.read(self._chunk)

            record_buffer.append(data)

            # infer:
            infer_buffer.append(data)

            if (
                len(infer_buffer) >= 8
            ):  # each `data` is 0.01 seconds (441 samples, sampling rate 44100), we need 60ms (0.06 seconds) for a single window
                self.infer_chunk(infer_buffer.copy())
                infer_buffer = []

            # record:
            if not self._store_flag:  # we just keep past frames in buffer
                if (
                    len(record_buffer) > past_frames_count
                ):  # discard some earlier frames
                    record_buffer = record_buffer[-past_frames_count:]
            else:  # got a signal to store the frames
                if (
                    len(record_buffer) >= past_frames_count + future_frames_count
                ):  # have enought frames to dump to a file
                    self._dump_file(record_buffer.copy())
                    record_buffer = record_buffer[-past_frames_count:]

                    self._store_flag = False
                else:  # keep recording until the desired len is reached
                    pass

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()

    def stop(self):
        self._stop_flag = True
        if self._worker_thread:
            self._worker_thread.join()

    def store_clip(self):
        self._logger.info("Got a store request...")
        self._store_flag = True

    def _dump_file(self, frames):
        t = threading.Thread(target=self._dump_worker, args=[frames])
        t.start()

    def _dump_worker(self, frames):
        filename = f"{time.time_ns()}.wav"
        # Save the recorded data as a WAV file
        wf = wave.open(filename, "wb")
        wf.setnchannels(self._channels)
        wf.setsampwidth(self._sample_size)
        wf.setframerate(self._fs)
        wf.writeframes(b"".join(frames))
        self._logger.info(f"Stored {filename}")

    def get_last_pred(self):
        return self._model_last_pred

    def infer_chunk(self, frames):
        t = threading.Thread(target=self.infer_worker, args=[frames])
        t.start()

    def infer_worker(self, frames):
        audio_array = np.copy(np.frombuffer(b"".join(frames), dtype=np.int16))
        del frames
        audio_array = audio_array.reshape((2, -1), order="F")

        corr = corr_matrix_estimate(audio_array.T, imp="fast")
        doa1 = np.argmax(DOA_Bartlett(corr, self.ula_scanning_vectors))
        doa2 = np.argmax(DOA_Capon(corr, self.ula_scanning_vectors))
        doa3 = np.argmax(DOA_MEM(corr, self.ula_scanning_vectors))

        audio_array_torch = torch.from_numpy(audio_array)
        audio_array_float = audio_array_torch / torch.iinfo(torch.int16).max
        resampler = T.Resample(
            self._fs, self._model_sample_rate, dtype=audio_array_float.dtype
        )
        resampled_waveform = resampler(audio_array_float)
        mel_spectrogram = torchaudio.compliance.kaldi.fbank(
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            waveform=resampled_waveform,
        )
        mel_spectrogram = mel_spectrogram.flatten().unsqueeze(0)

        if mel_spectrogram.size()[1] != 480:
            self._logger.error("Wrong size for LMEL features", mel_spectrogram.size())
            return

        with torch.no_grad():
            pred = self._model(mel_spectrogram).detach().item()

        with self._pred_lock:
            if "bark_probability" not in self._model_last_pred:
                self._model_last_pred["bark_probability"] = [pred]
            else:
                while len(self._model_last_pred["bark_probability"]) > 16:
                    del self._model_last_pred["bark_probability"][0]
                self._model_last_pred["bark_probability"].append(pred)

            self._model_last_pred["datetime"] = datetime.datetime.now().isoformat()

        if pred >= self._bark_prob_threshold:
            print(
                f"[{datetime.datetime.now().isoformat()}, {doa1:03d}, {doa2:03d}, {doa3:03d}]: *** BARKING ***: {pred}"
            )
            with open("./log.txt", "a") as f:
                f.write(
                    f"{datetime.datetime.now().isoformat()}\t{pred}\t{doa1}\t{doa2}\t{doa3}\n"
                )
            last_preds.append(1)
            if len(last_preds) >= 6:
                del last_preds[0]
            if sum(last_preds) >= 3:
                if self.ef.fire():
                    self.ifttt_event()
                    self.store_clip()
        else:
            if len(last_preds) > 0:
                del last_preds[0]
                print(
                    f"[{datetime.datetime.now().isoformat()}, {doa1:03d}, {doa2:03d}, {doa3:03d}]: Not barking: {pred}\r",
                    end="",
                )


    def ifttt_event(self):
        # URL for the Maker Webhooks API endpoint
        ifttt_url = f"https://maker.ifttt.com/trigger/{IFTTT_EVENT_NAME}/with/key/{IFTTT_KEY}"

        # Send the HTTP POST request to trigger the IFTTT applet
        response = requests.post(ifttt_url)

        # Check the response
        if response.status_code == 200:
            self._logger.info("IFTTT applet triggered successfully.")
        else:
            self._logger.warning("Failed to trigger the IFTTT applet.")
