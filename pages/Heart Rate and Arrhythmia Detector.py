import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import threading
from collections import deque
import time

#Bandpass filter
def bandpass_filter(signal, fs=30, low=0.7, high=3.0, order=5):
    nyquist = 0.5 * fs
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    return filtfilt(b, a, signal)

#Arrhythmia detection
def detect_arrhythmia(rr_intervals):
    """
    This detects arrhythmia based on interval variability between heartbeats.
    It returns rhythym_status and confidence of prediction.
    """
    if len(rr_intervals) < 5:  # At least 5 intervals needed for analysis to ensure reliability 
        return "Insufficient data for rhythm analysis", "low"
    
    #Variability metrics
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)
    cv = std_rr / mean_rr
    #Successive differences
    successive_diffs = np.abs(np.diff(rr_intervals))
    mean_diff = np.mean(successive_diffs)
    #Thresholding
    if cv > 0.15 or mean_diff > 0.15:
        return "Irregular rhythm detected", "medium"
    elif cv > 0.25 or mean_diff > 0.25:
        return "Highly irregular rhythm detected", "high"
    else:
        return "Regular rhythm", "medium"

#Video processor 
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.buffer = deque(maxlen=30 * 10)  # 10 sec buffer
        self.fps = 30
        self.lock = threading.Lock()
        self.rr_intervals_history = []  # Store intervals for analysis at end

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_intensity = gray.mean()
        
        with self.lock:
            self.buffer.append(mean_intensity)
            
        return frame

#Streamlit interface
st.title("Heart Rate and Arrhythmia Detector")
st.markdown(
"""This tool uses your camera to detect your heart rate, along with any arrhythmia, through a technique called photoplethysmography (PPG).
PPG works by analyzing the tiny changes in blood volume in your fingertip as your heart pumps blood.
When you place your finger on the camera, the app can measure these changes.
**Turn on your camera flash and ensure that you are in a well-lit area.
For accurate readings, steadily hold the tip of your index finger completely on the camera, applying gentle but firm pressure.**""")

#Initialize session state
if 'signal_data' not in st.session_state:
    st.session_state.signal_data = []
if 'peaks_data' not in st.session_state:
    st.session_state.peaks_data = []
if 'heart_rate' not in st.session_state:
    st.session_state.heart_rate = None
if 'rr_intervals_history' not in st.session_state:
    st.session_state.rr_intervals_history = []
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if "start_time" not in st.session_state:
    st.session_state.start_time = None

ctx = webrtc_streamer(key="ppg", video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False}, mode=WebRtcMode.SENDRECV)

plot_placeholder = st.empty()
status_placeholder = st.empty()

#Continuous updates
if ctx.video_processor:
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()
    
    while True:
        with ctx.video_processor.lock:
            buf = list(ctx.video_processor.buffer) # Get buffer
        
        fps = ctx.video_processor.fps
        
        if len(buf) > fps * 3:  # Need at least 3 sec of data
            try:
                #Apply bandpass filter
                signal = bandpass_filter(np.array(buf), fs=fps)
                
                #Filter raw waveform for peaks with parameters; these select peaks are then used to compute HR
                peaks, _ = find_peaks(signal, 
                                    distance=int(fps * 0.3),  # Peaks must be at least 0.4 seconds apart
                                    height=np.mean(signal) + 0.2 * np.std(signal), # Peaks must have enough amplitude
                                    prominence=0.2) # Peaks must have prominence compared to neighbors
                
                if len(peaks) > 2:  # Need more than 2 peaks for RR intervals
                    rr_intervals = np.diff(peaks) / fps
                    #Remove outlier RR intervals
                    valid_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]
                    
                    if len(valid_intervals) > 1:
                        heart_rate = 60.0 / np.mean(valid_intervals)
                        st.session_state.heart_rate = heart_rate
                        status_placeholder.success(f"Estimated heart rate: {heart_rate:.1f} bpm")
                        
                        #Store intervals for final analysis
                        st.session_state.rr_intervals_history.extend(valid_intervals.tolist())
                        
                    else:
                        status_placeholder.warning("Signal is too noisy - keep your finger steady on the camera.")
                        st.session_state.heart_rate = None
                else:
                    status_placeholder.warning(f"Not enough peaks found for processing: {len(peaks)} peaks detected. "
                                               "Please keep your finger steady on the camera.")
                    st.session_state.heart_rate = None
                
                #Store for plotting
                st.session_state.signal_data = signal
                st.session_state.peaks_data = peaks
                
            except Exception as e:
                status_placeholder.error(f"Processing error: {str(e)}. Please try again.")
                st.session_state.heart_rate = None
        else:
            status_placeholder.info(f"Collecting data... ({len(buf)}/{fps*3} samples)")
            st.session_state.heart_rate = None
        
        #Update plot on the screen
        if len(st.session_state.signal_data) > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(st.session_state.signal_data, label="Filtered Signal", color='blue', alpha=0.7)
            
            if len(st.session_state.peaks_data) > 0:
                ax.plot(st.session_state.peaks_data, 
                       st.session_state.signal_data[st.session_state.peaks_data], 
                       "ro", label="Peaks", markersize=6)
            
            ax.set_title("Your Heart Rate PPG Signal with Peak Detection")
            ax.set_xlabel("Number of Samples")
            ax.set_ylabel("Intensity")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_placeholder.pyplot(fig)
            plt.close(fig)
        
        time.sleep(0.1)
else:
    status_placeholder.info("Start the camera to begin detection. Press stop to receive your diagnosis.")
    
    #Perform final analysis when video stops
    if not st.session_state.analysis_done and len(st.session_state.rr_intervals_history) > 0:
        st.session_state.analysis_done = True
        if st.session_state.start_time:
            recording_time = time.time() - st.session_state.start_time
        else:
            recording_time = 0.0
        
        st.subheader("Your Analysis Results")
        
        #Normal / Tachycardia / Bradycardia classification
        if st.session_state.heart_rate is not None:
            if st.session_state.heart_rate > 100:
                st.error("Tachycardia detected (resting heart rate is above 100 beats per minute)! Consider consulting a healthcare professional.")
            elif st.session_state.heart_rate < 60:
                st.error("Bradycardia detected (resting heart rate is less than 60 beats per minute)! Consider consulting a healthcare professional.")
            else:
                st.success("Normal heart rate detected.")
        
        #Arrhythmia classification
        rhythm_status, confidence = detect_arrhythmia(st.session_state.rr_intervals_history)
        
        if "irregular" in rhythm_status.lower():
            if confidence == "high":
                st.error("Rhythm Analysis: an irregular heartbeat is detected. Your heartbeat shows significant variability between beats. "
                         "This could indicate an irregular heartbeat pattern. Consider consulting a healthcare professional.")
            else:
                st.warning("Rhythm Analysis: a slightly irregular heartbeat is detected. Your heartbeat shows some variability between beats. "
                           "This be caused by normal factors such as breathing or moving while your finger is on the camera. "
                           "Consider consulting a healthcare professional for further details.")
        else:
            st.success(f"Rhythm Analysis: a regular heartbeat is detected. The time between your heartbeats shows a stable pattern. "
                       "This is a positive sign of normal heart function.")
        
        st.info(f"This analysis is based on {len(st.session_state.peaks_data)} heartbeats over approximately {recording_time:.1f} seconds. "
                "Please note that this tool cannot replace a professional consultation and is for informational purposes only.")
        
