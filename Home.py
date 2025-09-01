# Home page
import streamlit as st
import torch.nn as nn

st.set_page_config(page_title="Home")

st.write("# VascularPulse: Your Vascular-Related Health Toolkit")

st.sidebar.success("Select a tool above.")

st.markdown(
    """
    VascularPulse is a free, accessible application designed to assist in the early detection and management of vascular-related conditions.
    This app leverages artificial intelligence (AI) and mathematical methods to support your health and well-being.
    **Select a tool from the sidebar to get started.**
    ### Stroke Risk Assessment System
    This tool is an advanced AI-driven system that provides a preliminary assessment for a potential stroke. It analyzes key features
    from your facial image and combines them with essential health record data you provide to give an initial diagnosis. 
    ### Heart Rate and Arrhythmia Detector
    The Heart Rate and Arrhythmia Detector analyzes your heart rhythms using photoplethysmography (PPG) technology to screen for fast, slow,
    or irregular heart rhythms.
    ### Chatbot
    The Chatbot is an AI-powered virtual assistant to help answer your vascular-related health needs. It can provide clear and reliable answers
    on topics like symptoms, medical conditions, and treatment options.
"""
)


#Model definitions
class StrokeClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(262144,1024),
            nn.ReLU(),
            nn.Dropout(0.3),    ### Dropout 
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),    ### Dropout 
            nn.Linear(512,1),
            nn.Sigmoid()
        )
    
    def forward(self, xb):
        return self.network(xb)
