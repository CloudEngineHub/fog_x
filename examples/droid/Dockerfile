# docker build --network=host -t droid-downloader . 
# docker run --network=host -v $(pwd)/droid_data:/root/droid-example/droid_downloaded_data droid-downloader bash -c  "python3 droid_downloader.py"
FROM stereolabs/zed:4.2-runtime-cuda11.8-ubuntu22.04

# RUN apt-get update -y && apt-get install -y \
#     fish \
#     python3-pip \
#     python3-opencv \
#     git

# Install Python dependencies
RUN pip install \
    argparse \
    scipy==1.10.1 \
    h5py \
    gcsfs \
    tensorflow_datasets \
    tensorflow \
    ray[default] \
    flask \
    spacy \
    numpy \
    requests \
    opencv-python

# Install Google Cloud SDK
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/install_google_cloud_sdk.bash
RUN chmod +x install_google_cloud_sdk.bash
RUN ./install_google_cloud_sdk.bash --disable-prompts

# Add gsutil to PATH
ENV PATH="/root/google-cloud-sdk/bin:$PATH"

# Set working directory
WORKDIR /root/droid-example

# Copy the scripts
COPY . .

# Install robodm (assuming it's in the parent directory during build)
# You'll need to mount or copy the robodm package
# COPY ../../robodm /root/robodm_pkg
# RUN cd /root/robodm_pkg && pip install -e .