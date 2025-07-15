# docker build --network=host -t droid-downloader . 
# docker run -ti --gpus=all --shm-size=10g --network=host -v $(pwd):/root/droid-example droid-downloader bash
FROM stereolabs/zed:4.2-runtime-cuda11.8-ubuntu22.04

# RUN apt-get update -y && apt-get install -y \
#     fish \
#     python3-pip \
#     python3-opencv \
#     git

# Install Google Cloud SDK
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/install_google_cloud_sdk.bash
RUN chmod +x install_google_cloud_sdk.bash
RUN ./install_google_cloud_sdk.bash --disable-prompts

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


# Add gsutil to PATH
ENV PATH="/root/google-cloud-sdk/bin:$PATH"

# Set working directory
WORKDIR /root/droid-example

# Copy the scripts
# COPY . .
