import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="CNN Visualisation Toolbox",
    layout="wide"
)

import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function
add_bg_from_local("C://CNN Visualisation toolbox 251100660023//background.jpeg")


st.title("CNN Visualisation Toolbox (251100660023)")

# Sidebar
st.sidebar.header("CNN Parameters to choose from")

kernel_size = st.sidebar.selectbox( 
    "Kernel Size", [1, 2, 3, 5, 7] #set of kernal sizes given
) 

# Multiple strides
strides = st.sidebar.multiselect(
    "Select Strides",
    options=[1, 2, 3, 4], #set of options for strides number
    default=[1, 2] #default stride values
)

num_filters = st.sidebar.slider(
    "Number of Filters",
    min_value=1, #minim value of filter
    max_value=9, #maximum value of filter
    value=3
)

num_layers = st.sidebar.slider(
    "Number of Convolution Layers",
    min_value=1,
    max_value=5,
    value=2
)

# -------- DROPOUT CONTROLS (ADDED) --------
num_dropout_layers = st.sidebar.slider(
    "Number of Dropout Layers",
    min_value=0,
    max_value=5,
    value=1
)

dropout_prob = st.sidebar.slider(
    "Dropout Probability",
    min_value=0.0,
    max_value=0.9,
    value=0.3,
    step=0.05
)
# -----------------------------------------

activation_name = st.sidebar.selectbox(
    "Activation Function", #since RELU has some disadvantages we also use variants of RELU
    ["None", "ReLU", "LeakyReLU", "Tanh", "ELU", "SELU"] #activation functions used
)

padding_type = st.sidebar.selectbox(
    "Padding Type", #padding
    ["None", "Zero Padding", "One Padding"]
    #Zero padding : padding wirth zeros
    #One padding : padding with ones
)

padding_size = st.sidebar.slider(
    "Padding Size", 0, 5, 1 #padding sizes
)

pooling_type = st.sidebar.selectbox(
    "Pooling Type",
    ["None", "Max Pooling", "Min Pooling", "Average Pooling"] #types of pooling
)

pool_size = st.sidebar.selectbox(
    "Pool Size", [2, 3]
)

# Image Upload
uploaded = st.file_uploader(
    "Upload an Image",
    type=["png", "jpg", "jpeg"] 
)

if uploaded is None:
    st.info("Please upload an image to continue.")
    st.stop()

# Image Preprocessing
transform = T.Compose([
    T.Grayscale(),
    T.Resize((128, 128)),
    T.ToTensor()
])

image = transform(Image.open(uploaded)).unsqueeze(0)

# Padding
padded_image = image.clone() 

if padding_type == "Zero Padding":
    padded_image = F.pad(image, (padding_size,) * 4, value=0)
elif padding_type == "One Padding":
    padded_image = F.pad(image, (padding_size,) * 4, value=1)


def apply_activation(x, name):
    if name == "ReLU":
        return F.relu(x)
    elif name == "LeakyReLU":
        return F.leaky_relu(x, 0.1)
    elif name == "Tanh":
        return torch.tanh(x)
    elif name == "ELU":
        return F.elu(x)
    elif name == "SELU":
        return F.selu(x)
    else:
        return x

# Pooling
def min_pool2d(x, k):
    return -F.max_pool2d(-x, k)

def apply_pooling(x, pool_type, k):
    if pool_type == "Max Pooling":
        return F.max_pool2d(x, k)
    elif pool_type == "Average Pooling":
        return F.avg_pool2d(x, k)
    elif pool_type == "Min Pooling":
        return min_pool2d(x, k)
    else:
        return x

# Visualization Helper
def show_tensor(tensor, title, cols=4):
    n = tensor.shape[1]
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i in range(len(axes)):
        axes[i].axis("off")
        if i < n:
            axes[i].imshow(
                tensor[0, i].detach().cpu().numpy(),
                cmap="gray"
            )
            axes[i].set_title(f"{title} {i}")

    st.pyplot(fig)

# Display Original & Padded Image
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(
        image[0, 0].cpu().numpy(),
        clamp=True,
        channels="GRAY"
    )

with col2:
    st.subheader("After Padding")
    st.image(
        padded_image[0, 0].cpu().numpy(),
        clamp=True,
        channels="GRAY"
    )
    st.write("Shape:", padded_image.shape)

st.divider()

# STRIDE CONVOLUTION
st.subheader("Effect of Different Strides")

for stride in strides:
    st.markdown(f"## Stride = {stride}")

    x = padded_image
    in_channels = 1

    for layer in range(num_layers):
        st.markdown(f"### Layer {layer + 1}")

        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=stride
        )

        x = conv(x)
        st.markdown("Convolution Output")
        show_tensor(x, f"Conv L{layer+1}")

        x = apply_activation(x, activation_name)
        st.markdown("After Activation")
        show_tensor(x, f"Act L{layer+1}")

        # -------- DROPOUT APPLIED (ADDED) --------
        if layer < num_dropout_layers:
            dropout = nn.Dropout2d(p=dropout_prob)
            dropout.train()  # force dropout visualization
            x = dropout(x)
            st.markdown("After Dropout")
            show_tensor(x, f"Dropout L{layer+1}")
        # ----------------------------------------

        x = apply_pooling(x, pooling_type, pool_size)
        st.markdown("After Pooling")
        show_tensor(x, f"Pool L{layer+1}")

        st.write("Shape:", x.shape)

        # Update input channels for next layer
        in_channels = num_filters

        st.divider()

# Activation Distribution
st.subheader("Activation Distribution")

fig = plt.figure()
plt.hist(
    x.flatten().detach().cpu().numpy(),
    bins=50
)
plt.title("Activation Value Distribution")
st.pyplot(fig)
