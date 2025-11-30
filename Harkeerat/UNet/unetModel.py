from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    BatchNormalization,
    Activation,
    Dropout,
)


def conv_block(input_tensor, num_filters):
    """
    Standard U-Net Convolution Block:
    Conv2D -> BatchNormalization -> ReLU -> Conv2D -> BatchNormalization -> ReLU
    """
    x = Conv2D(num_filters, (3, 3), padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def build_unet(input_shape):
    """
    Builds the U-Net architecture.
    """
    inputs = Input(input_shape)

    # --- Contracting Path (Encoder) ---
    c1 = conv_block(inputs, 32)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 256)
    p4 = MaxPooling2D((2, 2))(c4)

    # --- Bridge (Bottleneck) ---
    c5 = conv_block(p4, 512)
    c5 = Dropout(0.5)(c5)  # Dropout to prevent overfitting

    # --- Expansive Path (Decoder) ---
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, 256)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, 128)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block(u8, 64)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block(u9, 32)

    # --- Output Layer ---
    # Sigmoid activation for binary classification (Road vs Background)
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
