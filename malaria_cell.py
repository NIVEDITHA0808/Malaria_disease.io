{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNczbQumYyBUDZlp1kq1XJ0",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU",
    "gpuClass": "standard"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/NIVEDITHA0808/Malaria_disease.io/blob/main/malaria_cell.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FBRZSKTH-Slg",
        "outputId": "08b1fc3b-6482-4e74-e8cc-6d530840ea1e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "from keras.models import Sequential, Model, load_model\n",
        "from keras.applications.vgg19 import VGG19\n",
        "from glob import glob\n",
        "from keras.layers import Flatten, Dense\n",
        "from keras.preprocessing.image import ImageDataGenerator\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.preprocessing import image\n",
        "from tensorflow.keras.applications.resnet50 import preprocess_input"
      ],
      "metadata": {
        "id": "WovN4UsT-s7a"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import tensorflow as tf\n",
        "tf.__version__"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "tFVw_NbHubq6",
        "outputId": "074803a2-bb38-45b2-c1ce-3e0c9dc611cc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'2.11.0'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "image_size=[224,224]\n",
        "train_data=\"/content/drive/MyDrive/Dataset/Train\"\n",
        "test_data=\"/content/drive/MyDrive/Dataset/Test\""
      ],
      "metadata": {
        "id": "3c3Hpqaq_7US"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "vgg19=VGG19(input_shape=image_size+[3], weights=\"imagenet\", include_top= False)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pxHEsNERtw8v",
        "outputId": "c7ce3292-2183-4ceb-eb5b-a24483a0507f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5\n",
            "80134624/80134624 [==============================] - 0s 0us/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "vgg19.summary()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vGxS1w6FvdA1",
        "outputId": "2942c6ff-fc4e-4e75-a0b6-03d94a44cca7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"vgg19\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " input_1 (InputLayer)        [(None, 224, 224, 3)]     0         \n",
            "                                                                 \n",
            " block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      \n",
            "                                                                 \n",
            " block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     \n",
            "                                                                 \n",
            " block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         \n",
            "                                                                 \n",
            " block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     \n",
            "                                                                 \n",
            " block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    \n",
            "                                                                 \n",
            " block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         \n",
            "                                                                 \n",
            " block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    \n",
            "                                                                 \n",
            " block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    \n",
            "                                                                 \n",
            " block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    \n",
            "                                                                 \n",
            " block3_conv4 (Conv2D)       (None, 56, 56, 256)       590080    \n",
            "                                                                 \n",
            " block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         \n",
            "                                                                 \n",
            " block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   \n",
            "                                                                 \n",
            " block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   \n",
            "                                                                 \n",
            " block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   \n",
            "                                                                 \n",
            " block4_conv4 (Conv2D)       (None, 28, 28, 512)       2359808   \n",
            "                                                                 \n",
            " block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         \n",
            "                                                                 \n",
            " block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   \n",
            "                                                                 \n",
            " block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   \n",
            "                                                                 \n",
            " block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   \n",
            "                                                                 \n",
            " block5_conv4 (Conv2D)       (None, 14, 14, 512)       2359808   \n",
            "                                                                 \n",
            " block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 20,024,384\n",
            "Trainable params: 20,024,384\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#storing each layer in an address and setting training off.\n",
        "for layer in vgg19.layers:\n",
        "  layer.trainable=False"
      ],
      "metadata": {
        "id": "OgDlEZytPpKE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "folders=glob(\"/content/drive/MyDrive/Dataset/Train/*\")"
      ],
      "metadata": {
        "id": "KjMq8xRYTYQQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "folders"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CqAeK7AgT9GE",
        "outputId": "6180e6fa-da70-486a-e7d5-31b6158abcd2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['/content/drive/MyDrive/Dataset/Train/Parasite',\n",
              " '/content/drive/MyDrive/Dataset/Train/Uninfected']"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x=Flatten()(vgg19.output)"
      ],
      "metadata": {
        "id": "KclQd_y5TI7E"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "prediction= Dense(len(folders), activation='softmax')(x)\n",
        "model=Model(inputs=vgg19.input, outputs=prediction)"
      ],
      "metadata": {
        "id": "su0CkmmFUzrm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.summary()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lYbuzPL1aaPZ",
        "outputId": "924bf582-c14a-4933-d51e-0ebb42f51340"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"model\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " input_1 (InputLayer)        [(None, 224, 224, 3)]     0         \n",
            "                                                                 \n",
            " block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      \n",
            "                                                                 \n",
            " block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     \n",
            "                                                                 \n",
            " block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         \n",
            "                                                                 \n",
            " block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     \n",
            "                                                                 \n",
            " block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    \n",
            "                                                                 \n",
            " block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         \n",
            "                                                                 \n",
            " block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    \n",
            "                                                                 \n",
            " block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    \n",
            "                                                                 \n",
            " block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    \n",
            "                                                                 \n",
            " block3_conv4 (Conv2D)       (None, 56, 56, 256)       590080    \n",
            "                                                                 \n",
            " block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         \n",
            "                                                                 \n",
            " block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   \n",
            "                                                                 \n",
            " block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   \n",
            "                                                                 \n",
            " block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   \n",
            "                                                                 \n",
            " block4_conv4 (Conv2D)       (None, 28, 28, 512)       2359808   \n",
            "                                                                 \n",
            " block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         \n",
            "                                                                 \n",
            " block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   \n",
            "                                                                 \n",
            " block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   \n",
            "                                                                 \n",
            " block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   \n",
            "                                                                 \n",
            " block5_conv4 (Conv2D)       (None, 14, 14, 512)       2359808   \n",
            "                                                                 \n",
            " block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         \n",
            "                                                                 \n",
            " flatten (Flatten)           (None, 25088)             0         \n",
            "                                                                 \n",
            " dense (Dense)               (None, 2)                 50178     \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 20,074,562\n",
            "Trainable params: 50,178\n",
            "Non-trainable params: 20,024,384\n",
            "_________________________________________________________________\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#to tell the model what loss funtion and optimization method to use, we use model.compile\n",
        "model.compile(loss=\"categorical_crossentropy\",optimizer=\"adam\", metrics=[\"accuracy\"])\n"
      ],
      "metadata": {
        "id": "Bi_qKbr-cI0w"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#to read the data we use image data generator to import the data from the dataset\n",
        "train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2, zoom_range=0.2, horizontal_flip=True)\n",
        "test_datagen=ImageDataGenerator(rescale=1./255)"
      ],
      "metadata": {
        "id": "bKWKPyINhJ1G"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "training_set=train_datagen.flow_from_directory(\"/content/drive/MyDrive/Dataset/Train\",target_size=(224,224),batch_size=32,class_mode=\"categorical\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iRDWv6chjY5R",
        "outputId": "940c56f2-2eda-4f65-b50c-53d960612ae8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 416 images belonging to 2 classes.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "training_set"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "C7_EbUINjZIa",
        "outputId": "c1c47ab2-5390-4374-d2fe-f0220a777a36"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.preprocessing.image.DirectoryIterator at 0x7f0c4c0bc3a0>"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "test_set=test_datagen.flow_from_directory(\"/content/drive/MyDrive/Dataset/Test\",target_size=(224,224),batch_size=32,class_mode=\"categorical\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "IFn5UqtOkNJZ",
        "outputId": "2dcef38b-7f1f-4f9e-a67d-1e53891cefab"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 134 images belonging to 2 classes.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#fit the model for training and validation\n",
        "r=model.fit(training_set,validation_data=test_set, epochs=2, steps_per_epoch=len(training_set), validation_steps=len(test_set))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7msr2S87kNQ1",
        "outputId": "20797031-b563-4965-d446-d4755e76bdf9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/2\n",
            "13/13 [==============================] - 7s 553ms/step - loss: 0.4707 - accuracy: 0.7837 - val_loss: 0.5831 - val_accuracy: 0.6866\n",
            "Epoch 2/2\n",
            "13/13 [==============================] - 6s 476ms/step - loss: 0.3775 - accuracy: 0.8197 - val_loss: 0.4012 - val_accuracy: 0.7910\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "if tf.test.gpu_device_name():\n",
        "  print(\"default gpu name: {}\", format(tf.test.gpu_device_name()))\n",
        "else:\n",
        "  print(\"install gpu\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "eqDrZEZUkNTr",
        "outputId": "892aa424-4406-49cb-fe77-51ec3339ab8a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "default gpu name: {} /device:GPU:0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.plot(r.history['loss'],\"r\",label='train loss')\n",
        "plt.plot(r.history['val_loss'],\"b\",label='train val_loss')\n",
        "plt.xlabel(\"epochs\")\n",
        "plt.ylabel(\"Loss\")\n",
        "plt.legend()\n",
        "plt.show()\n",
        "plt.plot(r.history['accuracy'],\"orange\",label='train accuracy')\n",
        "plt.plot(r.history['val_accuracy'],\"g\",label='train val_accuracy')\n",
        "plt.xlabel(\"epochs\")\n",
        "plt.ylabel(\"Accuracy\")\n",
        "plt.legend()\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 541
        },
        "id": "bBy53Cl6m3Zv",
        "outputId": "511b607c-025c-4443-e567-e8fc08eaec22"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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ObVu7fspRKToUIKQXTr+ePjyS99s98030Lw5dO8OP/wQdWQiUhCUIGS3iheHSy/1+kTv3j7dVLs23H8/qEmnSOGmBCE5sv/+3gBw7lxo3x7uuMM32r32muoTIoWVEoTslSOPhH//Gz74AMqV8812bdvC9OlRRyYi+U0JQnLlpJPgq6/gscd8VHHMMXDFFfDf/0YdmYjkFyUIybUSJeDqq72t+I03wnPPeVvxQYNgy5aooxORvFKCkDyrUAEGD4bZs+GEE/yAovr1YfRo1SdEUpkShOSbo4+GMWPgnXegZEk47TQvaM+eHXVkIpIbShCS7zp0gBkzYPhwmDYNGjWC666Dn3+OOjIR2RtKEJIQJUvCX//q9Ylrr4V//MPrE8OGwdatUUcnIjmhBCEJVakSPPywjyiaNfNidlqaH1gkIslNCUIKRP36MH68F663b4dOnfxj/vyoIxORXVGCkAJj5l1iZ8/2pbCffgoNG8JNN8Evv0QdnYj8mRKEFLhSpeDmm70+cemlXpeoVcs33W3bFnV0IpJFCUIic+CBXrz++msfSVx7LTRp4m08RCR6ShASuUaN4MMPvfHf+vVwyinQtSssXhx1ZCJFmxKEJAUzOPNMmDfPW4m//74Xtm+9FX79NeroRIomJQhJKmXKwG23eX3i/PPhwQe9PvH00776SUQKjhKEJKWqVeHZZ/2Y06OOgssv930UkyZFHZlI0aEEIUktPR0++cRPsvv5Z28GeM45sHRp1JGJFH5KEJL0zODcc31T3d13e0PAOnWgb18vaotIYihBSMrYd1+4804/H/uss+C++7yD7D//CTt2RB2dSOGjBCEpp3p1eOEF+OwzqFYNLrkEjjsOPv886shEChclCElZxx0HkyfDyJGwbBm0bAkXXADLl0cdmUjhoAQhKa1YMbj4Yp92uuMOeP11qF0b+veHjRujjk4ktSlBSKGw335w771eyM7IgH79vJD9yis69lQkt5QgpFCpUQNefRUmToTKleG88+D442Hq1KgjE0k9ShBSKJ1wgm+ye+op35XdvDn85S+wcmXUkYmkDiUIKbSKF4fLLvMEccst8OKLXp944AHYvDnq6ESSnxKEFHr77w9//zvMnQsnnwy33w716sG//636hMjuKEFIkXHUUfDmm94ptmxZ7x570kl+XraI/C8lCClyTj7ZDyl69FGYNQuO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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred=model.predict(test_set)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iPYmehkiobnM",
        "outputId": "d6cc7b1c-1b1c-407f-b497-d86c1d3754af"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "5/5 [==============================] - 1s 151ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#probability of it being parasite and uninfected respectively\n",
        "y_pred"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "caq6HUiZobwT",
        "outputId": "9b32b924-e81b-4c61-a022-95b5f172098e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[0.695729  , 0.304271  ],\n",
              "       [0.7488213 , 0.2511787 ],\n",
              "       [0.634921  , 0.365079  ],\n",
              "       [0.99594975, 0.00405024],\n",
              "       [0.950672  , 0.049328  ],\n",
              "       [0.42323557, 0.57676446],\n",
              "       [0.6796488 , 0.32035118],\n",
              "       [0.9409877 , 0.05901233],\n",
              "       [0.9106295 , 0.08937051],\n",
              "       [0.99491936, 0.00508062],\n",
              "       [0.6290082 , 0.3709918 ],\n",
              "       [0.64370364, 0.3562964 ],\n",
              "       [0.7161087 , 0.28389132],\n",
              "       [0.96247435, 0.03752562],\n",
              "       [0.5370171 , 0.46298292],\n",
              "       [0.5503223 , 0.44967774],\n",
              "       [0.34558946, 0.6544105 ],\n",
              "       [0.7909326 , 0.2090674 ],\n",
              "       [0.37598157, 0.62401843],\n",
              "       [0.46109512, 0.5389049 ],\n",
              "       [0.79566455, 0.20433544],\n",
              "       [0.7894642 , 0.21053576],\n",
              "       [0.6907908 , 0.3092093 ],\n",
              "       [0.9343201 , 0.06567988],\n",
              "       [0.40392426, 0.5960757 ],\n",
              "       [0.9323911 , 0.06760889],\n",
              "       [0.5939139 , 0.4060861 ],\n",
              "       [0.6173523 , 0.38264772],\n",
              "       [0.9445721 , 0.05542787],\n",
              "       [0.75917405, 0.24082598],\n",
              "       [0.66874003, 0.33125997],\n",
              "       [0.53190523, 0.46809477],\n",
              "       [0.6002227 , 0.39977732],\n",
              "       [0.7589107 , 0.2410893 ],\n",
              "       [0.4486644 , 0.55133563],\n",
              "       [0.5420909 , 0.4579091 ],\n",
              "       [0.93576247, 0.06423754],\n",
              "       [0.98427963, 0.01572039],\n",
              "       [0.50963414, 0.49036592],\n",
              "       [0.49717504, 0.50282496],\n",
              "       [0.9443825 , 0.05561751],\n",
              "       [0.98619276, 0.01380723],\n",
              "       [0.5016955 , 0.49830452],\n",
              "       [0.9734234 , 0.02657654],\n",
              "       [0.30675167, 0.69324833],\n",
              "       [0.33513436, 0.6648657 ],\n",
              "       [0.679586  , 0.320414  ],\n",
              "       [0.4854043 , 0.51459575],\n",
              "       [0.21138805, 0.78861195],\n",
              "       [0.9962727 , 0.00372731],\n",
              "       [0.70512325, 0.29487672],\n",
              "       [0.90643185, 0.0935681 ],\n",
              "       [0.83372056, 0.16627942],\n",
              "       [0.34376347, 0.6562365 ],\n",
              "       [0.40657607, 0.59342396],\n",
              "       [0.96596533, 0.0340347 ],\n",
              "       [0.97176886, 0.02823118],\n",
              "       [0.8907852 , 0.10921484],\n",
              "       [0.29183972, 0.7081603 ],\n",
              "       [0.97171974, 0.02828021],\n",
              "       [0.75589776, 0.24410222],\n",
              "       [0.7369938 , 0.26300624],\n",
              "       [0.38754663, 0.61245334],\n",
              "       [0.2417868 , 0.75821316],\n",
              "       [0.32182637, 0.6781736 ],\n",
              "       [0.8911115 , 0.10888851],\n",
              "       [0.9313862 , 0.06861375],\n",
              "       [0.9191304 , 0.08086963],\n",
              "       [0.9626701 , 0.03732989],\n",
              "       [0.6568205 , 0.34317955],\n",
              "       [0.6518924 , 0.34810752],\n",
              "       [0.26359987, 0.7364001 ],\n",
              "       [0.5097848 , 0.49021518],\n",
              "       [0.9733455 , 0.02665444],\n",
              "       [0.45052144, 0.5494786 ],\n",
              "       [0.7758804 , 0.2241196 ],\n",
              "       [0.6283582 , 0.37164187],\n",
              "       [0.2552599 , 0.7447401 ],\n",
              "       [0.65772986, 0.3422701 ],\n",
              "       [0.3200727 , 0.67992723],\n",
              "       [0.6340493 , 0.36595073],\n",
              "       [0.92894405, 0.07105598],\n",
              "       [0.6458446 , 0.35415545],\n",
              "       [0.69269735, 0.3073026 ],\n",
              "       [0.24959677, 0.7504033 ],\n",
              "       [0.68505144, 0.3149486 ],\n",
              "       [0.36860245, 0.63139755],\n",
              "       [0.98112154, 0.01887844],\n",
              "       [0.91879624, 0.08120377],\n",
              "       [0.5122953 , 0.4877047 ],\n",
              "       [0.988608  , 0.01139193],\n",
              "       [0.8520713 , 0.1479287 ],\n",
              "       [0.42880815, 0.57119185],\n",
              "       [0.6523712 , 0.34762874],\n",
              "       [0.6132664 , 0.3867336 ],\n",
              "       [0.14618418, 0.8538158 ],\n",
              "       [0.29831958, 0.70168036],\n",
              "       [0.1839311 , 0.8160689 ],\n",
              "       [0.36377597, 0.6362241 ],\n",
              "       [0.9424337 , 0.05756634],\n",
              "       [0.6571559 , 0.34284413],\n",
              "       [0.9200674 , 0.07993257],\n",
              "       [0.6313992 , 0.36860076],\n",
              "       [0.5283544 , 0.4716456 ],\n",
              "       [0.9049367 , 0.09506335],\n",
              "       [0.9366531 , 0.06334695],\n",
              "       [0.8684192 , 0.13158083],\n",
              "       [0.92322695, 0.07677305],\n",
              "       [0.25803265, 0.7419673 ],\n",
              "       [0.8973134 , 0.10268664],\n",
              "       [0.8924763 , 0.10752364],\n",
              "       [0.5356458 , 0.46435428],\n",
              "       [0.27506077, 0.7249392 ],\n",
              "       [0.9601823 , 0.03981771],\n",
              "       [0.41143748, 0.5885625 ],\n",
              "       [0.96154994, 0.03845009],\n",
              "       [0.7343825 , 0.26561752],\n",
              "       [0.43707255, 0.5629275 ],\n",
              "       [0.9958085 , 0.00419157],\n",
              "       [0.32493803, 0.67506194],\n",
              "       [0.23311876, 0.7668812 ],\n",
              "       [0.38430494, 0.61569506],\n",
              "       [0.19998376, 0.8000162 ],\n",
              "       [0.8197648 , 0.18023519],\n",
              "       [0.37824845, 0.6217515 ],\n",
              "       [0.34702957, 0.65297043],\n",
              "       [0.37510383, 0.6248961 ],\n",
              "       [0.5131151 , 0.48688492],\n",
              "       [0.23734882, 0.7626512 ],\n",
              "       [0.99809724, 0.00190273],\n",
              "       [0.9756565 , 0.02434349],\n",
              "       [0.92929125, 0.07070874],\n",
              "       [0.985898  , 0.01410197],\n",
              "       [0.48594502, 0.514055  ]], dtype=float32)"
            ]
          },
          "metadata": {},
          "execution_count": 42
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# we need the max in the matrix\n",
        "y_pred=np.argmax(y_pred,axis=1)"
      ],
      "metadata": {
        "id": "sO4bIUhJoxgx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#0: infected, 1: uninfected\n",
        "y_pred"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mmtPQCRZoxlq",
        "outputId": "f02e4b0e-eca8-4144-adb8-ba826489beaf"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0,\n",
              "       0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,\n",
              "       1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0,\n",
              "       0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0,\n",
              "       0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,\n",
              "       0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0,\n",
              "       0, 1])"
            ]
          },
          "metadata": {},
          "execution_count": 44
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.save('model_vgg19.h5')"
      ],
      "metadata": {
        "id": "lXwKp_j9sMA2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model=load_model(\"model_vgg19.h5\")"
      ],
      "metadata": {
        "id": "Rps_kxXWoxxJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "img=image.load_img(\"/content/drive/MyDrive/Dataset/Test/Parasite/C39P4thinF_original_IMG_20150622_110115_cell_115.png\", target_size=(224,224,3))"
      ],
      "metadata": {
        "id": "1GW4oDe0shdb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x=image.img_to_array(img)\n",
        "x"
      ],
      "metadata": {
        "id": "7FvkXFMOsY26"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rs8I6R6tty9U",
        "outputId": "b3881734-1c94-4eb8-be3d-03f5a6384676"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(224, 224, 3)"
            ]
          },
          "metadata": {},
          "execution_count": 46
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#reshaping\n",
        "x=x.astype('float32')/255"
      ],
      "metadata": {
        "id": "j28of4wNt0gF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x=np.expand_dims(x,axis=0)\n",
        "img_data=preprocess_input(x)\n",
        "img_data.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vCnEaLRquBGb",
        "outputId": "286228a6-a1ca-4b8c-fe32-9fa1ae8f03f5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(1, 224, 224, 3)"
            ]
          },
          "metadata": {},
          "execution_count": 47
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.predict(img_data)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PwkObdYqD5JS",
        "outputId": "b0e52671-a681-4b14-933d-0486653a462a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 1s 843ms/step\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[0.9105897 , 0.08941028]], dtype=float32)"
            ]
          },
          "metadata": {},
          "execution_count": 48
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "a=np.argmax(model.predict(img_data),axis=1)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZPrwkHndvC2r",
        "outputId": "f7f54c90-5eac-4714-d176-b44ca96e67de"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 0s 21ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "if a==1:\n",
        "  print(\"the patient is uninfected\")\n",
        "else:\n",
        "  print(\"the patient is infected\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5MiWRRl_vlxc",
        "outputId": "1c3a2c66-06c6-47f5-ea19-0d826649521e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "the patient is infected\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def results(data, model):\n",
        "  img=image.load_img(data, target_size=(224,224,3))\n",
        "  x=image.img_to_array(img)\n",
        "  x1=x.astype('float32')/255\n",
        "  x2=np.expand_dims(x1,axis=0)\n",
        "  img_data=preprocess_input(x2)\n",
        "  a=np.argmax(model.predict(img_data),axis=1)\n",
        "  if a==1:\n",
        "    print(\"the patient is uninfected\")\n",
        "  else:\n",
        "    print(\"the patient is infected\")"
      ],
      "metadata": {
        "id": "o8F9kdiJv7Fi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data=\"/content/drive/MyDrive/Dataset/Test/Parasite/C39P4thinF_original_IMG_20150622_110115_cell_115.png\"\n",
        "results(data,model)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "A8H_CTT4wp0-",
        "outputId": "6ec1634b-f19e-4afa-8de7-88e6c00275a1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 0s 23ms/step\n",
            "the patient is infected\n"
          ]
        }
      ]
    }
  ]
}