import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model

print("TensorFlow:", tf.__version__)

bert_preprocess = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.load("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2")

print("✅ BERT Preprocessor & Encoder loaded")

sample_text = ["TensorFlow is amazing!", "I love learning about transformers."]

text_inputs = tf.constant(sample_text)
encoder_inputs = bert_preprocess(text_inputs)

print("Keys:", encoder_inputs.keys())

outputs = bert_encoder(encoder_inputs)

# Pooled output → sentence representation
pooled_output = outputs['pooled_output']

print("Pooled output shape:", pooled_output.shape)


text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = hub.KerasLayer(bert_preprocess)(text_input)
outputs = hub.KerasLayer(bert_encoder, trainable=True)(preprocessed_text)

pooled_output = outputs['pooled_output']
dropout = layers.Dropout(0.1)(pooled_output)
classifier = layers.Dense(1, activation='sigmoid')(dropout)

model = Model(inputs=text_input, outputs=classifier)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Dummy dataset
texts = np.array(["Good movie", "Bad movie", "Great film", "Terrible film"])
labels = np.array([1, 0, 1, 0])

model.fit(texts, labels, epochs=2)

print("✅ Model trained on dummy dataset")