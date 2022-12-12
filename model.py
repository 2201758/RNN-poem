import tensorflow as tf
from data import tokenizer

#构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Input((None,)),
    tf.keras.layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=128),
    #第一层
    tf.keras.layers.LSTM(128, dropout=0.5, return_sequences=True),
    #第二层
    tf.keras.layers.LSTM(128, dropout=0.5, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(tokenizer.vocab_size, activation='softmax')),
])

#查看模型结构
model.summary()
#配置优化器和损失函数
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy)















