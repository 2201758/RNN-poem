from data import tokenizer
import tensorflow as tf
import settings
import utils

#加载训练模型
model = tf.keras.models.load_model(settings.BEST_MODEL_PATH)

for i in range(3):
    #续写
    print(utils.generate_random_poetry(tokenizer, model, input("请输入诗的开头： ")))
   #藏头诗
    print(utils.generate_acrostic(tokenizer, model, input("藏头诗内容：")))



