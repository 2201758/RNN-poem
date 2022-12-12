import tensorflow as tf
from data import PoetryDataGenerator, poetry, tokenizer
from model import model
import settings
import utils


class Evaluate(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save(settings.BEST_MODEL_PATH)
        print()
        for i in range(settings.SHOW_NUM):
            print(utils.generate_random_poetry(tokenizer, model))

#创建数据集
data_generator = PoetryDataGenerator(poetry, random=True)
#开始训练
model.fit_generator(data_generator.for_fit(),
                    steps_per_epoch=data_generator.steps,
                    epochs=settings.TRAIN_EPOCHS,
                    callbacks=[Evaluate()])
