import tensorflow as tf
import math

class OneCycleLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, total_steps, pct_start=0.3, div_factor=25.0, final_div_factor=1e4):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor

    def __call__(self, step):
        pct = tf.cast(step, tf.float32) / self.total_steps
        if pct < self.pct_start:
            # Linear warm-up
            phase = pct / self.pct_start
            lr = self.initial_lr + phase * (self.max_lr - self.initial_lr)
        else:
            # Cosine annealing
            phase = (pct - self.pct_start) / (1 - self.pct_start)
            lr = self.final_lr + 0.5 * (self.max_lr - self.final_lr) * (1 + tf.cos(math.pi * phase))
        return lr