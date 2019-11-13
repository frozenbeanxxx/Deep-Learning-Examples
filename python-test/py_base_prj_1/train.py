import os
import time
import numpy as np
import tensorflow as tf

import config
import tfutil
#import dataset
import misc

if __name__ == "__main__":
    misc.init_output_logging()
    print(config.tf_config)
    print(config.env)
    print(config.desc)
    print(config.dataset)
    print(config.train)
    print(config.G)
    print(config.D)
    print(config.G_opt)
    print(config.D_opt)
    print(config.G_loss)
    print(config.D_loss)
    print(config.sched)
    print(config.grid)
    
    np.random.seed(config.random_seed)
    os.environ.update(config.env)

    
    misc.set_output_log_file('log.txt')