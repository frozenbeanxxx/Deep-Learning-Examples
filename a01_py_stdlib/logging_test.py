import logging
import os
import time

def default_logging_print():
    print("run ", __name__)
    logging.debug(u"苍井空")
    logging.info(u"麻生希")
    logging.warning(u"小泽玛利亚")
    logging.error(u"桃谷绘里香")
    logging.critical(u"泷泽萝拉")

def set_logging_level():
    print("run ", __name__)
    logging.basicConfig(level=logging.NOTSET)
    logging.debug(u"苍井空")
    logging.info(u"麻生希")
    logging.warning(u"小泽玛利亚")
    logging.error(u"桃谷绘里香")
    logging.critical(u"泷泽萝拉")

def output_file():
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    outdir = 'logs/'
    os.makedirs(outdir, exist_ok=True)
    log_name = outdir + rq + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.NOTSET)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    # 日志
    logger.debug('this is a logger debug message')
    logger.info('this is a logger info message')
    logger.warning('this is a logger warning message')
    logger.error('this is a logger error message')
    logger.critical('this is a logger critical message')

def output_file_2():
    logger = logging.getLogger(__name__)
    logger.info('output_file_2')

def create_logger():
    logger = logging.Logger('log1')
    #logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # 输出到console的log等级的开关
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info('create_logger')

def use_logger():
    logger = logging.getLogger('log1')
    logger.setLevel(logging.INFO)
    logger.info('use_logger')

if __name__ == '__main__':
    log_dir = './logs'
    #default_logging_print()
    #set_logging_level()
    #output_file()
    #output_file_2()
    create_logger()
    use_logger()
