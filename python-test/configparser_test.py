import configparser

file_path1 = 'data/example.ini'

def t1():
    config = configparser.ConfigParser()    #类中一个方法 #实例化一个对象

    # DEFAULT对后面任何一组参数都有用
    config["DEFAULT"] = {'ServerAliveInterval': '45',
                        'Compression': 'yes',
                        'CompressionLevel': '9',
                        'ForwardX11':'yes'
                        }	#类似于操作字典的形式

    config['bitbucket.org'] = {'User':'Atlan'} #类似于操作字典的形式

    config['topsecret.server.com'] = {'Host Port':'50022','ForwardX11':'no'}

    with open(file_path1, 'w') as configfile:

        config.write(configfile)	#将对象写入文件

def t2():
    config = configparser.ConfigParser()

    #---------------------------查找文件内容,基于字典的形式

    print(config.sections())        #  []

    config.read(file_path1)

    print(config.sections())        #   ['bitbucket.org', 'topsecret.server.com']

    print('bytebong.com' in config) # False
    print('bitbucket.org' in config) # True


    print(config['bitbucket.org']["user"])  # Atlan

    print(config['DEFAULT']['Compression']) #yes

    print(config['topsecret.server.com']['ForwardX11'])  #no
    print(config['topsecret.server.com']['compressionlevel'])  #no


    print(config['bitbucket.org'])          #<Section: bitbucket.org>

    for key in config['bitbucket.org']:     # 注意,有default会默认default的键
        print('=====', key)

    print(config.options('bitbucket.org'))  # 同for循环,找到'bitbucket.org'下所有键

    print(config.items('bitbucket.org'))    #找到'bitbucket.org'下所有键值对

    print(config.get('bitbucket.org','compression')) # yes       get方法Section下的key对应的value


if __name__ == "__main__":
    t2()