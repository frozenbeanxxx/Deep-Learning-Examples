'''
validate label
created by AndrewWei, 20210727
'''

import os
import cv2
import numpy as np
from natsort import natsorted


def validate_label():
    lable_file = 'veg_200_labels.txt'
    with open(lable_file, 'r') as f:
        lines = f.readlines()
    label_dict = {}
    for l in lines:
        ll = l.strip().split(' ')
        label_dict[ll[1]] = ll[0]
    label_keys = label_dict.keys()

    root_dir = '/media/weixing/diskD/dataset_jkyy/FoodSingle/single_food_all_sz720_20'
    listdirs = os.listdir(root_dir)
    listdirs = natsorted(listdirs)

    lables = []
    for l in label_keys:
        if l in listdirs:
            lables.append(l)
            #print(l)
    print(label_dict)
    print(lables)
    print('over')

def validate_label_202107271548():
    '''
    error category
    '''
    label_key = ['粽子', '鸡蛋', '烧卖', '花生','豆浆', '西红柿', '咸鸭蛋', '樱桃', '葡萄', '芒果干', '火龙果', '李子', '肠粉', '海带丝',
        '洋葱切碎', '蛋黄酥', '鱼干', '杨梅', '巧克力冰棒', '橘子', '香蕉', '芒果', '葵瓜子', '蓝莓', '奥利奥饼干', '果冻布丁', '木耳', '黑芝麻丸', '椰子', '切盘西瓜',
        '豆腐块', '包子', '核桃', '小南瓜（贝贝南瓜）', '红龙果（切盘）', '玉米香肠', '桑葚', '蒸饺', '长豆角', '红枣', '蛋卷', '红薯', '长茄子', '海苔', '花生米',
        '榴莲（整个）', '黑玉米', '青团', '柚子', '绿豆饼', '油条', '圣女果', '腰果', '冰淇淋', '杨梅', '豌豆', '柠檬', '麻薯', '小龙虾', '苦瓜',
        '生姜', '八宝饭', '削皮苹果', '费列罗巧克力', '蛋筒冰淇淋', '枇杷', '基围虾', '西瓜子', '秋葵炒肉', '鸡腿', '山竹', '烤面筋',
        ]

    '''
    花生 -> 花生带壳
    海带丝 -> 凉拌海带丝
    鱼干 -> 鱼片干
    葵瓜子 -> 葵花瓜子
    果冻布丁 -> 果冻
    木耳 -> 黑木耳泡发
    切盘西瓜 -> 西瓜
    豆腐块 -> 豆腐
    核桃 -> 干核桃
    小南瓜 -> 南瓜
    红龙果 -> 火龙果
    长豆角 -> 豆角
    长茄子 -> 茄子
    花生米 -> 花生仁
    榴莲（整个） -> 榴莲
    青团 -> 青团子
    冰淇淋 -> 冰激凌
    柚子 -> 西柚、红心柚子、白心柚子
    '''
    label_keys = ['粽子', '鸡蛋', '烧卖', '花生带壳', '豆浆', '西红柿', '咸鸭蛋', '樱桃', '葡萄', '芒果干', '火龙果', '李子', '肠粉', '凉拌海带丝',
                  '洋葱切碎', '蛋黄酥', '鱼片干', '杨梅', '巧克力冰棒', '橘子', '香蕉', '芒果', '葵花瓜子', '蓝莓', '奥利奥饼干', '果冻', '黑木耳泡发', '黑芝麻丸', '椰子', '西瓜',
                  '豆腐', '包子', '干核桃', '南瓜', '火龙果', '玉米香肠', '桑葚', '蒸饺', '豆角', '红枣', '蛋卷', '红薯', '茄子', '海苔', '花生仁',
                  '榴莲', '黑玉米', '青团子', '柚子', '绿豆饼', '油条', '圣女果', '腰果', '冰激凌', '杨梅', '豌豆', '柠檬', '麻薯', '小龙虾', '苦瓜',
                  '生姜', '八宝饭', '削皮苹果', '费列罗巧克力', '蛋筒冰淇淋', '枇杷', '基围虾', '西瓜子', '秋葵炒肉', '鸡腿', '山竹', '烤面筋',
                  ]

    root_dir = '/media/weixing/diskD/dataset_jkyy/FoodSingle/single_food_all_sz720_20'
    listdirs = os.listdir(root_dir)
    listdirs = natsorted(listdirs)

    lables = []
    for l in label_keys:
        if l in listdirs:
            lables.append(l)
        else:
            print(l)
    print(lables)
    print('len(label_keys)', len(label_keys))
    print('len(lables)', len(lables))
    print('over')


if __name__ == "__main__":
    #validate_label()
    validate_label_202107271548()


def tmp():
    label_dict = {'牛膝': 'achyranthes', '沙参': 'adenophora', '双孢蘑菇': 'agaricus_bisporus', '姬松茸': 'agaricus_blazei_murill',
     '龙芽草': 'agrimony', '茶树菇': 'agrocybe_aegerita', '葱': 'allium', '白茨菰': 'arrowhead', '水蒿': 'artemisia_selengensis',
     '芦笋': 'asparagus', '0': 'horst', '莴笋': 'asparagus_lettuce', '四棱豆': 'asparagus_pea', '红豆': 'azuki_beans',
     '苦瓜': 'balsam_pear', '竹笋': 'bamboo_shoot', '落葵': 'basella_rubra', '罗勒': 'basil', '细枝雾冰藜': 'bassia_scoparia',
     '紫苏': 'beefsteak_plant', '甜根菜': 'beetroot', '野辣椒': 'bird_pepper', '黑豆芽': 'black_bean_sprouts',
     '菊牛蒡': 'black_salsify', '黑豆': 'black_soya_bean', '牛肝菌': 'bolete', '葫芦': 'bottle_gourd', '蚕豆': 'broad_bean',
     '西兰花': 'broccoli', '孢子甘蓝': 'brussels_sprouts', '小葱': 'bunching_onion', '苜蓿草': 'burclover', '牛蒡根': 'burdock_root',
     '菇娘': 'cape_gooseberry', '飞廉': 'carduus', '胡萝卜': 'carrot', '香蒲': 'cattail', '根芹': 'celeriac', '芹菜': 'celery',
     '积雪草': 'centella_asiatica', '鸡油菌': 'chantarelle', '菊苣': 'chicory', '地笋': 'Chinese_artichoke',
     '大白菜': 'Chinese_cabbage', '芥蓝': 'Chinese_kale', '锦葵': 'Chinese_mallow', '中国南瓜': 'Chinese_pumpkin',
     '山药': 'Chinese_yam', '欧洲韭菜': 'chive', '佛手瓜': 'chocho', '菊花': 'chrysanthemum', '鸭跖草': 'commelina',
     '毛头鬼伞': 'coprinus_comatus', '香菜': 'coriander', '玉米': 'corn', '豇豆': 'cowpea', '水芹': 'cress', '黄瓜': 'cucumber',
     '鼠曲草': 'cudweed', '羽衣甘蓝': 'curly_kale', '小花琉璃草': 'cynoglossum_lanceolatum', '蒲公英': 'dandelion', '黄花菜': 'day_lily',
     '竹荪': 'dictyophora', '苋菜': 'edible_amaranth', '茄子': 'eggplant', '苦苣': 'endive', '金针菇': 'enoki_mushroom',
     '笔管草': 'equisetum_debile', '何首乌': 'fallopia_multiflora', '青葙子': 'feather_cockscomb', '茴香': 'fennel',
     '菜薹': 'flower_Chinese_cabbage', '牛膝菊': 'galinsoga_parviflora', '大蒜': 'garlic', '韭菜': 'garlic_chive',
     '蒜苔': 'garlic_sprouts', '姜': 'ginger', '洋蓟': 'globe_artichoke', '枸杞': 'goji_berry', '芡实': 'gorgon_fruit_seed',
     '瓠子': 'gourd', '玉竹': "great_Solomon's-seal", '大葱': 'green_Chinese_onion', '绿茄子': 'green_eggplant',
     '青萝卜': 'green_radish', '观音苋': 'gynura_bicolor', '毛瓜': 'hairy_squash', '结球甘蓝': 'head_cabbage',
     '鸭脚艾': 'Herb_of_Ghostplant_Wormwood', '猴头菇': 'hericium', '鱼腥草': 'houttuynia_cordata', '扁豆': 'hyacinth_bean',
     '真姬菇': 'hypsizigus_marmoreus', '菊芋': 'jerusalem_artichoke', '木耳': "Jew's-ear", '马兰': 'kalimeris',
     '芸豆': 'kidney_bean', '芸豆种子': 'kidney_bean_seed', '球茎甘蓝': 'kohlrabi', '魔芋': 'konnyaku', '野葛': 'kudzu',
     '油麦菜': 'leaf_lettuce', '韭葱': 'leek', '生菜': 'lettuce', '食用百合': 'Lily', '藕带': 'lotus', '藕': 'lotus_root',
     '莲子': 'lotus_seed', '莲蓬': 'lotus_seedpod', '广东丝瓜': 'luffa_acutangula', '丝瓜': 'luffa_cylindrica', '松茸': 'matsutake',
     '水飞蓟': 'milk_thistle', '薄荷': 'mint', '蘘荷': 'mioga_ginger', '鸭儿芹': 'mitsuba', '羊肚菌': 'morel', '绿豆': 'mung_bean',
     '绿豆芽': 'mung_bean_sprouts', '芥菜': 'mustard', '滑菇': 'nameko', '菊花脑': 'nankimgense', '番杏': 'New_Zealand_spinach',
     '秋葵': 'okra', '洋葱': 'onion', '蕨菜': 'ostrich_fern', '平菇': 'oyster_mushroom', '小白菜': 'pakchoi', '欧芹': 'parsley',
     '欧防风': 'parsnip', '豌豆': 'pea', '花生芽': 'peanut_sprouts', '辣椒': 'pepper', '甜椒': 'pimento',
     '桔梗': 'platycodon_grandiflorum', '杏鲍菇': 'pleurotus_eryngii', '白灵菇': 'pleurotus_nebrodensis',
     '黄精': 'polygonatum_sibiricum', '酸模叶蓼': 'polygonum_lapathifolium', '马铃薯': 'potato', '带刺莴苣': 'prickly_lettuce',
     '南瓜': 'pumpkin', '紫菜薹': 'purple_cai-tai', '马齿苋': 'purslane', '紫甘蓝': 'red_cabbage', '水萝卜': 'red_radish',
     '大黄': 'rhubarb', '绿菇': 'russula_virescens', '皱叶甘蓝': 'savoy_caggage', '芥头': 'scallion',
     '海发菜': 'sea_of_nostoc_flagelliforme', '夏枯草': 'self-heal', '红葱头': 'shallot', '荠菜': "shepherd's_purse",
     '香菇': 'shiitake', '豌豆荚': 'sieva_bean', '豌豆种子': 'sieva_bean_seed', '蕨麻': 'silverweed', '蛇瓜': 'snake_gourd',
     '酢浆草': 'sorrel', '毛豆': 'soybean', '黄豆': 'soybean_seed', '黄豆芽': 'soybean_sprouts', '菠菜': 'spinach',
     '花椰菜': 'sprouting_broccoli', '草莓': 'strawberry', '草菇': 'straw_mushroom', '红薯': 'sweet_potato',
     '瑞士甜菜': 'swiss_chard', '刀豆': 'sword_bean', '芋头': 'taro', '鸡枞': 'termite_mushroom', '刺苋': 'thorny_amaranth',
     '西红柿': 'tomato', '香椿芽': 'toon', '银耳': 'tremella_fuciformis', '口蘑': 'tricholoma_flavovirens',
     '擘蓝': 'turnip_cabbage', '巢菜': 'vetch', '紫花地丁': 'viola_philippica', '山葵': 'wasabi', '菱角': 'water_caltrop',
     '荸荠': 'water_chestnuts', '西洋菜': 'watercress', '西瓜': 'watermelon', '莼菜': 'water_shield', '空心菜': 'water_spinach',
     '冬瓜': 'wax_gourd', '白茄子': 'white_eggplant', '白萝卜': 'white_radish', '野苋菜': 'wild_amaranth',
     '野菊花': 'wild_chrysanthemum', '乌塌菜': 'Wuta-tsai', '地瓜': 'yam_bean', '榨菜': 'zha-tsai', '茭白': 'zizania_aquatica',
     '西葫芦': 'zucchini'}
    labels = ['茶树菇', '葱', '芦笋', '莴笋', '红豆', '苦瓜', '竹笋', '黑豆', '西兰花', '胡萝卜', '芹菜', '大白菜', '芥蓝', '山药', '佛手瓜', '香菜', '玉米', '黄瓜',
     '黄花菜', '苋菜', '茄子', '金针菇', '茴香', '大蒜', '韭菜', '蒜苔', '大葱', '猴头菇', '油麦菜', '生菜', '莲子', '丝瓜', '绿豆', '芥菜', '秋葵', '洋葱',
     '平菇', '小白菜', '甜椒', '杏鲍菇', '南瓜', '紫甘蓝', '香菇', '毛豆', '黄豆', '黄豆芽', '菠菜', '草莓', '红薯', '芋头', '西红柿', '口蘑', '西瓜', '空心菜',
     '冬瓜', '白茄子', '白萝卜', '榨菜', '西葫芦']
