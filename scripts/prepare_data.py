#!/usr/bin/env python3
"""
prism-lora 数据合成脚本
为 Judge LoRA（记忆冲突检测）和 Poet LoRA（古诗写作）生成训练/验证/测试数据
输出格式: LLaMAFactory sharegpt JSON, 3-round conversations (system + human + gpt)

数据目录结构:
  data/judge/train.json, val.json, test.json
  data/poet/train.json, val.json, test.json
  data/dataset_info.json
"""

import json
import random
import os
import re

# ============================================================
# Judge LoRA: 知识列表与系统提示
# ============================================================

JUDGE_SYSTEM_PROMPT = (
    "你是一个记忆冲突检测专家。给定旧记忆和新事实，你需要判断它们是否在同一维度上存在冲突。"
    "如果冲突则输出UPDATE并用新事实替换旧记忆，如果不冲突则输出KEEP让两条记忆共存。"
    "请以JSON格式输出：{\"decision\": \"UPDATE/KEEP\", \"reason\": \"...\", \"updated_memory\": \"...\"}"
)

PERSONS = [
    "张三", "李四", "王五", "赵六", "钱七",
    "孙八", "周九", "吴十", "郑明", "陈芳",
    "刘伟", "杨丽", "黄强", "林静", "何勇",
    "马云飞", "徐小凤", "朱大成", "秦玉兰", "许志远",
]

OBJECTS = [
    "苹果", "香蕉", "橙子", "西瓜", "葡萄",
    "草莓", "芒果", "樱桃", "桃子", "梨",
    "篮球", "足球", "羽毛球", "乒乓球", "游泳",
    "跑步", "登山", "骑行", "瑜伽", "滑雪",
]

CITIES = [
    "北京", "上海", "广州", "深圳", "成都",
    "杭州", "武汉", "南京", "重庆", "西安",
    "长沙", "青岛", "大连", "厦门", "苏州",
]

COMPANIES = [
    "蚂蚁集团", "阿里巴巴", "腾讯", "百度", "字节跳动",
    "华为", "小米", "京东", "美团", "滴滴",
    "网易", "快手", "拼多多", "比亚迪", "大疆",
]

SCHOOLS = [
    "清华大学", "北京大学", "浙江大学", "复旦大学", "上海交通大学",
    "南京大学", "武汉大学", "中山大学", "四川大学", "哈尔滨工业大学",
    "中国人民大学", "同济大学", "北京航空航天大学", "东南大学", "西安交通大学",
]

LIKES = ["喜欢", "不喜欢", "爱吃", "不爱吃", "擅长", "不擅长", "热爱", "讨厌", "偏好", "厌恶"]
LIKES_FLIP = {
    "喜欢": "不喜欢", "不喜欢": "喜欢",
    "爱吃": "不爱吃", "不爱吃": "爱吃",
    "擅长": "不擅长", "不擅长": "擅长",
    "热爱": "讨厌", "讨厌": "热爱",
    "偏好": "厌恶", "厌恶": "偏好",
}

NUMERIC_VALUES = [
    ("人口", CITIES, ["800万", "1000万", "1200万", "1500万", "2000万", "2200万", "2500万", "3000万"]),
    ("面积", CITIES, ["500平方公里", "800平方公里", "1000平方公里", "1200平方公里", "1500平方公里", "2000平方公里"]),
    ("成立年份", COMPANIES, ["1998年", "2000年", "2003年", "2008年", "2010年", "2012年", "2015年"]),
    ("员工数", COMPANIES, ["5000人", "1万人", "2万人", "5万人", "8万人", "10万人", "15万人"]),
    ("在校生人数", SCHOOLS, ["2万人", "3万人", "4万人", "5万人", "6万人", "8万人"]),
]

SAME_ATTR_DIFF_OBJ = [
    ("喜欢吃", ["苹果", "香蕉", "橙子", "西瓜", "葡萄", "草莓", "芒果", "桃子", "梨"]),
    ("擅长", ["篮球", "足球", "羽毛球", "乒乓球", "游泳", "跑步", "登山", "骑行", "瑜伽"]),
    ("爱好", ["读书", "旅行", "摄影", "绘画", "音乐", "烹饪", "园艺", "钓鱼", "太极"]),
    ("居住在", CITIES),
    ("毕业于", SCHOOLS),
    ("就职于", COMPANIES),
]

DIFF_DOMAIN_PAIRS = [
    ("年龄", "职业"),
    ("爱好", "工作地点"),
    ("学历", "居住地"),
    ("年龄", "爱好"),
    ("体重", "毕业学校"),
    ("身高", "就职公司"),
]


# ============================================================
# Judge LoRA: 数据生成函数 (UPDATE / KEEP 四种类型)
# ============================================================

def generate_update_conflict_samples(n):
    """生成同维度值冲突(喜好反转)和同维度数值更新的UPDATE样本"""
    samples = []

    # 喜好反转: ~80% of n, 数值更新: ~20% of n (5类数值属性)
    n_like = int(n * 0.8)
    for _ in range(n_like):
        person = random.choice(PERSONS)
        like = random.choice(list(LIKES_FLIP.keys()))
        obj = random.choice(OBJECTS[:10])
        old_memory = f"{person}{like}{obj}"
        flipped = LIKES_FLIP[like]
        new_fact = f"{person}{flipped}{obj}"

        input_text = f"旧记忆：{old_memory}\n新事实：{new_fact}\n请判断新事实与旧记忆的关系，并决定处理策略。"
        reason_text = f"新事实与旧记忆在同维度上直接冲突：{person}对{obj}的态度从'{like}'变为'{flipped}'，属于喜好反转，应更新记忆。"

        output_dict = {"decision": "UPDATE", "reason": reason_text, "updated_memory": new_fact}
        output_text = json.dumps(output_dict, ensure_ascii=False)

        samples.append({
            "conversations": [
                {"from": "system", "value": JUDGE_SYSTEM_PROMPT},
                {"from": "human", "value": input_text},
                {"from": "gpt", "value": output_text},
            ]
        })

    # 数值更新: ~20% of n (5类: 人口/面积/成立年份/员工数/在校生)
    n_numeric = n - n_like
    for _ in range(n_numeric):
        attr_name, entities, values = random.choice(NUMERIC_VALUES)
        entity = random.choice(entities)
        old_val = random.choice(values)
        remaining = [v for v in values if v != old_val]
        new_val = random.choice(remaining) if remaining else old_val

        old_memory = f"{entity}{attr_name}{old_val}"
        new_fact = f"{entity}{attr_name}{new_val}"

        input_text = f"旧记忆：{old_memory}\n新事实：{new_fact}\n请判断新事实与旧记忆的关系，并决定处理策略。"
        reason_text = f"新事实对旧记忆中的数值进行了更新：{entity}的{attr_name}从'{old_val}'更新为'{new_val}'，属于数值更新，应更新记忆。"

        output_dict = {"decision": "UPDATE", "reason": reason_text, "updated_memory": new_fact}
        output_text = json.dumps(output_dict, ensure_ascii=False)

        samples.append({
            "conversations": [
                {"from": "system", "value": JUDGE_SYSTEM_PROMPT},
                {"from": "human", "value": input_text},
                {"from": "gpt", "value": output_text},
            ]
        })

    return samples


def generate_update_attribute_samples(n):
    """生成属性反转(擅长/不擅长等)的UPDATE样本"""
    samples = []

    ATTRS_FLIPPABLE = [
        ("擅长", "不擅长"),
        ("精通", "不精通"),
        ("熟悉", "不熟悉"),
        ("了解", "不了解"),
        ("掌握", "未掌握"),
    ]
    SKILL_OBJECTS = ["数学", "编程", "绘画", "音乐", "写作",
                     "游泳", "英语", "演讲", "舞蹈", "钢琴",
                     "书法", "摄影", "烹饪", "中医", "围棋"]

    for _ in range(n):
        person = random.choice(PERSONS)
        pos_attr, neg_attr = random.choice(ATTRS_FLIPPABLE)
        skill = random.choice(SKILL_OBJECTS)

        if random.random() < 0.5:
            old_attr, new_attr = pos_attr, neg_attr
        else:
            old_attr, new_attr = neg_attr, pos_attr

        old_memory = f"{person}{old_attr}{skill}"
        new_fact = f"{person}{new_attr}{skill}"

        input_text = f"旧记忆：{old_memory}\n新事实：{new_fact}\n请判断新事实与旧记忆的关系，并决定处理策略。"
        reason_text = f"新事实与旧记忆在同维度上直接冲突：{person}对{skill}的能力描述从'{old_attr}'变为'{new_attr}'，属于属性反转，应更新记忆。"

        output_dict = {"decision": "UPDATE", "reason": reason_text, "updated_memory": new_fact}
        output_text = json.dumps(output_dict, ensure_ascii=False)

        samples.append({
            "conversations": [
                {"from": "system", "value": JUDGE_SYSTEM_PROMPT},
                {"from": "human", "value": input_text},
                {"from": "gpt", "value": output_text},
            ]
        })

    return samples


def generate_keep_different_dimension_samples(n):
    """生成同属性不同对象(不同维度共存)的KEEP样本"""
    samples = []

    for _ in range(n):
        attr, objects = random.choice(SAME_ATTR_DIFF_OBJ)
        person = random.choice(PERSONS)

        obj1 = random.choice(objects)
        remaining = [o for o in objects if o != obj1]
        obj2 = random.choice(remaining) if remaining else obj1

        old_memory = f"{person}{attr}{obj1}"
        new_fact = f"{person}{attr}{obj2}"

        input_text = f"旧记忆：{old_memory}\n新事实：{new_fact}\n请判断新事实与旧记忆的关系，并决定处理策略。"
        reason_text = f"新事实与旧记忆属于同属性的不同维度：{person}可以同时'{attr}{obj1}'和'{attr}{obj2}'，两者并不冲突，应保持共存。"

        output_dict = {"decision": "KEEP", "reason": reason_text, "updated_memory": f"{old_memory}；{new_fact}"}
        output_text = json.dumps(output_dict, ensure_ascii=False)

        samples.append({
            "conversations": [
                {"from": "system", "value": JUDGE_SYSTEM_PROMPT},
                {"from": "human", "value": input_text},
                {"from": "gpt", "value": output_text},
            ]
        })

    return samples


def generate_keep_different_domain_samples(n):
    """生成完全不同领域共存(不同领域)的KEEP样本"""
    samples = []

    DOMAIN_A_VALUES = [
        ("年龄", ["25岁", "28岁", "30岁", "35岁", "40岁", "45岁", "50岁"]),
        ("体重", ["60公斤", "65公斤", "70公斤", "75公斤", "80公斤"]),
        ("身高", ["170厘米", "175厘米", "180厘米", "185厘米"]),
        ("学历", ["本科", "硕士", "博士", "大专"]),
    ]
    DOMAIN_B_VALUES = [
        ("职业", ["工程师", "教师", "医生", "律师", "设计师", "程序员", "研究员"]),
        ("工作地点", CITIES),
        ("就职于", COMPANIES),
        ("毕业于", SCHOOLS),
        ("爱好", ["读书", "旅行", "摄影", "绘画", "音乐", "烹饪"]),
    ]

    for _ in range(n):
        person = random.choice(PERSONS)
        domain_a_name, domain_a_vals = random.choice(DOMAIN_A_VALUES)
        domain_b_name, domain_b_vals = random.choice(DOMAIN_B_VALUES)

        val_a = random.choice(domain_a_vals)
        val_b = random.choice(domain_b_vals)

        old_memory = f"{person}{domain_a_name}{val_a}"
        new_fact = f"{person}{domain_b_name}{val_b}"

        input_text = f"旧记忆：{old_memory}\n新事实：{new_fact}\n请判断新事实与旧记忆的关系，并决定处理策略。"
        reason_text = f"新事实与旧记忆属于完全不同的领域：'{domain_a_name}'和'{domain_b_name}'互不干扰，两者可以共存，应保持旧记忆不变。"

        output_dict = {"decision": "KEEP", "reason": reason_text, "updated_memory": f"{old_memory}；{new_fact}"}
        output_text = json.dumps(output_dict, ensure_ascii=False)

        samples.append({
            "conversations": [
                {"from": "system", "value": JUDGE_SYSTEM_PROMPT},
                {"from": "human", "value": input_text},
                {"from": "gpt", "value": output_text},
            ]
        })

    return samples


# ============================================================
# Poet LoRA: 系统提示与配置
# ============================================================

POET_SYSTEM_PROMPT = (
    "你是一位精通古诗词的创作大师，擅长根据要求创作符合格律和意境的古典诗词。"
    "你的创作严格遵守古典诗词的体裁规范，包括字数、行数和押韵。"
)

POETRY_FORMS = ["五言绝句", "七言绝句", "五言律诗", "七言律诗"]

FORM_SPEC = {
    "五言绝句": {"lines": 4, "chars_per_line": 5},
    "七言绝句": {"lines": 4, "chars_per_line": 7},
    "五言律诗": {"lines": 8, "chars_per_line": 5},
    "七言律诗": {"lines": 8, "chars_per_line": 7},
}

TOPICS = [
    "春雨", "秋月", "山水", "离别", "思乡", "登高", "夜思",
    "梅花", "荷花", "柳树", "春风", "秋霜", "雪景", "落日",
    "污水", "渔舟", "田园", "边塞", "故人", "归途",
    "月夜", "寒夜", "晨曦", "暮色", "远行", "怀古",
    "照湖", "听雨", "咏竹", "望远",
]

POETS_TO_IMITATE = [
    "杜甫", "李白", "王维", "白居易", "苏轼", "陆游",
    "孟浩然", "刘禹锡", "柳宗元", "韩愈", "贺知章", "王昌龄",
]

INSTRUCTION_TEMPLATES = [
    "写一首{form}，以{topic}为题。",
    "请创作一首{form}，描写{topic}的意境。",
    "以{topic}为主题，创作一首{form}。要求意境深远。",
    "模仿{poet}的风格，写一首关于{topic}的{form}。",
    "请以{poet}的笔触，描绘{topic}，体裁为{form}。",
    "创作{form}一首，咏{topic}。",
    "用{form}的形式，表达对{topic}的感受。",
    "以{topic}入诗，作{form}一首。",
]

# ============================================================
# Poet LoRA: 真实古诗数据库 (从参考实现复制, 98首经典古诗)
# ============================================================

POEMS_DB = {
    "五言绝句": [
        {"title": "春晓", "author": "孟浩然", "text": "春眠不觉晓，处处闻啼鸟。\n夜来风雨声，花落知多少。"},
        {"title": "登鹳雀楼", "author": "王之涣", "text": "白日依山尽，黄河入海流。\n欲穷千里目，更上一层楼。"},
        {"title": "静夜思", "author": "李白", "text": "床前明月光，疑是地上霜。\n举头望明月，低头思故乡。"},
        {"title": "鹿柴", "author": "王维", "text": "空山不见人，但闻人语响。\n返景入深林，复照青苔上。"},
        {"title": "相思", "author": "王维", "text": "红豆生南国，春来发几枝。\n愿君多采撷，此物最相思。"},
        {"title": "竹里馆", "author": "王维", "text": "独坐幽篁里，弹琴复长啸。\n深林人不知，明月来相照。"},
        {"title": "送别", "author": "王维", "text": "山中相送罢，日暮掩柴扉。\n春草明年绿，王孙归不归。"},
        {"title": "江雪", "author": "柳宗元", "text": "千山鸟飞绝，万径人踪灭。\n孤舟蓑笠翁，独钓寒江雪。"},
        {"title": "秋夜寄邱员外", "author": "韦应物", "text": "怀君属秋夜，散步咏凉天。\n空山松子落，幽人应未眠。"},
        {"title": "送灵澈上人", "author": "刘长卿", "text": "苍苍竹林寺，杳杳钟声晚。\n荷笠带夕阳，青山独归远。"},
        {"title": "独坐敬亭山", "author": "李白", "text": "众鸟高飞尽，孤云独去闲。\n相看两不厌，只有敬亭山。"},
        {"title": "八阵图", "author": "杜甫", "text": "功盖三分国，名成八阵图。\n江流石不转，遗恨失吞吴。"},
        {"title": "寻隐者不遇", "author": "贾岛", "text": "松下问童子，言师采药去。\n只在此山中，云深不知处。"},
        {"title": "宿建德江", "author": "孟浩然", "text": "移舟泊烟渚，日暮客愁新。\n野旷天低树，江清月近人。"},
        {"title": "秋浦歌", "author": "李白", "text": "白发三千丈，缘愁似个长。\n不知明镜里，何处得秋霜。"},
        {"title": "夜宿山寺", "author": "李白", "text": "危楼高百尺，手可摘星辰。\n不敢高声语，恐惊天上人。"},
        {"title": "悯农", "author": "李绅", "text": "锄禾日当午，汗滴禾下土。\n谁知盘中餐，粒粒皆辛苦。"},
        {"title": "悯农其二", "author": "李绅", "text": "春种一粒粟，秋收万颗子。\n四海无闲田，农夫犹饿死。"},
        {"title": "塞下曲", "author": "卢纶", "text": "月黑雁飞高，单于夜遁逃。\n欲将轻骑逐，大雪满弓刀。"},
        {"title": "归园田居", "author": "陶渊明", "text": "种豆南山下，草盛豆苗稀。\n晨兴理荒秽，带月荷锄归。"},
        {"title": "蝉", "author": "虞世南", "text": "垂緌饮清露，流响出疏桐。\n居高声自远，非是藉秋风。"},
        {"title": "风", "author": "李峤", "text": "解落三秋叶，能开二月花。\n过江千尺浪，入竹万竿斜。"},
        {"title": "杂诗", "author": "王维", "text": "君自故乡来，应知故乡事。\n来日绮窗前，寒梅著花未？"},
        {"title": "鸟鸣涧", "author": "王维", "text": "人闲桂花落，夜静春山空。\n月出惊山鸟，时鸣春涧中。"},
        {"title": "池上", "author": "白居易", "text": "小娃撑小艇，偷采白莲回。\n不解藏踪迹，浮萍一道开。"},
        {"title": "问刘十九", "author": "白居易", "text": "绿蚁新醅酒，红泥小火炉。\n晚来天欲雪，能饮一杯无？"},
        {"title": "遗爱寺", "author": "白居易", "text": "弄石临溪坐，寻花绕寺行。\n时时闻鸟语，处处是泉声。"},
        {"title": "新嫁娘词", "author": "王建", "text": "三日入厨下，洗手作羹汤。\n未谙姑食性，先遣小姑尝。"},
        {"title": "行宫", "author": "元稹", "text": "寥落古行宫，宫花寂寞红。\n白头宫女在，闲坐说玄宗。"},
        {"title": "登乐游原", "author": "李商隐", "text": "向晚意不适，驱车登古原。\n夕阳无限好，只是近黄昏。"},
        {"title": "夜雨", "author": "白居易", "text": "早蛩啼露歇，残灯栖壁寒。\n只嫌秋夜长，无梦到江南。"},
    ],
    "七言绝句": [
        {"title": "望庐山瀑布", "author": "李白", "text": "日照香炉生紫烟，遥看瀑布挂前川。\n飞流直下三千尺，疑是银河落九天。"},
        {"title": "枫桥夜泊", "author": "张继", "text": "月落乌啼霜满天，江枫渔火对愁眠。\n姑苏城外寒山寺，夜半钟声到客船。"},
        {"title": "望天门山", "author": "李白", "text": "天门中断楚江开，碧水东流至此回。\n两岸青山相对出，孤帆一片日边来。"},
        {"title": "早发白帝城", "author": "李白", "text": "朝辞白帝彩云间，千里江陵一日还。\n两岸猿声啼不住，轻舟已过万重山。"},
        {"title": "绝句", "author": "杜甫", "text": "两个黄鹂鸣翠柳，一行白鹭上青天。\n窗含西岭千秋雪，门泊东吴万里船。"},
        {"title": "清明", "author": "杜牧", "text": "清明时节雨纷纷，路上行人欲断魂。\n借问酒家何处有？牧童遥指杏花村。"},
        {"title": "山行", "author": "杜牧", "text": "远上寒山石径斜，白云生处有人家。\n停车坐爱枫林晚，霜叶红于二月花。"},
        {"title": "泊秦淮", "author": "杜牧", "text": "烟笼寒水月笼沙，夜泊秦淮近酒家。\n商女不知亡国恨，隔江犹唱后庭花。"},
        {"title": "秋夕", "author": "杜牧", "text": "银烛秋光冷画屏，轻罗小扇扑流萤。\n天阶夜色凉如水，坐看牵牛织女星。"},
        {"title": "江南春", "author": "杜牧", "text": "千里莺啼绿映红，水村山郭酒旗风。\n南朝四百八十寺，多少楼台烟雨中。"},
        {"title": "黄鹤楼送孟浩然之广陵", "author": "李白", "text": "故人西辞黄鹤楼，烟花三月下扬州。\n孤帆远影碧空尽，唯见长江天际流。"},
        {"title": "送元二使安西", "author": "王维", "text": "渭城朝雨浥轻尘，客舍青青柳色新。\n劝君更尽一杯酒，西出阳关无故人。"},
        {"title": "出塞", "author": "王昌龄", "text": "秦时明月汉时关，万里长征人未还。\n但使龙城飞将在，不教胡马度阴山。"},
        {"title": "凉州词", "author": "王翰", "text": "葡萄美酒夜光杯，欲饮琵琶马上催。\n醉卧沙场君莫笑，古来征战几人回？"},
        {"title": "登飞来峰", "author": "王安石", "text": "飞来山上千寻塔，闻说鸡鸣见日升。\n不畏浮云遮望眼，自缘身在最高层。"},
        {"title": "题西林壁", "author": "苏轼", "text": "横看成岭侧成峰，远近高低各不同。\n不识庐山真面目，只缘身在此山中。"},
        {"title": "饮湖上初晴后雨", "author": "苏轼", "text": "水光潋滟晴方好，山色空蒙雨亦奇。\n欲把西湖比西子，淡妆浓抹总相宜。"},
        {"title": "春日", "author": "朱熹", "text": "胜日寻芳泗水滨，无边光景一时新。\n等闲识得东风面，万紫千红总是春。"},
        {"title": "惠崇春江晚景", "author": "苏轼", "text": "竹外桃花三两枝，春江水暖鸭先知。\n蒌蒿满地芦芽短，正是河豚欲上时。"},
        {"title": "回乡偶书", "author": "贺知章", "text": "少小离家老大回，乡音无改鬓毛衰。\n儿童相见不相识，笑问客从何处来。"},
        {"title": "咏柳", "author": "贺知章", "text": "碧玉妆成一树高，万条垂下绿丝绦。\n不知细叶谁裁出，二月春风似剪刀。"},
        {"title": "从军行", "author": "王昌龄", "text": "青海长云暗雪山，孤城遥望玉门关。\n黄沙百战穿金甲，不破楼兰终不还。"},
        {"title": "从军行其二", "author": "王昌龄", "text": "大漠风尘日色昏，红旗半卷出辕门。\n前军夜战洮河北，已报生擒吐谷浑。"},
        {"title": "竹枝词", "author": "刘禹锡", "text": "杨柳青青江水平，闻郎江上踏歌声。\n东边日出西边雨，道是无晴却有晴。"},
        {"title": "乌衣巷", "author": "刘禹锡", "text": "朱雀桥边野草花，乌衣巷口夕阳斜。\n旧时王谢堂前燕，飞入寻常百姓家。"},
        {"title": "秋思", "author": "张籍", "text": "洛阳城里见秋风，欲作家书意万重。\n复恐匆匆说不尽，行人临发又开封。"},
        {"title": "秋词", "author": "刘禹锡", "text": "自古逢秋悲寂寥，我言秋日胜春朝。\n晴空一鹤排云上，便引诗情到碧霄。"},
        {"title": "夜上受降城闻笛", "author": "李益", "text": "回乐烽前沙似雪，受降城外月如霜。\n不知何处吹芦管，一夜征人尽望乡。"},
        {"title": "金陵酒肆留别", "author": "李白", "text": "风吹柳花满店香，吴姬压酒唤客尝。\n金陵子弟来相送，欲行不行各尽觞。"},
        {"title": "赠花卿", "author": "杜甫", "text": "锦城丝管日纷纷，半入江风半入云。\n此曲只应天上有，人间能得几回闻。"},
        {"title": "征人怨", "author": "柳中庸", "text": "岁岁金河复玉关，朝朝马策与刀环。\n三春白雪归青冢，万里黄河绕黑山。"},
    ],
    "五言律诗": [
        {"title": "春望", "author": "杜甫", "text": "国破山河在，城春草木深。\n感时花溅泪，恨别鸟惊心。\n烽火连三月，家书抵万金。\n白头搔更短，浑欲不胜簪。"},
        {"title": "月夜", "author": "杜甫", "text": "今夜鄜州月，闺中只独看。\n遥怜小儿女，未解忆长安。\n香雾云鬟湿，清辉玉臂寒。\n何时倚虚幌，双照泪痕干。"},
        {"title": "山居秋暝", "author": "王维", "text": "空山新雨后，天气晚来秋。\n明月松间照，清泉石上流。\n竹喧归浣女，莲动下渔舟。\n随意春芳歇，王孙自可留。"},
        {"title": "送友人", "author": "李白", "text": "青山横北郭，白水绕东城。\n此地一为别，孤蓬万里征。\n浮云游子意，落日故人情。\n挥手自兹去，萧萧班马鸣。"},
        {"title": "过故人庄", "author": "孟浩然", "text": "故人具鸡黍，邀我至田家。\n绿树村边合，青山郭外斜。\n开轩面场圃，把酒话桑麻。\n待到重阳日，还来就菊花。"},
        {"title": "使至塞上", "author": "王维", "text": "单车欲问边，属国过居延。\n征蓬出汉塞，归雁入胡天。\n大漠孤烟直，长河落日圆。\n萧关逢候骑，都护在燕然。"},
        {"title": "归嵩山作", "author": "王维", "text": "清川带长薄，车马去闲闲。\n流水如有意，暮禽相与还。\n荒城临古渡，落日满秋山。\n迢递嵩高下，归来且闭关。"},
        {"title": "终南别业", "author": "王维", "text": "中岁颇好道，晚家南山陲。\n兴来每独往，胜事空自知。\n行到水穷处，坐看云起时。\n偶然值林叟，谈笑无还期。"},
        {"title": "望月怀远", "author": "张九龄", "text": "海上生明月，天涯共此时。\n情人怨遥夜，竟夕起相思。\n灭烛怜光满，披衣觉露滋。\n不堪盈手赠，还寝梦佳期。"},
        {"title": "与诸子登岘山", "author": "孟浩然", "text": "人事有代谢，往来成古今。\n江山留胜迹，我辈复登临。\n水落鱼梁浅，天寒梦泽深。\n羊公碑尚在，读罢泪沾襟。"},
        {"title": "宴梅道士山房", "author": "孟浩然", "text": "林卧愁春尽，开轩览物华。\n忽逢青鸟使，邀入赤松家。\n丹灶初开火，仙桃正发花。\n童颜若可驻，何惜醉流霞。"},
        {"title": "岁暮归南山", "author": "孟浩然", "text": "北阙休上书，南山归敝庐。\n不才明主弃，多病故人疏。\n白发催年老，青阳逼岁除。\n永怀愁不寐，松月夜窗虚。"},
        {"title": "宿业师山房期丁大不至", "author": "孟浩然", "text": "夕阳度西岭，群壑倏已暝。\n松月生夜凉，风泉满清听。\n樵人归欲尽，烟鸟栖初定。\n之子期宿来，孤琴候萝径。"},
        {"title": "宿桐庐江寄广陵旧游", "author": "孟浩然", "text": "山暝听猿愁，沧江急夜流。\n风鸣两岸叶，月照一孤舟。\n建德非吾土，维扬忆旧游。\n还将两行泪，遥寄海西头。"},
        {"title": "题大庾岭北驿", "author": "宋之问", "text": "阳月南飞雁，传闻至此回。\n我行殊未已，何日复归来。\n江静潮初落，林昏瘴不开。\n明朝望乡处，应见陇头梅。"},
        {"title": "次北固山下", "author": "王湾", "text": "客路青山外，行舟绿水前。\n潮平两岸阔，风正一帆悬。\n海日生残夜，江春入旧年。\n乡书何处达？归雁洛阳边。"},
        {"title": "破山寺后禅院", "author": "常建", "text": "清晨入古寺，初日照高林。\n曲径通幽处，禅房花木深。\n山光悦鸟性，潭影空人心。\n万籁此皆寂，惟闻钟磬音。"},
    ],
    "七言律诗": [
        {"title": "蜀相", "author": "杜甫", "text": "丞相祠堂何处寻，锦官城外柏森森。\n映阶碧草自春色，隔叶黄鹂空好音。\n三顾频烦天下计，两朝开济老臣心。\n出师未捷身先死，长使英雄泪满襟。"},
        {"title": "登高", "author": "杜甫", "text": "风急天高猿啸哀，渚清沙白鸟飞回。\n无边落木萧萧下，不尽长江滚滚来。\n万里悲秋常作客，百年多病独登台。\n艰难苦恨繁霜鬓，潦倒新停浊酒杯。"},
        {"title": "客至", "author": "杜甫", "text": "舍南舍北皆春水，但见群鸥日日来。\n花径不曾缘客扫，蓬门今始为君开。\n盘飧市远无兼味，樽酒家贫只旧醅。\n肯与邻翁相对饮，隔篱呼取尽余杯。"},
        {"title": "闻官军收河南河北", "author": "杜甫", "text": "剑外忽传收蓟北，初闻涕泪满衣裳。\n却看妻子愁何在，漫卷诗书喜欲狂。\n白日放歌须纵酒，青春作伴好还乡。\n即从巴峡穿巫峡，便下襄阳向洛阳。"},
        {"title": "无题", "author": "李商隐", "text": "相见时难别亦难，东风无力百花残。\n春蚕到死丝方尽，蜡炬成灰泪始干。\n晓镜但愁云鬓改，夜吟应觉月光寒。\n蓬山此去无多路，青鸟殷勤为探看。"},
        {"title": "钱塘湖春行", "author": "白居易", "text": "孤山寺北贾亭西，水面初平云脚低。\n几处早莺争暖树，谁家新燕啄春泥。\n乱花渐欲迷人眼，浅草才能没马蹄。\n最爱湖东行不足，绿杨阴里白沙堤。"},
        {"title": "望蓟门", "author": "祖咏", "text": "燕台一望客心惊，箬鼓喧喧汉将营。\n万里寒光生积雪，三边曙色动危旌。\n沙场烽火侵胡月，海畔云山拥蓟城。\n少小虽非投笔吏，论功还欲请长缨。"},
        {"title": "九日齐山登高", "author": "杜牧", "text": "江涵秋影雁初飞，与客携壶上翠微。\n尘世难逢开口笑，菊花须插满头归。\n但将酩酊酬佳节，不用登临叹落晖。\n古往今来只如此，牛山何必独沾衣。"},
        {"title": "登柳州城楼寄漳汀封连四州", "author": "柳宗元", "text": "城上高楼接大荒，海天愁思正茫茫。\n惊风乱飐芙蓉水，密雨斜侵薜荔墙。\n岭树重遮千里目，江流曲似九回肠。\n共来百越文身地，犹自音书滞一乡。"},
        {"title": "西塞山怀古", "author": "刘禹锡", "text": "王濬楼船下益州，金陵王气黯然收。\n千寻铁锁沉江底，一片降幡出石头。\n人世几回伤往事，山形依旧枕寒流。\n从今四海为家日，故垒萧萧芦荻秋。"},
        {"title": "利州南渡", "author": "温庭筠", "text": "澹然空水对斜晖，曲岛苍茫接翠微。\n波上马嘶看棹去，柳边人歇待船归。\n数丛沙草群鸥散，万顷江田一鹭飞。\n谁解乘舟寻范蠡，五湖烟水独忘机。"},
    ],
}


# ============================================================
# Poet LoRA: 诗歌体裁检测与指令生成
# ============================================================

def detect_poetry_form(text: str) -> str | None:
    """Detect poetry form from text."""
    lines = []
    for line in text.strip().split("\n"):
        # Remove punctuation and whitespace for counting
        cleaned = re.sub(r'[，。！？；：、""''（）《》\s]', '', line.strip())
        if cleaned and len(cleaned) >= 3:
            lines.append(cleaned)

    if not lines:
        return None

    n_lines = len(lines)
    avg_chars = sum(len(l) for l in lines) / len(lines)

    if n_lines == 4 and avg_chars <= 5.5:
        return "五言绝句"
    elif n_lines == 4 and avg_chars >= 6.5:
        return "七言绝句"
    elif n_lines == 8 and avg_chars <= 5.5:
        return "五言律诗"
    elif n_lines == 8 and avg_chars >= 6.5:
        return "七言律诗"
    elif n_lines == 4:
        if sum(1 for l in lines if len(l) == 5) >= 2:
            return "五言绝句"
        elif sum(1 for l in lines if len(l) == 7) >= 2:
            return "七言绝句"
    elif n_lines == 8:
        if sum(1 for l in lines if len(l) == 5) >= 4:
            return "五言律诗"
        elif sum(1 for l in lines if len(l) == 7) >= 4:
            return "七言律诗"

    return None


def generate_instruction(poem: dict, template_idx: int = None, topic_override: str = None) -> str:
    """Generate an instruction prompt for a given poem."""
    form = poem["form"]

    # Topic selection
    if topic_override:
        topic = topic_override
    else:
        title = poem.get("title", "")
        text = poem.get("text", "")
        matched_topics = [t for t in TOPICS if t in title or t in text]

        if matched_topics:
            topic = random.choice(matched_topics)
        else:
            topic = random.choice(TOPICS)

    # Poet to imitate: use original author if available, otherwise random
    author = poem.get("author", "")
    poet_choice = author if author in POETS_TO_IMITATE and random.random() < 0.4 else random.choice(POETS_TO_IMITATE)

    if template_idx is not None:
        template = INSTRUCTION_TEMPLATES[template_idx % len(INSTRUCTION_TEMPLATES)]
    else:
        template = random.choice(INSTRUCTION_TEMPLATES)

    return template.format(form=form, topic=topic, poet=poet_choice)


# ============================================================
# Poet LoRA: 数据生成
# ============================================================

def generate_poet_samples():
    """从 POEMS_DB 生成古诗写作的训练样本: 每首诗×8模板+N随机变体"""
    # Build poem pool with detected forms
    all_poems = []
    for form, poems in POEMS_DB.items():
        for poem in poems:
            form_detected = detect_poetry_form(poem["text"])
            if form_detected is None:
                form_detected = form
            all_poems.append({
                "title": poem["title"],
                "author": poem["author"],
                "text": poem["text"],
                "form": form_detected,
            })

    target_total = 2000 + 200 + 300  # train + val + test
    n_per_poem = 30  # enough variants to reach 2500 total

    records = []
    for poem in all_poems:
        # 8 template variations (one per template)
        for t_idx in range(len(INSTRUCTION_TEMPLATES)):
            instruction = generate_instruction(poem, template_idx=t_idx)
            response = f"{poem['title']}\n\n{poem['text']}"
            records.append({
                "conversations": [
                    {"from": "system", "value": POET_SYSTEM_PROMPT},
                    {"from": "human", "value": instruction},
                    {"from": "gpt", "value": response},
                ]
            })

        # Additional random variations (different topics per same poem)
        additional = n_per_poem - len(INSTRUCTION_TEMPLATES)
        for _ in range(additional):
            instruction = generate_instruction(poem, template_idx=None)
            response = f"{poem['title']}\n\n{poem['text']}"
            records.append({
                "conversations": [
                    {"from": "system", "value": POET_SYSTEM_PROMPT},
                    {"from": "human", "value": instruction},
                    {"from": "gpt", "value": response},
                ]
            })

    random.shuffle(records)
    return records


# ============================================================
# Judge LoRA: 数据生成 (组合四种类型)
# ============================================================

def generate_judge_samples(total, seed_value):
    """生成指定数量的 Judge LoRA 样本，保持各类型比例"""
    random.seed(seed_value)

    n_conflict = int(total * 0.40)
    n_attribute = int(total * 0.10)
    n_diff_dim = int(total * 0.30)
    n_diff_domain = total - n_conflict - n_attribute - n_diff_dim

    samples = []
    samples.extend(generate_update_conflict_samples(n_conflict))
    samples.extend(generate_update_attribute_samples(n_attribute))
    samples.extend(generate_keep_different_dimension_samples(n_diff_dim))
    samples.extend(generate_keep_different_domain_samples(n_diff_domain))

    random.shuffle(samples)
    return samples


# ============================================================
# dataset_info.json
# ============================================================

DATASET_INFO = {
    "judge_train": {"file_name": "judge/train.json", "formatting": "sharegpt"},
    "judge_val": {"file_name": "judge/val.json", "formatting": "sharegpt"},
    "poet_train": {"file_name": "poet/train.json", "formatting": "sharegpt"},
    "poet_val": {"file_name": "poet/val.json", "formatting": "sharegpt"},
}


# ============================================================
# 主函数
# ============================================================

def main():
    random.seed(42)

    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    judge_dir = os.path.join(base_dir, "judge")
    poet_dir = os.path.join(base_dir, "poet")

    # Create subdirectories
    os.makedirs(judge_dir, exist_ok=True)
    os.makedirs(poet_dir, exist_ok=True)

    # ── Judge LoRA data ──
    print("=== Judge LoRA Data Generation ===")

    judge_train = generate_judge_samples(2000, seed_value=42)
    judge_val = generate_judge_samples(200, seed_value=123)
    judge_test = generate_judge_samples(300, seed_value=456)

    for name, data, path in [
        ("train", judge_train, os.path.join(judge_dir, "train.json")),
        ("val", judge_val, os.path.join(judge_dir, "val.json")),
        ("test", judge_test, os.path.join(judge_dir, "test.json")),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  judge/{name}.json: {len(data)} samples")

    update_count = sum(1 for s in judge_train if '"decision": "UPDATE"' in s["conversations"][2]["value"])
    keep_count = sum(1 for s in judge_train if '"decision": "KEEP"' in s["conversations"][2]["value"])
    print(f"  Judge train decisions: UPDATE={update_count}, KEEP={keep_count}")

    # ── Poet LoRA data ──
    print("\n=== Poet LoRA Data Generation ===")
    random.seed(42)

    poet_records = generate_poet_samples()

    n_train = min(2000, len(poet_records))
    n_val = min(200, len(poet_records) - n_train)
    n_test = min(300, len(poet_records) - n_train - n_val)

    poet_train = poet_records[:n_train]
    poet_val = poet_records[n_train:n_train + n_val]
    poet_test = poet_records[n_train + n_val:]

    for name, data, path in [
        ("train", poet_train, os.path.join(poet_dir, "train.json")),
        ("val", poet_val, os.path.join(poet_dir, "val.json")),
        ("test", poet_test, os.path.join(poet_dir, "test.json")),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  poet/{name}.json: {len(data)} samples")

    # Form distribution in poet train
    form_counts = {}
    for r in poet_train:
        instruction = r["conversations"][1]["value"]
        for form in POETRY_FORMS:
            if form in instruction:
                form_counts[form] = form_counts.get(form, 0) + 1
                break
    print(f"  Poet train form distribution: {form_counts}")

    # ── dataset_info.json ──
    info_path = os.path.join(base_dir, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(DATASET_INFO, f, ensure_ascii=False, indent=2)
    print(f"\n  dataset_info.json saved to {info_path}")

    # ── Remove old flat data files ──
    old_files = [
        os.path.join(base_dir, "judge_train.json"),
        os.path.join(base_dir, "judge_test.json"),
        os.path.join(base_dir, "poet_train.json"),
        os.path.join(base_dir, "poet_test.json"),
    ]
    removed = []
    for old_path in old_files:
        if os.path.exists(old_path):
            os.remove(old_path)
            removed.append(old_path)
    if removed:
        print(f"\n  Removed old flat files: {[os.path.basename(p) for p in removed]}")

    # ── Sample inspection ──
    print("\n--- Sample Inspection ---")
    if judge_train:
        sample = judge_train[0]
        print(f"Judge sample (first 3 rounds):")
        for turn in sample["conversations"]:
            role = turn["from"]
            val_preview = turn["value"][:80] + "..." if len(turn["value"]) > 80 else turn["value"]
            print(f"  {role}: {val_preview}")

    if poet_train:
        sample = poet_train[0]
        print(f"\nPoet sample (first 3 rounds):")
        for turn in sample["conversations"]:
            role = turn["from"]
            val_preview = turn["value"][:80] + "..." if len(turn["value"]) > 80 else turn["value"]
            print(f"  {role}: {val_preview}")

    # ── Summary ──
    print("\n--- Data Generation Summary ---")
    print(f"Judge LoRA: {len(judge_train)} train + {len(judge_val)} val + {len(judge_test)} test = {len(judge_train) + len(judge_val) + len(judge_test)} total")
    print(f"Poet LoRA:  {len(poet_train)} train + {len(poet_val)} val + {len(poet_test)} test = {len(poet_train) + len(poet_val) + len(poet_test)} total")


if __name__ == "__main__":
    main()
