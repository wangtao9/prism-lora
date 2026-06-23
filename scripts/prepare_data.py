#!/usr/bin/env python3
"""
prism-lora 数据合成脚本
为 Judge LoRA（记忆冲突检测）和 Poet LoRA（古诗写作）生成训练/测试数据
输出格式: LLaMAFactory sharegpt JSON
"""

import json
import random
import os

# ============================================================
# 知识列表
# ============================================================

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

# ============================================================
# 古诗种子数据 (7首经典古诗)
# ============================================================

POETRY_DATA = [
    {
        "prompt": "请写一首关于春天的七言绝句，风格要求：清新明快。",
        "response": (
            "诗题：春晓\n\n"
            "春眠不觉晓，\n处处闻啼鸟。\n夜来风雨声，\n花落知多少。\n\n"
            "赏析：此诗以春晨为景，写出春日睡醒不觉天明的慵懒，"
            "鸟鸣声声入耳的生机，以及夜雨过后落花满地的惋惜。"
            "全诗语言朴实自然，意境清新，是唐诗中描写春天的经典之作。"
        ),
    },
    {
        "prompt": "请写一首关于月亮的五言绝句，风格要求：宁静悠远。",
        "response": (
            "诗题：静夜思\n\n"
            "床前明月光，\n疑是地上霜。\n举头望明月，\n低头思故乡。\n\n"
            "赏析：此诗以月光起兴，由《疑是地上霜》的错觉写出夜之清冷，"
            "由望月而思乡，将羁旅之愁写得质朴深情。"
            "语言简练而意境深远，千百年来为游子共吟。"
        ),
    },
    {
        "prompt": "请写一首关于山水的五言绝句，风格要求：空灵淡远。",
        "response": (
            "诗题：鹿柴\n\n"
            "空山不见人，\n但闻人语响。\n返景入深林，\n复照青苔上。\n\n"
            "赏析：此诗以《空山》起笔，写出深山的幽静与孤独，"
            "人语的回响衬托山的空寂，夕阳余光穿过深林照在青苔之上，"
            "光影交错间尽显禅意与超然。"
        ),
    },
    {
        "prompt": "请写一首关于思乡的七言绝句，风格要求：深沉感慨。",
        "response": (
            "诗题：回乡偶书\n\n"
            "少小离家老大回，\n乡音无改鬓毛衰。\n儿童相见不相识，\n笑问客从何处来。\n\n"
            "赏析：此诗写久客还乡的感慨，少小离家与老大归来对比强烈，"
            "乡音未改而鬓毛已衰，儿童不识而笑问客来，"
            "将岁月流逝与身份变化写得含蓄深沉，令人动容。"
        ),
    },
    {
        "prompt": "请写一首关于边塞的七言绝句，风格要求：雄壮悲凉。",
        "response": (
            "诗题：出塞\n\n"
            "秦时明月汉时关，\n万里长征人未还。\n但使龙城飞将在，\n不教胡马度阴山。\n\n"
            "赏析：此诗以秦汉明月边关起兴，写出万里征人不归的悲凉，"
            "又以《但使龙城飞将在》的期许表达对良将的渴望，"
            "气势雄壮而情感悲凉，是边塞诗中的千古名作。"
        ),
    },
    {
        "prompt": "请写一首关于离别送行的七言绝句，风格要求：深情眷恋。",
        "response": (
            "诗题：送元二使安西\n\n"
            "渭城朝雨浥轻尘，\n客舍青青柳色新。\n劝君更尽一杯酒，\n西出阳关无故人。\n\n"
            "赏析：此诗以渭城朝雨柳色新写出离别时的清新景象，"
            "劝酒之举含蓄深挚，《西出阳关无故人》将远行的孤独与眷恋推向高潮，"
            "成为千古送别诗的典范之作。"
        ),
    },
    {
        "prompt": "请写一首关于秋天的七言绝句，风格要求：萧疏凄清。",
        "response": (
            "诗题：秋夕\n\n"
            "银烛秋光冷画屏，\n轻罗小扇扑流萤。\n天阶夜色凉如水，\n坐看牵牛织女星。\n\n"
            "赏析：此诗以《银烛秋光冷画屏》写出秋夜的清冷华美，"
            "扑流萤的细微动作写出宫人的无聊与寂寞，"
            "夜色如水与望星的意象将孤寂推向极致，凄清萧疏，意蕴悠长。"
        ),
    },
]

POETRY_TOPICS = [
    "春天", "月亮", "山水", "思乡", "边塞",
    "离别", "秋天", "梅花", "雪", "酒",
    "荷花", "长江", "黄河", "日落", "清晨",
    "夜雨", "独居", "田园", "战争", "岁月",
]

POETRY_STYLES = [
    "清新明快", "宁静悠远", "空灵淡远", "深沉感慨",
    "雄壮悲凉", "深情眷恋", "萧疏凄清", "婉约柔美",
    "豪放奔逸", "含蓄蕴藉", "质朴自然", "典雅端庄",
]

POETRY_FORMS = [
    ("五言绝句", 5, 4),
    ("七言绝句", 7, 4),
    ("五言律诗", 5, 8),
    ("七言律诗", 7, 8),
]

CLASSIC_LINES_5 = [
    "空山不见人", "但闻人语响", "返景入深林", "复照青苔上",
    "床前明月光", "疑是地上霜", "举头望明月", "低头思故乡",
    "春眠不觉晓", "处处闻啼鸟", "夜来风雨声", "花落知多少",
    "白日依山尽", "黄河入海流", "欲穷千里目", "更上一层楼",
    "松下问童子", "言师采药去", "只在此山中", "云深不知处",
    "千山鸟飞绝", "万径人踪灭", "孤舟蓑笠翁", "独钓寒江雪",
    "日出江花红", "春来江水绿", "能不忆江南", "风景旧曾谙",
    "移舟泊烟渚", "日暮客愁新", "野旷天低树", "江清月近人",
]

CLASSIC_LINES_7 = [
    "秦时明月汉时关", "万里长征人未还", "但使龙城飞将在", "不教胡马度阴山",
    "渭城朝雨浥轻尘", "客舍青青柳色新", "劝君更尽一杯酒", "西出阳关无故人",
    "银烛秋光冷画屏", "轻罗小扇扑流萤", "天阶夜色凉如水", "坐看牵牛织女星",
    "少小离家老大回", "乡音无改鬓毛衰", "儿童相见不相识", "笑问客从何处来",
    "日照香炉生紫烟", "遥看瀑布挂前川", "飞流直下三千尺", "疑是银河落九天",
    "两个黄鹂鸣翠柳", "一行白鹭上青天", "窗含西岭千秋雪", "门泊东吴万里船",
    "千里黄云白日曛", "北风吹雁雪纷纷", "莫愁前路无知己", "天下谁人不识君",
    "月落乌啼霜满天", "江枫渔火对愁眠", "姑苏城外寒山寺", "夜半钟声到客船",
    "折戟沉沙铁未销", "自将磨洗认前朝", "东风不与周郎便", "铜雀春深锁二乔",
    "远上寒山石径斜", "白云生处有人家", "停车坐爱枫林晚", "霜叶红于二月花",
]

# ============================================================
# Judge LoRA 数据生成函数
# ============================================================

LIKES = ["喜欢", "不喜欢", "爱吃", "不爱吃", "擅长", "不擅长", "热爱", "讨厌", "偏好", "厌恶"]
LIKES_FLIP = {
    "喜欢": "不喜欢", "不喜欢": "喜欢",
    "爱吃": "不爱吃", "不爱吃": "爱吃",
    "擅长": "不擅长", "不擅长": "擅长",
    "热爱": "讨厌", "讨厌": "热爱",
    "偏好": "厌恶", "厌恶": "偏好",
}

ATTRIBUTES_REVERSAL = [
    ("擅长", ["数学", "编程", "绘画", "音乐", "写作", " cooking", "游泳", "英语", "演讲", "舞蹈"]),
    ("不擅长", ["数学", "编程", "绘画", "音乐", "写作", "游泳", "英语", "演讲", "舞蹈"]),
]

NUMERIC_VALUES = [
    ("人口", CITIES, ["800万", "1000万", "1200万", "1500万", "2000万", "2200万", "2500万", "3000万"]),
    ("面积", CITIES, ["500平方公里", "800平方公里", "1000平方公里", "1200平方公里", "1500平方公里"]),
    ("成立年份", COMPANIES, ["1998年", "2000年", "2003年", "2008年", "2010年", "2012年", "2015年"]),
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


def generate_update_conflict_samples(n):
    """生成同维度值冲突(喜好反转)和同维度数值更新的UPDATE样本"""
    samples = []

    # 喜好反转: ~90% of n
    n_like = int(n * 0.9)
    for _ in range(n_like):
        person = random.choice(PERSONS)
        like = random.choice(list(LIKES_FLIP.keys()))
        obj = random.choice(OBJECTS[:10])  # food/fruit objects for likes
        old_memory = f"{person}{like}{obj}"
        flipped = LIKES_FLIP[like]
        new_fact = f"{person}{flipped}{obj}"

        input_text = f"旧记忆：{old_memory}\n新事实：{new_fact}\n请判断新事实与旧记忆的关系，并决定处理策略。"

        if like in ["喜欢", "爱吃", "热爱", "偏好"]:
            reason_text = f"新事实与旧记忆在同维度上直接冲突：{person}对{obj}的态度从'{like}'变为'{flipped}'，属于喜好反转，应更新记忆。"
        else:
            reason_text = f"新事实与旧记忆在同维度上直接冲突：{person}对{obj}的态度从'{like}'变为'{flipped}'，属于喜好反转，应更新记忆。"

        output_dict = {
            "decision": "UPDATE",
            "reason": reason_text,
            "updated_memory": new_fact,
        }
        output_text = json.dumps(output_dict, ensure_ascii=False)

        samples.append({
            "conversations": [
                {"from": "human", "value": input_text},
                {"from": "gpt", "value": output_text},
            ]
        })

    # 数值更新: ~10% of n
    n_numeric = n - n_like
    for _ in range(n_numeric):
        attr_name, entities, values = random.choice(NUMERIC_VALUES)
        entity = random.choice(entities)
        old_val = random.choice(values)
        # Pick a different value for the new fact (larger/updated)
        remaining = [v for v in values if v != old_val]
        if remaining:
            new_val = random.choice(remaining)
        else:
            new_val = old_val  # fallback, unlikely

        old_memory = f"{entity}{attr_name}{old_val}"
        new_fact = f"{entity}{attr_name}{new_val}"

        input_text = f"旧记忆：{old_memory}\n新事实：{new_fact}\n请判断新事实与旧记忆的关系，并决定处理策略。"

        reason_text = f"新事实对旧记忆中的数值进行了更新：{entity}的{attr_name}从'{old_val}'更新为'{new_val}'，属于数值更新，应更新记忆。"

        output_dict = {
            "decision": "UPDATE",
            "reason": reason_text,
            "updated_memory": new_fact,
        }
        output_text = json.dumps(output_dict, ensure_ascii=False)

        samples.append({
            "conversations": [
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

        # Randomly pick whether old is positive or negative
        if random.random() < 0.5:
            old_attr, new_attr = pos_attr, neg_attr
        else:
            old_attr, new_attr = neg_attr, pos_attr

        old_memory = f"{person}{old_attr}{skill}"
        new_fact = f"{person}{new_attr}{skill}"

        input_text = f"旧记忆：{old_memory}\n新事实：{new_fact}\n请判断新事实与旧记忆的关系，并决定处理策略。"

        reason_text = f"新事实与旧记忆在同维度上直接冲突：{person}对{skill}的能力描述从'{old_attr}'变为'{new_attr}'，属于属性反转，应更新记忆。"

        output_dict = {
            "decision": "UPDATE",
            "reason": reason_text,
            "updated_memory": new_fact,
        }
        output_text = json.dumps(output_dict, ensure_ascii=False)

        samples.append({
            "conversations": [
                {"from": "human", "value": input_text},
                {"from": "gpt", "value": output_text},
            ]
        })

    return samples


def generate_keep_different_dimension_samples(n):
    """生成不同维度共存(同属性不同对象)的KEEP样本"""
    samples = []

    for _ in range(n):
        attr, objects = random.choice(SAME_ATTR_DIFF_OBJ)
        person = random.choice(PERSONS)

        obj1 = random.choice(objects)
        remaining = [o for o in objects if o != obj1]
        if remaining:
            obj2 = random.choice(remaining)
        else:
            obj2 = obj1  # fallback

        old_memory = f"{person}{attr}{obj1}"
        new_fact = f"{person}{attr}{obj2}"

        input_text = f"旧记忆：{old_memory}\n新事实：{new_fact}\n请判断新事实与旧记忆的关系，并决定处理策略。"

        reason_text = f"新事实与旧记忆属于同属性的不同维度：{person}可以同时'{attr}{obj1}'和'{attr}{obj2}'，两者并不冲突，应保持共存。"

        output_dict = {
            "decision": "KEEP",
            "reason": reason_text,
            "updated_memory": f"{old_memory}；{new_fact}",
        }
        output_text = json.dumps(output_dict, ensure_ascii=False)

        samples.append({
            "conversations": [
                {"from": "human", "value": input_text},
                {"from": "gpt", "value": output_text},
            ]
        })

    return samples


def generate_keep_different_domain_samples(n):
    """生成不同领域共存(完全不同领域)的KEEP样本"""
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

        output_dict = {
            "decision": "KEEP",
            "reason": reason_text,
            "updated_memory": f"{old_memory}；{new_fact}",
        }
        output_text = json.dumps(output_dict, ensure_ascii=False)

        samples.append({
            "conversations": [
                {"from": "human", "value": input_text},
                {"from": "gpt", "value": output_text},
            ]
        })

    return samples


# ============================================================
# Poet LoRA 数据生成函数
# ============================================================

def _generate_placeholder_poem(topic, lines, char_per_line):
    """从经典诗句池中选取指定数量的诗句，组成格式正确的诗歌内容"""
    if char_per_line == 5:
        pool = CLASSIC_LINES_5
    else:
        pool = CLASSIC_LINES_7

    # Try to fill with lines from pool; if not enough, reuse
    selected = []
    available = list(pool)
    for i in range(lines):
        if available:
            line = random.choice(available)
            selected.append(line)
            available.remove(line)
        else:
            # Reuse from pool if we run out
            selected.append(random.choice(pool))

    poem_text = "\n".join(selected)
    return poem_text


def generate_poet_template_samples(n):
    """生成古诗写作的训练样本：经典种子 + 主题×风格×诗体模板变体"""
    samples = []

    # First, add the 7 classic seed poems
    for poem in POETRY_DATA:
        samples.append({
            "conversations": [
                {"from": "human", "value": poem["prompt"]},
                {"from": "gpt", "value": poem["response"]},
            ]
        })

    # Then generate template-based variants for remaining samples
    remaining = n - len(POETRY_DATA)
    if remaining <= 0:
        return samples

    for _ in range(remaining):
        topic = random.choice(POETRY_TOPICS)
        style = random.choice(POETRY_STYLES)
        form_name, char_per_line, num_lines = random.choice(POETRY_FORMS)

        input_text = f"请写一首关于{topic}的{form_name}，风格要求：{style}。"

        # Generate poem title from topic + random modifier
        title_modifiers = ["咏", "题", "怀", "望", "忆", "叹", "赋", "秋", "春", "夜"]
        title = f"{random.choice(title_modifiers)}{topic}"

        poem_content = _generate_placeholder_poem(topic, num_lines, char_per_line)

        # Generate appreciation text
        appreciation_templates = [
            f"此诗以{topic}为题，意境{style}，语言凝练而意蕴深远，将{topic}之情写得含蓄生动。",
            f"此诗咏{topic}，风格{style}，由景入情，将{topic}之意层层递进，韵味悠长。",
            f"此诗描写{topic}，笔法{style}，意象鲜明而情感内敛，令人回味无穷。",
            f"此诗以{topic}起兴，{style}之中见深情，将{topic}之美与人之感交织，意境浑然。",
            f"此诗题咏{topic}，格调{style}，以简练笔墨写出{topic}的神韵与气象。",
        ]
        appreciation = random.choice(appreciation_templates)

        output_text = f"诗题：{title}\n\n{poem_content}\n\n赏析：{appreciation}"

        samples.append({
            "conversations": [
                {"from": "human", "value": input_text},
                {"from": "gpt", "value": output_text},
            ]
        })

    return samples


# ============================================================
# 主函数 - 生成全部4个数据文件
# ============================================================

def main():
    random.seed(42)

    # Ensure data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)

    # ----- Judge LoRA data -----
    # Train: ~800 samples, with ratio: conflict 40%, attribute ~10%, diff_dim 30%, diff_domain 20%
    judge_train_total = 800
    n_conflict = int(judge_train_total * 0.40)   # 320
    n_attribute = int(judge_train_total * 0.10)   # 80
    n_diff_dim = int(judge_train_total * 0.30)    # 240
    n_diff_domain = judge_train_total - n_conflict - n_attribute - n_diff_dim  # 160

    judge_train_samples = []
    judge_train_samples.extend(generate_update_conflict_samples(n_conflict))
    judge_train_samples.extend(generate_update_attribute_samples(n_attribute))
    judge_train_samples.extend(generate_keep_different_dimension_samples(n_diff_dim))
    judge_train_samples.extend(generate_keep_different_domain_samples(n_diff_domain))

    # Shuffle training data
    random.shuffle(judge_train_samples)

    # Test: ~200 samples, same ratio
    judge_test_total = 200
    n_conflict_test = int(judge_test_total * 0.40)   # 80
    n_attribute_test = int(judge_test_total * 0.10)   # 20
    n_diff_dim_test = int(judge_test_total * 0.30)    # 60
    n_diff_domain_test = judge_test_total - n_conflict_test - n_attribute_test - n_diff_dim_test  # 40

    # Re-seed for test data to ensure different samples
    random.seed(123)

    judge_test_samples = []
    judge_test_samples.extend(generate_update_conflict_samples(n_conflict_test))
    judge_test_samples.extend(generate_update_attribute_samples(n_attribute_test))
    judge_test_samples.extend(generate_keep_different_dimension_samples(n_diff_dim_test))
    judge_test_samples.extend(generate_keep_different_domain_samples(n_diff_domain_test))

    random.shuffle(judge_test_samples)

    # Write judge data files
    judge_train_path = os.path.join(data_dir, "judge_train.json")
    judge_test_path = os.path.join(data_dir, "judge_test.json")

    with open(judge_train_path, "w", encoding="utf-8") as f:
        json.dump(judge_train_samples, f, ensure_ascii=False, indent=2)

    with open(judge_test_path, "w", encoding="utf-8") as f:
        json.dump(judge_test_samples, f, ensure_ascii=False, indent=2)

    print(f"Judge train: {len(judge_train_samples)} samples -> {judge_train_path}")
    print(f"Judge test:  {len(judge_test_samples)} samples -> {judge_test_path}")

    # ----- Poet LoRA data -----
    # Train: ~500 samples (7 classic seeds + template variants)
    random.seed(42)  # Reset seed for poet data
    poet_train_samples = generate_poet_template_samples(500)
    random.shuffle(poet_train_samples)

    # Test: ~100 samples
    random.seed(456)
    poet_test_samples = generate_poet_template_samples(100)
    random.shuffle(poet_test_samples)

    # Write poet data files
    poet_train_path = os.path.join(data_dir, "poet_train.json")
    poet_test_path = os.path.join(data_dir, "poet_test.json")

    with open(poet_train_path, "w", encoding="utf-8") as f:
        json.dump(poet_train_samples, f, ensure_ascii=False, indent=2)

    with open(poet_test_path, "w", encoding="utf-8") as f:
        json.dump(poet_test_samples, f, ensure_ascii=False, indent=2)

    print(f"Poet train: {len(poet_train_samples)} samples -> {poet_train_path}")
    print(f"Poet test:  {len(poet_test_samples)} samples -> {poet_test_path}")

    # ----- Summary -----
    print("\n--- Data Generation Summary ---")
    print(f"Judge LoRA: {len(judge_train_samples)} train + {len(judge_test_samples)} test = {len(judge_train_samples) + len(judge_test_samples)} total")
    print(f"Poet LoRA:  {len(poet_train_samples)} train + {len(poet_test_samples)} test = {len(poet_train_samples) + len(poet_test_samples)} total")

    # Count decision types in judge train
    update_count = sum(1 for s in judge_train_samples if '"decision": "UPDATE"' in s["conversations"][1]["value"])
    keep_count = sum(1 for s in judge_train_samples if '"decision": "KEEP"' in s["conversations"][1]["value"])
    print(f"\nJudge train decisions: UPDATE={update_count}, KEEP={keep_count}")


if __name__ == "__main__":
    main()
