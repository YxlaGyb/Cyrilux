"""
生成两个极端异构数据集用于灾难性遗忘验证：
  task_zh.jsonl — 纯中文文本（中文新闻/谚语/文章）
  task_en.jsonl — 纯英文随机文本（无结构噪声字节）

输出格式: 每行 {"text": "..."}，可直接被 _LocalDataset 读取。
"""
import os, json, random, struct

DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
os.makedirs(DST, exist_ok=True)

# ── 中文语料 ──
ZH_SENTENCES = [
    "近日，国家统计局发布了最新经济数据，显示国内生产总值同比增长百分之五点二。",
    "科技创新是引领发展的第一动力，要加快实现高水平科技自立自强。",
    "生态文明建设关系人民福祉和民族未来，必须坚持绿色发展理念。",
    "教育公平是社会公平的重要基础，要推动优质教育资源均衡配置。",
    "乡村振兴战略是新时代做好三农工作的总抓手，要全面推进农业农村现代化。",
    "人工智能技术正在深刻改变各行各业的生产方式和商业模式。",
    "传统文化是一个民族的根和魂，要加强对非物质文化遗产的保护和传承。",
    "数字经济已成为经济增长的新引擎，数字化转型是企业的必答题。",
    "健康是人民幸福生活的基础，要完善公共卫生体系，提高医疗服务水平。",
    "新能源汽车产业快速发展，中国品牌在全球市场上的竞争力不断提升。",
    "深化改革开放是推动高质量发展的根本动力，要持续优化营商环境。",
    "网络安全是国家安全的重要组成部分，要加强数据安全和个人信息保护。",
    "碳达峰碳中和目标引领着中国绿色低碳转型的方向和步伐。",
    "中医药是中华民族的瑰宝，在疫情防控中发挥了独特的重要作用。",
    "人才是第一资源，要加快建设世界重要人才中心和创新高地。",
    "全过程人民民主是社会主义民主政治的本质属性，是最广泛最真实最管用的民主。",
    "共同富裕是社会主义的本质要求，要在高质量发展中促进共同富裕。",
    "粮食安全是国之大者，要牢牢把住粮食安全主动权。",
    "产业链供应链安全稳定是构建新发展格局的基础。",
    "区域协调发展战略是解决发展不平衡不充分问题的重要途径。",
]
# 扩展：组合句子生成更长文本
ZH_LONG = []
for _ in range(5000):
    n = random.randint(3, 8)
    text = '。'.join(random.choices(ZH_SENTENCES, k=n)) + '。'
    ZH_LONG.append({'text': text})

# ── 英文噪声字节 ──
# 生成完全随机的字节序列，与中文 UTF-8 字节分布截然不同
EN_NOISE = []
for _ in range(5000):
    length = random.randint(60, 300)
    # 随机 ASCII 可打印字符 + 空格 + 换行
    chars = []
    for _ in range(length):
        # 70% 小写字母, 15% 空格/标点, 10% 大写字母, 5% 数字
        r = random.random()
        if r < 0.70:
            chars.append(chr(random.randint(97, 122)))  # a-z
        elif r < 0.85:
            chars.append(random.choice(' .,!?;\n'))
        elif r < 0.95:
            chars.append(chr(random.randint(65, 90)))   # A-Z
        else:
            chars.append(chr(random.randint(48, 57)))   # 0-9
    EN_NOISE.append({'text': ''.join(chars)})


# 写入
def write_jsonl(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'{path}: {len(data)} samples written')


write_jsonl(os.path.join(DST, 'task_zh.jsonl'), ZH_LONG)
write_jsonl(os.path.join(DST, 'task_en.jsonl'), EN_NOISE)
print('Done — extreme test datasets ready.')
