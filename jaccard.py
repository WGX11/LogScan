import re

#日志预处理
def log_preprocess(log):
    #去除原始日志中的符号、十进制和十六进制数字并小写
    log_onlyletter = re.sub(r'0[xX][0-9a-fA-F]+', ' ', log) #替换十六进制数字，'0x'或'0X'开头，后跟至少一个十六进制数字
    log_onlyletter = re.sub(r'[^a-zA-Z]+', ' ', log_onlyletter) #替换非字母字符
    log_onlyletter = log_onlyletter.lower()
    return log_onlyletter
# 计算日志的杰卡德距离
def jaccard_distance(log_1, log_2):
    #构建日志单词集合
    log_letterset_1 = set(log_1.split())
    log_letterset_2 = set(log_2.split())
    #计算两个单词集合之间的杰卡德相似度
    intersection = len(log_letterset_1 & log_letterset_2)
    union = len(log_letterset_1 | log_letterset_2)
    similarity = intersection / union
    return 1 - similarity

if __name__ == '__main__':
    log_1 = "HEllo! NICE to meet you 95200::"
    log_2 = "Hello! Nice to not mee+et you 95200"
    print(jaccard_distance(log_preprocess(log_1), log_preprocess(log_2)))
