import numpy as np
import settings

# 随机生成诗的方法
def generate_random_poetry(tokenizer, model, s=''):
    token_ids = tokenizer.encode(s)
    token_ids = token_ids[:-1]
    while len(token_ids) < settings.MAX_LEN:
        output = model(np.array([token_ids, ], dtype=np.int32))
        _probas = output.numpy()[0, -1, 3:]
        del output
        #按顺序排列
        p_args = _probas.argsort()[::-1][:100]
        #排列后的概率顺序    
        p = _probas[p_args]
        #概率归一
        p = p / sum(p)
        target_index = np.random.choice(len(p), p=p)
        target = p_args[target_index] + 3
        token_ids.append(target)
        if target == 3:
            break
    return tokenizer.decode(token_ids)

#藏头诗的方法
def generate_acrostic(tokenizer, model, head):
    token_ids = tokenizer.encode('')
    token_ids = token_ids[:-1]
    punctuations = ['，', '。']
    punctuations_ids = {tokenizer.token_to_id(token) for token in punctuations}
    poetry = []
    for ch in head:
        #记录这个字
        poetry.append(ch)
        #藏头诗的字符转成token_id
        token_id = tokenizer.token_to_id(ch)
        #开始生成一个短句
        token_ids.append(token_id)
        while True:
            #进行预测 只保留第一个样例
            output = model(np.array([token_ids, ], dtype=np.int32))
            _probas = output.numpy()[0, -1, 3:]
            del output
            #按出现概率 对所有token倒序排列
            p_args = _probas.argsort()[::-1][:100]
            # 排列后的概率顺序
            p = _probas[p_args]
            #概率归一
            p = p / sum(p)
            target_index = np.random.choice(len(p), p=p)
            target = p_args[target_index] + 3
            token_ids.append(target)
            if target > 3:
                poetry.append(tokenizer.id_to_token(target))
            if target in punctuations_ids:
                break
    return ''.join(poetry)










