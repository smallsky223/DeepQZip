# -*- coding: utf-8 -*-
import sys
import numpy as np

if len(sys.argv[1])<4:
        print("Usage: python extract_ba.py input.fastq output_dir param_file")
        exit(-1)

# 碱基合并成一行，质量值按行排列
with open(sys.argv[1]) as fp1, open(sys.argv[2]+'.base.txt','w') as fp2, open(sys.argv[2]+'.qv.txt','w') as fp3:
        for i,x in enumerate(fp1):
                if i%4==1:
                        x = x.replace('\n', '')
                        fp2.write(x)
                elif i%4==3:
                        x = x.replace('\n', '')
                        fp3.write(x)

# 生成碱基字典，将碱基字符文件转换为数值
with open(sys.argv[2]+'.base.txt', 'r') as fp:
        text_base = fp.read()
# text_base = text_base.replace('\n', '')
print(len(text_base))
vals = list(set(text_base))
char2id_dict = {c: i for (i,c) in enumerate(vals)}
id2char_dict = {i: c for (i,c) in enumerate(vals)}

with open(sys.argv[3],'w') as f:
    f.write(str(len(id2char_dict)) + '\n')
    f.write('\t'.join([str(v) for k,v in id2char_dict.items()]) + '\n')
    f.write(str(len(text_base)) + '\n')

out = [char2id_dict[c] for c in text_base]
integer_encoded = np.array(out)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
print(integer_encoded[:10])

# .npy存储N*1的二维数组，即integer_encoded[[0],[1],[2],...]
np.savez_compressed(sys.argv[2]+'.base.npz', data=integer_encoded)

# 将质量值字符文件转换为对应碱基概率数值
with open(sys.argv[2]+'.qv.txt', 'r') as fp:
        text_qv = fp.read()
# text_qv = text_qv.replace('\n', '')
out = [ ord(char)-32 for char in text_qv]
prob_encoded = np.array(out)
prob_encoded = prob_encoded.reshape(len(prob_encoded), 1)
print(prob_encoded[:10])

np.savez_compressed(sys.argv[2]+'.qv.npz', data=prob_encoded)
