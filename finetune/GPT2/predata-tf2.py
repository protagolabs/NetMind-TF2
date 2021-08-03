import re
from sklearn.model_selection import train_test_split
import opencc

converter = opencc.OpenCC('s2t.json')

f = open("train.txt", "r")
data = []
for line in f:
  stripped_line = line.strip()
  data.append(stripped_line)
f.close()

def build_text_files(src_path, dest_path):
    fs = open(src_path, "r")
    with open(dest_path, 'w', encoding='utf8') as fd:

      for line in fs:
        stripped_line = str(line).strip()
        # out_line = stripped_line.replace('\n', '[SEP]')
        out_line = stripped_line.replace(' ', '[SEP]')
        out_line = out_line + '[SEP]'
        out_line = '[CLS]' + out_line
        fd.write(out_line+"\n")

    fs.close()


build_text_files("train.txt",'train_dataset.txt')
build_text_files("valid.txt",'test_dataset.txt')