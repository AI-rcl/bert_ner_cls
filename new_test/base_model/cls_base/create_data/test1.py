import sys
from xlrd import open_workbook
import xlwt
from tqdm import tqdm
import re
import os
import random



workbook=open_workbook('汉语分类.xls')
label=workbook.sheet_names()
#['陈述句', '疑问句', '祈使句', '感叹句', '陈述句分类', '疑问句分类', '复句常用关联词']

def create_first_cls():
    original_data=open('ori_data.tsv','w')

    for i in tqdm(range(4)):
        sheet=workbook.sheet_by_index(i)
        label=[word.strip() for word in sheet.col_values(1)[1:]]
        content=[re.sub('\s',',',sentence) for sentence in sheet.col_values(0)[1:]]
        records=list(map(lambda x,y:x+'\t'+y,label,content))
        for once in records:
            record=once+'\n'
            original_data.write(record)

def create_state_cls():
    state_data=open('state_data.tsv','w')
    sheet = workbook.sheet_by_index(4)
    label=[word.strip() for word in sheet.col_values(1)[1:]]
    content=[re.sub('\s',',',sentence) for sentence in sheet.col_values(0)[1:]]
    records = list(map(lambda x, y: x + '\t' + y, label, content))
    for once in records:
        record = once + '\n'
        state_data.write(record)

def create_query_cls():
    query_data=open('query_data.tsv','w')
    sheet = workbook.sheet_by_index(5)
    label=[word.strip() for word in sheet.col_values(1)[1:]]
    content = [re.sub('\s', ',', sentence) for sentence in sheet.col_values(0)[1:]]
    records = list(map(lambda x, y: x + '\t' + y, label,content))
    for once in records:
        record = once + '\n'
        query_data.write(record)


def create_data(file_name):
    base_path='../data_file/'
    name=file_name.split('.')[0]
    dir_path = base_path + name
    if os.path.exists(dir_path):
        pass
    else:
        os.mkdir(base_path+name)
    train_file=open(dir_path+'/train.tsv','w')
    test_file=open(dir_path+'/test.tsv','w')
    file=open(file_name,'r')
    data=file.readlines()
    data=data*3
    random.shuffle(data)
    lenth=len(data)

    for times,val in enumerate(data):

        train_file.write(val)
        if times>0.3*lenth and times<0.6*lenth:
            test_file.write(val)


    train_file.close()
    test_file.close()




# create_first_cls()
# create_state_cls()
# create_query_cls()
create_data('ori_data.tsv')

