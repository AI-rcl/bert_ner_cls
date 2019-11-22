import os
import shutil

# os.system('python base_model/ner_base/train.py %s'%('ner002'))
# os.system('''echo hello;
#           echo world''')

# label_list='a,b,c,d'
# os.system('python system_test.py %s'%label_list)
def mk_dir(mode,bot_id):
    if mode=='ner':
        bot_dir='bots/ner_bots/'+bot_id
    elif mode=='cls':
        bot_dir='bots/cls/'+bot_id
    else:
        raise ValueError

    data_path = bot_dir + '/data'
    model_path = bot_dir + '/model'
    if os.path.exists(bot_dir):
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
    else:
        os.mkdir(bot_dir)
        os.mkdir(data_path)
        os.mkdir(model_path)
    return 0

def remove_dir(mode,bot_id):
    if mode == 'ner':
        bot_dir = 'bots/ner_bots/' + bot_id
    elif mode == 'cls':
        bot_dir = 'bots/cls/' + bot_id
    else:
        raise ValueError
    if os.path.exists(bot_dir):
        shutil.rmtree(bot_dir)
    return 0

def train_ner(args):
    train_file='base_model/ner_base/train.py'
    bot_id=args['bot_id']
    if args['train_epoch']:
        train_epoch=args['train_epoch']
    else:
        train_epoch=5
    mk_dir('ner',bot_id)
    os.system('python %s %d %s'%(train_file,train_epoch,bot_id))

# ner_args={'bot_id':'ner001',
#       'train_epoch':5}

def train_cls(args):
    train_file='base_model/cls_base/run_classifier.py'
    ckpt2pd_file='base_model/cls_base/ckpt2pd.py'
    cls_name=args['cls_name']
    if args['train_epoch']:
        train_epoch=args['train_epoch']
    else:
        train_epoch=5
    if args['label_list']:
        if type(args['label_list']):
            label_list=','.join(args['label_list'])
        else:
            raise TypeError('label_list must be a list')
    else:
        raise ValueError("label_list cant't be empty")
    mk_dir('cls',cls_name)
    os.system('''python %s %s %d %s;
                python %s %s %s'''%(train_file,label_list,train_epoch,cls_name,ckpt2pd_file,label_list,cls_name))

cls_args={'cls_name':'test_sentence',
      'train_epoch':10,
      'label_list':['陈述句','疑问句','感叹句','祈使句']}
train_cls(cls_args)

