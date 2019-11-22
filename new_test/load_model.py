import os,sys
import tensorflow as tf

BASE_DIR=os.path.dirname(__file__)
NER_BASE_PATH=os.path.join(BASE_DIR,'base_model/ner_base')
CLS_BASE_PATH=os.path.join(BASE_DIR,'base_model/cls_base')
sys.path.append(NER_BASE_PATH)
sys.path.append(CLS_BASE_PATH)
from base_model.cls_base.predict import Cls
from base_model.ner_base.predict import Ner

class Ner_loader():
    def __init__(self):
        self.bots_path='bots/ner_bots'
        self._get_model_path()
        self.model_dict={}
        self._create_model()
    def _get_model_path(self):
        self.bots_list=os.listdir(self.bots_path)
        self.model_path_list=[os.path.join(self.bots_path,bot,'model') for bot in self.bots_list]
    def _create_model(self):
        for bot,model_path in zip(self.bots_list,self.model_path_list):
            graph=tf.Graph()
            sess=tf.Session(graph=graph)
            with sess.as_default():
                with graph.as_default():
                    tf.global_variables_initializer().run()
                    ner_model=Ner(sess,model_path)
                    self.model_dict.setdefault(bot,ner_model)

class Cls_loader():
    def __init__(self):
        self.cls_path='bots/cls'
        self.model_dict={}
        self._get_model_path()
        self._create_model()
    def _get_model_path(self):
        self.cls_list=os.listdir(self.cls_path)
        self.model_path_list=[os.path.join(self.cls_path,cls,'model') for cls in self.cls_list]
    def _create_model(self):
        for cls,model_path in zip(self.cls_list,self.model_path_list):
            cls_model=Cls(model_path)
            self.model_dict.setdefault(cls,cls_model)

# path='bots/ner_bots'
# print(os.listdir(path))
# text='华为在哪里'
# ner_loader=Ner_Loader()
# print(ner_loader.model_dict)
# print(ner_loader.model_dict['global_ner'].predict(text))
# cls_loader=Cls_loader()
# print(cls_loader.model_dict['base_sentence'].predict(text))