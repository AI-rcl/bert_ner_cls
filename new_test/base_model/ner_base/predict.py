import os
import pickle
import tensorflow as tf
from utils import create_model, get_logger,load_config
from model import Model
from loader import input_from_line



class Ner():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    def __init__(self,sess,model_dir):
        self.config = load_config(os.path.join(model_dir,'config_file'))
        self.map_file=os.path.join(model_dir,'maps.pkl')
        self.ckpt_path=os.path.join(model_dir,'ckpt')
        self.logger = get_logger()
        # limit GPU memory
        self.tf_config = tf.ConfigProto()
        self.max_seq_len=56
        self.tf_config.gpu_options.allow_growth = True
        with open(self.map_file, "rb") as f:
            self.tag_to_id, self.id_to_tag = pickle.load(f)
        self.sess=sess
        self.model=create_model(self.sess, Model, self.ckpt_path, self.config, self.logger)
        # self.del_all_flags(FLAGS)
    def predict(self,sentece):
        result=self.model.evaluate_line(self.sess,input_from_line(sentece, self.max_seq_len, self.tag_to_id),self.id_to_tag)
        return result['entities']
    def del_all_flags(self,FLAGS):
        flags_dict = FLAGS._flags()
        keys_list = [keys for keys in flags_dict]
        for keys in keys_list:
            FLAGS.__delattr__(keys)


# g1=tf.Graph()
# sess=tf.Session(graph=g1)
# with sess.as_default():
#     with g1.as_default():
#         tf.global_variables_initializer().run()
#         ner=Ner(sess)
#         while True:
#             sentence=input('>>>:')
#             res=ner.predict(sentence)
#             print(res)
# if __name__ == '__main__':
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     tf.app.run(main)
