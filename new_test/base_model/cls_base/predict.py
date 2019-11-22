#! usr/bin/env python
import tensorflow as tf
import os,pickle



class Cls(object):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    def __init__(self,model_dir):
        self.dir_path = os.path.dirname(__file__)
        self.closed = False
        self.first_run = True
        self.bert_model_dir=self.dir_path+'/chinese_L-12_H-768_A-12'
        self.seq_length = 56
        self.text = None
        self.num_examples = None
        self.predictions = None
        self.estimator = self.get_estimator()
        self.model_dir=model_dir
        self.graph_path=self.model_dir+'/classification_model.pb'
        _,self.label=self.init_predict_var()

    def del_all_flags(self,FLAGS):
        flags_dict = FLAGS._flags()
        keys_list = [keys for keys in flags_dict]
        for keys in keys_list:
            FLAGS.__delattr__(keys)

    def init_predict_var(self):
        """
        初始化NER所需要的一些辅助数据
        :param path:
        :return:
        """
        label_list_file = os.path.join(self.model_dir, 'label_list.pkl')
        label_list = []
        if os.path.exists(label_list_file):
            with open(label_list_file, 'rb') as fd:
                label_list = pickle.load(fd)
        num_labels = len(label_list)

        with open(os.path.join(self.model_dir, 'label2id.pkl'), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
        return label2id, id2label


    def get_estimator(self):
        from tensorflow.python.estimator.estimator import Estimator
        from tensorflow.python.estimator.run_config import RunConfig
        from tensorflow.python.estimator.model_fn import EstimatorSpec


        def classification_model_fn(features):
            """
            文本分类模型的model_fn
            :param features:
            :param labels:
            :param mode:
            :param params:
            :return:
            """
            with tf.gfile.GFile(self.graph_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            input_map = {"input_ids": input_ids, "input_mask": input_mask}
            pred_probs = tf.import_graph_def(graph_def, name='', input_map=input_map, return_elements=['pred_prob:0'])

            return EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions={
                'encodes': tf.argmax(pred_probs[0], axis=-1),
                'score': tf.reduce_max(pred_probs[0], axis=-1)
            })

        # 0 表示只使用CPU 1 表示使用GPU
        config = tf.ConfigProto(device_count={'GPU':1})
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
        config.log_device_placement = False
        # session-wise XLA doesn't seem to work on tf 1.10
        # if args.xla:
        #     config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        return Estimator(model_fn=classification_model_fn, config=RunConfig(session_config=config))

    def gen(self):
        from extract_features import convert_lst_to_features
        from tokenization import FullTokenizer
        tokenizer = FullTokenizer(vocab_file=os.path.join(self.bert_model_dir, 'vocab.txt'))
        # Windows does not support logger in MP environment, thus get a new logger
        #这个while 循环保证生成器hold住，estimator.predict不用重新加载
        while not self.closed:
            is_tokenized = all(isinstance(el, list) for el in self.text)
            tmp_f = list(convert_lst_to_features(self.text, self.seq_length, tokenizer,
                                                 is_tokenized, mask_cls_sep=True))
            # print([f.input_ids for f in tmp_f])
            yield {
                'input_ids': [f.input_ids for f in tmp_f],
                'input_mask': [f.input_mask for f in tmp_f],
                'input_type_ids': [f.input_type_ids for f in tmp_f]
            }

    def input_fn_builder(self):
        dataset= (tf.data.Dataset.from_generator(
                self.gen,
                output_types={'input_ids': tf.int32,
                              'input_mask': tf.int32,
                              'input_type_ids': tf.int32,
                              },
                output_shapes={
                    'input_ids': (None, self.seq_length),
                    'input_mask': (None, self.seq_length), #.shard(num_shards=4, index=4)
                    'input_type_ids': (None, self.seq_length)}).prefetch(0))

        return dataset


    def predict(self, text):
        self.text = [text]
        if self.first_run:
            self.predictions = self.estimator.predict(
                input_fn=self.input_fn_builder, yield_single_examples=False)
            self.first_run = False
        probabilities = next(self.predictions)
        pred_label_result = [self.label.get(x, -1) for x in probabilities['encodes']]
        pred_score_result = probabilities['score'].tolist()
        return pred_label_result
    def close(self):
        self.closed = True





# for r in estimator.predict(input_fn=input_fn_builder(text), yield_single_examples=False):
#     pred_label_result = [label.get(x, -1) for x in r['encodes']]
#     pred_score_result = r['score'].tolist()
#     print(pred_label_result)

# res=estimator.predict(input_fn_builder(text),yield_single_examples=False)
# for r in res:
#     print(r)