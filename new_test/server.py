from flask import Flask, jsonify, abort, request
import tensorflow as tf
from load_model import Cls_loader
from load_model import Ner_loader

cls_loader=Cls_loader()
ner_loader=Ner_loader()
INIT_TEXT='欢迎来到泰康'

cls_dict=cls_loader.model_dict
ner_dict=ner_loader.model_dict
for cls in cls_dict.values():
    cls.predict(INIT_TEXT)

app = Flask(__name__)

@app.route('/AI_model/api/bot/', methods=['GET'])
def predict():
    bot_id=request.args.get('bot_id')
    sentence=request.args.get('sentence')
    res={}
    for bot_id,model in ner_dict.items():
        res.setdefault(bot_id,model.predict(sentence))
    for name,model in cls_dict.items():
        res.setdefault(name,model.predict(sentence))

    return jsonify(res)
    # abort(404)



@app.route('/model/api/bot/', methods=['PUT'])
def update_book():
    pass
    # abort(400)


@app.route('/bookstore/api/v1/books/<int:id>', methods=['DELETE'])
def delete_task(id):
    for book in books:
        if book['id']==id:
            books.remove(book)
            return jsonify({'model_data': True})
    abort(404)

    return jsonify({'model_data': True})


@app.route('/bookstore/api/v1/books/', methods=['POST'])
def create_task():
    if not request.form or not 'title' in request.form:
        abort(400)
    book = {
        'id': books[-1]['id'] + 1,
        'title': request.form['title'],
        'auther': request.form['auther'],
        'price': request.form['price'],
    }
    books.append(book)
    return jsonify({'book': book}), 201


if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True)