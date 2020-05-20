from flask import Flask, jsonify
import weka.core.jvm as jvm
from weka.core.converters import Loader 
from weka.associations import Associator
from weka.classifiers import Classifier
from pymongo import MongoClient
import pandas as pd
from dict2obj import Dict2Obj
from weka.core.dataset import create_instances_from_lists

app = Flask(__name__)

@app.route('/')
def hello():
    client = MongoClient('localhost', 27017)
    db = client["aia-social"]
    familias = db["36e7cbfe-08b7-4fae-8b05-7ddfc8133289"]
    # pessoas = db["f8493a4e-f61e-441e-b802-d9b582a90f55"]

    cursor = familias.find({}, {'_id': False}).limit(10)

    dici = list()

    for element in cursor:
        dici.append(Dict2Obj(element))

    if(jvm.started is None):
        jvm.start(max_heap_size="4092m", packages=True)

    
    data = create_instances_from_lists(list(cursor))

    apriori = Associator(classname="weka.associations.Apriori", options=["-N", "9", "-I"])
    result = apriori.build_associations(data)

    print(result)

    return jsonify({'result': 'OK'})
    # result = list(familias.find().limit(100))
    

    # if(jvm.started is None):
    #     jvm.start(max_heap_size="4092m", packages=True)

    # # loader = Loader(classname="weka.core.converters.CSVLoader")
    # # data = loader.load_file('./bases_dados/governamental/convertida.csv')
    # # print(data)

    # # apriori = Associator(classname="weka.associations.Apriori")
    # # apriori.build_associations(data)

    # cba = Classifier(classname="weka.classifiers.cba")
    # cba.build_classifier(result)

    # pipeline = [{'$lookup': 
    #             {'from' : 'f8493a4e-f61e-441e-b802-d9b582a90f55',
    #              'localField' : 'id_familia',
    #              'foreignField' : 'id_familia',
    #              'as' : 'integrantes'},
    #              '$limit': 1}]
    

    # pipeline = [
    #             {
    #             '$lookup':
    #                 {
    #                 'from': 'f8493a4e-f61e-441e-b802-d9b582a90f55',
    #                 'localField': 'id_familia',
    #                 'foreignField': 'id_familia',
    #                 'as': 'integrantes'
    #                 }
    #             }
    #             ,{'$limit': 5000}
    #             ]

    # result = list(familias.aggregate(pipeline))
    # #result2 = list(familias.find({}, {'_id':False}).limit(50000))

    # df = pd.DataFrame(result)

    # data = TransactionDB.from_DataFrame(df)

    # # rules = top_rules(data.string_representation)

    # # cars = createCARs(rules)

    # # classifier = M1Algorithm(cars, data).build()

    # cba = CBA(support=0.10, confidence=0.2, algorithm="m1")
    # re = cba.fit(data)

    # accuracy = cba.rule_model_accuracy(data) 

    # print(re)

    