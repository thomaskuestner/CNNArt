import json

labels = {'names':{'list':['label_1', 'label_2', 'label_3', 'label_4', 'label_5', 'label_6', 'label_7', 'label_8']},
          'colors':{'list':['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink']},
          'layer':{}}
wFile = json.dumps(labels)
with open('C:/Users/hansw/Videos/ma/Ma_code/PyQt_main/Markings/07_hs.json', 'w') as json_data:
    # with open('the desired path' + 'patient name.json', 'w') as json_data:
    json_data.write(wFile)