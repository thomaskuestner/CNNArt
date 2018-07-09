import json

with open('C:/Users/hansw/Desktop/Ma_code/PyQt_main/Markings/01_ab.json', 'r') as json_data:
    infos = json.load(json_data)
    print(infos)
    # for key in infos.keys():
    #     print(key)
    # # sepkey = 't1_tse_tra_Kopf_Motion_0003'
    #     layer = infos['layer']
    #     print(layer)
    #     for key1 in layer.keys():
    #         print(key1)

    # for key in layer['19_31_0'].keys():
    #     print(key)
#
# a = '1_22_3'
# b = list(a)
# print(b)



