# import json
#
# with open('C:/Users/hansw/Desktop/Ma_code/PyQt_main/Markings/01_ab.json', 'r') as json_data:
#     infos = json.load(json_data)
#     infos['name'] = ['label_1', 'label_2', 'label_3', 'label_4', 'label_5', 'label_6', 'label_7', 'label_8']
# with open('C:/Users/hansw/Desktop/Ma_code/PyQt_main/Markings/01_ab.json', 'w') as json_data:
#     json_data.write(json.dumps(infos))

# a = [(1,2,3),(4,2,8),(9,6,5),(2,7,6),(1,7,6),(4,2,3),(1,9,8),(1,2,7)]
# aa = ['123','428','965','276','176','423','198','127']
# s= sorted(aa, key = lambda e:e.__getitem__(0))
# print(s)
# b=sorted(s, key = lambda e:(e.__getitem__(0),e.__getitem__(1)))
# print(b)
# c=sorted(b, key = lambda e:(e.__getitem__(0),e.__getitem__(1),e.__getitem__(2)))
# print(c)

# key='123456'
# newkey = (key[0]+key[1], key[2], key[3], key[4]+key[5])
# print(newkey)
# a=newkey[0]+newkey[1]+newkey[2]+newkey[3]
# print(a)