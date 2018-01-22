import shelve
import dumbdbm


def dumbdbm_shelve(filename,flag="c"):
    return shelve.Shelf(dumbdbm.open(filename,flag))

#shelve.open('04_dc.slv')

out_shelf=dumbdbm_shelve("Markings3/16_mj.dumbdbm.slv")
in_shelf=shelve.open("Markings/16_mj.slv")

key_list=in_shelf.keys()
first = 0

for key in key_list:
    #print(key)
    #print(in_shelf[key])
    out_shelf[key]=in_shelf[key]

#for key in out_shelf.keys():
    #print(key)

out_shelf.close()
in_shelf.close()