from gen_topo import *


if __name__ == '__main__':

    json_file = json.load(open("./database/data.json"))
    with open('./parallel/analytic.csv', newline='') as f:
       reader = csv.reader(f)
       result = list(reader)

    data_folder='component_data_random'
    target_folder='database'

    circuit_dic={}

    data={}

    good_count=0
    bad_count=0

    list_L={}

#    print(result[0])

    good_topology=[]

    flag=0

    prior_name=result[0][0]
    current_name=''

    for i in range(len(result)):

            current_name=result[i][0]
  
            print(prior_name,current_name)
            if prior_name != current_name:
               key_list=[]
               key_list = key_circuit(result[i][0],json_file)
               L=len(key_list)
               print(L)
               prior_name=current_name
               if str(L) in list_L:
                  list_L[str(L)]+=1
               else:
                   list_L[str(L)]=1

               if flag==1:
                   good_count+=L
                   flag=0
               else:
                   bad_count+=L
            try:
                vout=float(result[i][3])
                eff=float(result[i][4])
            except:
                continue

            if 40<vout<60 and eff>90:
                flag=1

              
print(good_count,bad_count)
print(list_L)


#       for key in key_list:
# 
#            for fn1 in json_file_iso:
#
#                print(fn,fn1)
#                   
#                if key==json_file_iso[fn1]["key"]:
#                    if key in good_topology:
#                        print("Good found")


#        tmp=json_file
#
#        key_list=[]
#
#        key_list = key_circuit(fn,tmp)
#
#        print(json_file[fn]["component_pool"])
#        for key in key_list:
#            for fn1 in json_file_iso:
#                if key==json_file_iso[fn1]["key"]:
#                   for item in result:
#                       if result
#
#            if key in circuit_dic:
#                circuit_dic[key].append(fn)
#            else:
#                circuit_dic[key]=[]
#                circuit_dic[key].append(fn)
#
#
#    count=0;
#
#    with open(target_folder+'/key.json', 'w') as outfile:
#            json.dump(circuit_dic, outfile)
#    outfile.close()
# 
#
#    filename_list=[]
#
#    json_file = json.load(open("./components_data_random/data.json"))
#    
#    for key in circuit_dic:
#
#            print(key, count)
#            filename=circuit_dic[key][0]
#
#            if filename not in filename_list:
#                filename_list.append(filename)
#            else:
#                continue
#
#            list_of_node=json_file[filename]['list_of_node']
#            list_of_edge=json_file[filename]['list_of_edge']
#            netlist=json_file[filename]['netlist']
#
##            print(netlist)
#            
#            name='Topo-'+ format(count, '04d')
#            topo_file=target_folder+'/topo/'+ name + '.png'
#
#            save_topo(list_of_node, list_of_edge,topo_file)
#
#            count=count+1
#
#            data[name] = {
#                            "key": key,
#                            "list_of_node":list_of_node,
#                            "list_of_edge":list_of_edge,
#                            "netlist":netlist
#                         }
#
#    
#    with open(target_folder+'/data.json', 'w') as outfile:
#            json.dump(data, outfile)
#    outfile.close()
#            
#    print(len(data))


                
                    

                
                


