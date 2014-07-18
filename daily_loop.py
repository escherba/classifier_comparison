
import os



dirs = os.listdir('corpora/fixed_data_daily')
for i in range(len(dirs)):
    train_dir = dirs[i]
    for j in range(i+1,len(dirs)):
        test_dir = dirs[j]
        newpath = r'./loopresults/daily/'+train_dir+'/'+test_dir
        if not os.path.exists(newpath): os.makedirs(newpath)
        os.system('python classify.py' + ' --output '+ newpath +'/results.csv' +' --data_train ./corpora/fixed_data_daily/'+train_dir+' --data_test ./corpora/fixed_data_daily/'+ test_dir +  ' --top_terms 100')







