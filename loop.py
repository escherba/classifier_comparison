
import os



for i in range(18,30):
    i = str(i)

    newpath = r'./loopresults/dec'+i 
    if not os.path.exists(newpath): os.makedirs(newpath)

    os.system('python classify.py' + ' --output '+ newpath +'/results.csv' +' --data_train ~/dev/py-nlp/var/corpora/livefyre/dec17'+' --data_test ~/dev/py-nlp/var/corpora/livefyre/dec'+ i +  ' --top_terms 100')





