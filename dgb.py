filelist = ['filelist1', 'filelist2']

filelist1 = ['1.csv', '2.csv']
filelist2 = ['3.csv', '4.csv']



for fidx,filename in enumerate(filelist):
    print(fidx, filename)
    print(eval(filename))