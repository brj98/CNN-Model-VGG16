import split_folder
dataSet = 'EyesDataBase/'
usingDataSet = 'UsingDataBase/'

splitFolder = split_folder.ratio(
    dataSet ,
    usingDataSet , 
    seed=42 , 
    ratio=(.8 , .2),
    group_prefix=None
)
