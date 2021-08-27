import os
cwd = os.getcwd().split('/')[-1]
ProjectFolder = 'Fa21DiCOVA-II'
while cwd != ProjectFolder:
    os.chdir('..')  # move up a directory
    cwd = os.getcwd().split('/')[-1]
assert os.getcwd().split('/')[-1] == ProjectFolder