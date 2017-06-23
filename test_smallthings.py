grid_r = 3
grid_c = 3
for i in range(0,grid_r):
    for j in range(0,grid_c):
        if(i+1<grid_r):
            print(i, j, " to ", i + 1, j)
        if(j+1<grid_c):
            print(i, j, " to ", i , j + 1)