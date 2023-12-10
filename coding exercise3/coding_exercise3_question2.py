import numpy as np
import math
import matplotlib.pyplot as plt

rho_test = np.array([[10, 11, 11.7, 13, 14, 15, 16, 17, 17, 17, 16.5, 17, 17, 16, 14.5, 14, 13]]).T
n = rho_test.shape[0]
theta_test = (math.pi/180) * np.linspace(0, 85, n).reshape(-1, 1)

x_test = rho_test * np.cos(theta_test)
y_test = rho_test * np.sin(theta_test)

threshold = 1

start_point_index = 0
end_point_index = 1

def calculate_distance(x,y,slope,b):
    distance = np.abs(slope * x + b - y) / np.sqrt(slope ** 2 + 1)
    return distance

threshold_exceed_indices = []
threshold_exceed_indices.append(start_point_index)
slopes = []
intercept = []

while (end_point_index < len(x_test) - 1) :
    slope,b = np.polyfit(x_test[start_point_index:end_point_index + 1,0],y_test[start_point_index:end_point_index + 1,0],1)
    end_point_index = end_point_index + 1
    while end_point_index < len(x_test):
        distance = calculate_distance(x_test[end_point_index],y_test[end_point_index],slope,b)
        if distance < threshold:
            slope,b = np.polyfit(x_test[start_point_index:end_point_index + 1,0],y_test[start_point_index:end_point_index + 1,0],1)
            end_point_index = end_point_index + 1
        else:
            threshold_exceed_indices.append(end_point_index)
            start_point_index = end_point_index
            end_point_index = end_point_index + 1
            break
    
            

print(threshold_exceed_indices)

plt.scatter(x_test, y_test, label='Cartesian Coordinates')

for i in range(len(threshold_exceed_indices)) :
    start_index = 0
    if threshold_exceed_indices[i] != 0:
        start_index = threshold_exceed_indices[i] 
   
    if i < len(threshold_exceed_indices) - 1:
        end_index = threshold_exceed_indices[i + 1] 
    else:
        end_index = len(x_test) - 1

    x_line = np.linspace(x_test[start_index, 0], x_test[end_index, 0], 100)
    slope,b = np.polyfit(x_test[start_index:end_index + 1,0],y_test[start_index:end_index + 1, 0],1)
    y_line = slope * x_line + b
    plt.plot(x_line, y_line, 'r-', label=f'Line {i} from Point {start_index } to {end_index}')

plt.legend()

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Detected Lines')

plt.show()


