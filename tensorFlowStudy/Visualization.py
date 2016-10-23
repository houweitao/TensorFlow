# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: Visualization
# _date_ = 16/10/23 下午3:16

import  build_neural_network as build
import  matplotlib.pyplot as plt
import numpy as np

fig=plt.figure()
ax=fig.add_subplot(1,1,1)

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.5, x_data.shape)
y_data = np.square(x_data) - 0.5

ax.scatter(x_data,y_data)
plt.show()